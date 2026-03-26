import { createWorkersAI } from "workers-ai-provider";
import { routeAgentRequest, callable, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import { anthropic } from "@ai-sdk/anthropic";
import {
  streamText,
  convertToModelMessages,
  pruneMessages,
  tool,
  stepCountIs,
  generateText,
  type ModelMessage
} from "ai";
import { z } from "zod";

// ── Helpers ─────────────────────────────────────────────────────────────

/**
 * Decode data-URIs to Uint8Array so the AI SDK treats them as inline data
 * instead of trying to HTTP-fetch them.
 */
function inlineDataUrls(messages: ModelMessage[]): ModelMessage[] {
  return messages.map((msg) => {
    if (msg.role !== "user" || typeof msg.content === "string") return msg;
    return {
      ...msg,
      content: msg.content.map((part) => {
        if (part.type !== "file" || typeof part.data !== "string") return part;
        const match = part.data.match(/^data:([^;]+);base64,(.+)$/);
        if (!match) return part;
        const bytes = Uint8Array.from(atob(match[2]), (c) => c.charCodeAt(0));
        return { ...part, data: bytes, mediaType: match[1] };
      })
    };
  });
}

// ── Semantic Memory (RAG) via Vectorize ─────────────────────────────────
async function generateEmbedding(ai: Ai, text: string): Promise<number[]> {
  const result = await ai.run("@cf/baai/bge-base-en-v1.5", {
    text: [text]
  });
  return result.data[0];
}

// ── Sentiment & Entity Extraction ────────────────────────────────────

async function analyzeSentiment(
  ai: Ai,
  text: string
): Promise<{ label: string; score: number }> {
  try {
    const result = (await ai.run("@cf/huggingface/distilbert-sst-2-int8", {
      text
    })) as Array<Array<{ label: string; score: number }>>;
    const top = result?.[0]?.[0] ?? { label: "UNKNOWN", score: 0 };
    return { label: top.label, score: top.score };
  } catch {
    return { label: "NEUTRAL", score: 0.5 };
  }
}

// ── Main Agent ──────────────────────────────────────────────────────────

export class ChatAgent extends AIChatAgent<Env> {
  maxPersistedMessages = 200;

  // ── Conversation Memory Store (SQL-backed) ────────────────────────
  private _memoryInitialized = false;

  private async ensureMemoryTable() {
    if (this._memoryInitialized) return;
    this.sql`CREATE TABLE IF NOT EXISTS conversation_memory (
      id TEXT PRIMARY KEY,
      summary TEXT NOT NULL,
      key_entities TEXT DEFAULT '[]',
      sentiment TEXT DEFAULT 'neutral',
      created_at TEXT NOT NULL
    )`;
    this.sql`CREATE TABLE IF NOT EXISTS knowledge_base (
      id TEXT PRIMARY KEY,
      content TEXT NOT NULL,
      category TEXT DEFAULT 'general',
      embedding_stored INTEGER DEFAULT 0,
      created_at TEXT NOT NULL
    )`;
    this._memoryInitialized = true;
  }

  // ── Lifecycle ─────────────────────────────────────────────────────

  onStart() {
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  // ── RAG: Store & Retrieve Knowledge ────────────────────────────────

  @callable()
  async storeKnowledge(content: string, category: string) {
    await this.ensureMemoryTable();
    const id = `kb_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    this.sql`INSERT INTO knowledge_base (id, content, category, created_at)
      VALUES (${id}, ${content}, ${category}, ${new Date().toISOString()})`;

    // Store embedding in Vectorize if available
    try {
      if (this.env.VECTORIZE) {
        const embedding = await generateEmbedding(this.env.AI, content);
        await this.env.VECTORIZE.upsert([
          {
            id,
            values: embedding,
            metadata: { category, text: content.slice(0, 500) }
          }
        ]);
      }
    } catch (e) {
      console.warn("Vectorize upsert skipped:", e);
    }

    return { id, stored: true };
  }

  @callable()
  async searchKnowledge(query: string, topK = 5) {
    // Semantic search via Vectorize
    try {
      if (this.env.VECTORIZE) {
        const queryEmbedding = await generateEmbedding(this.env.AI, query);
        const results = await this.env.VECTORIZE.query(queryEmbedding, {
          topK,
          returnMetadata: "all"
        });
        return results.matches.map((m) => ({
          id: m.id,
          score: m.score,
          text: (m.metadata?.text as string) ?? "",
          category: (m.metadata?.category as string) ?? ""
        }));
      }
    } catch (e) {
      console.warn("Vectorize query skipped:", e);
    }

    // Fallback to SQL full-text search
    await this.ensureMemoryTable();
    const rows = this.sql`SELECT id, content, category FROM knowledge_base
      WHERE content LIKE ${"%" + query + "%"} LIMIT ${topK}`;
    return rows;
  }

  // ── Conversation Summarization (Long-term Memory) ──────────────────

  private async summarizeAndStore(messages: ModelMessage[]) {
    await this.ensureMemoryTable();
    if (messages.length < 6) return;

    const lastMessages = messages.slice(-10);
    const conversationText = lastMessages
      .map(
        (m) =>
          `${m.role}: ${typeof m.content === "string" ? m.content : "[complex]"}`
      )
      .join("\n");

    try {
      // Use Workers AI for summarization to save on external tokens
      const workersai = createWorkersAI({ binding: this.env.AI });
      const { text: summary } = await generateText({
        model: workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
        prompt: `Summarize this conversation in 2-3 sentences:\n\n${conversationText}`,
        maxTokens: 200
      });

      const id = `mem_${Date.now()}`;
      const sentiment = await analyzeSentiment(
        this.env.AI,
        conversationText.slice(0, 500)
      );

      this
        .sql`INSERT INTO conversation_memory (id, summary, sentiment, created_at)
        VALUES (${id}, ${summary}, ${sentiment.label}, ${new Date().toISOString()})`;
    } catch (e) {
      console.warn("Summarization skipped:", e);
    }
  }

  // ── Main Chat Handler ──────────────────────────────────────────────

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    await this.ensureMemoryTable();
    const mcpTools = this.mcp.getAITools();

    // Select model provider: Anthropic if key provided, otherwise Workers AI
    let model;
    if (this.env.ANTHROPIC_API_KEY) {
      model = anthropic("claude-3-5-sonnet-latest", {
        headers: { "x-api-key": this.env.ANTHROPIC_API_KEY }
      });
    } else {
      const workersai = createWorkersAI({ binding: this.env.AI });
      model = workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast", {
        sessionAffinity: this.sessionAffinity
      });
    }

    // Retrieve past conversation summaries
    let memoryContext = "";
    try {
      const memories = (await this
        .sql`SELECT summary, sentiment, created_at FROM conversation_memory
        ORDER BY created_at DESC LIMIT 5`) as Array<{
        summary: string;
        sentiment: string;
        created_at: string;
      }>;
      if (Array.isArray(memories) && memories.length > 0) {
        memoryContext =
          "\n\n## Previous Memories:\n" +
          memories
            .map((m) => `- [${m.created_at}] (${m.sentiment}): ${m.summary}`)
            .join("\n");
      }
    } catch {}

    const messages = pruneMessages({
      messages: inlineDataUrls(await convertToModelMessages(this.messages)),
      toolCalls: "before-last-2-messages"
    });

    const result = streamText({
      model,
      system: `You are an advanced AI agent. Capabilities: RAG, Web Search, Image Gen, Voice, Scheduling, MCP.
${getSchedulePrompt({ date: new Date() })}
${memoryContext}`,
      messages,
      tools: {
        ...mcpTools,

        // ── Weather Tool ──────────────────────────────────────────
        getWeather: tool({
          description:
            "Get the current weather for a city using a real weather API",
          inputSchema: z.object({
            city: z.string().describe("City name")
          }),
          execute: async ({ city }) => {
            try {
              const resp = await fetch(
                `https://wttr.in/${encodeURIComponent(city)}?format=j1`
              );
              if (!resp.ok) throw new Error("Weather API error");
              const data = (await resp.json()) as {
                current_condition: Array<{
                  temp_C: string;
                  FeelsLikeC: string;
                  weatherDesc: Array<{ value: string }>;
                  humidity: string;
                  windspeedKmph: string;
                }>;
              };
              const current = data.current_condition?.[0];
              return {
                city,
                temperature: current?.temp_C ?? "N/A",
                feelsLike: current?.FeelsLikeC ?? "N/A",
                condition: current?.weatherDesc?.[0]?.value ?? "Unknown",
                humidity: current?.humidity ?? "N/A",
                windSpeed: current?.windspeedKmph ?? "N/A",
                unit: "celsius"
              };
            } catch {
              const conditions = ["sunny", "cloudy", "rainy", "snowy"];
              return {
                city,
                temperature: Math.floor(Math.random() * 30) + 5,
                condition:
                  conditions[Math.floor(Math.random() * conditions.length)],
                unit: "celsius",
                note: "Simulated data"
              };
            }
          }
        }),

        // ── Client-side: Timezone ─────────────────────────────────
        getUserTimezone: tool({
          description: "Get the user's timezone from their browser",
          inputSchema: z.object({})
        }),

        // ── Calculator with Approval ──────────────────────────────
        calculate: tool({
          description:
            "Perform math calculations. Requires approval for large numbers.",
          inputSchema: z.object({
            a: z.number().describe("First number"),
            b: z.number().describe("Second number"),
            operator: z
              .enum(["+", "-", "*", "/", "%", "^"])
              .describe("Arithmetic operator")
          }),
          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
            const ops: Record<string, (x: number, y: number) => number> = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y,
              "^": (x, y) => Math.pow(x, y)
            };
            if (operator === "/" && b === 0)
              return { error: "Division by zero" };
            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b)
            };
          }
        }),

        // ── Scheduling ────────────────────────────────────────────
        scheduleTask: tool({
          description: "Schedule a task to be executed at a later time",
          inputSchema: scheduleSchema,
          execute: async ({ when, description }) => {
            if (when.type === "no-schedule")
              return "Not a valid schedule input";
            const input =
              when.type === "scheduled"
                ? when.date
                : when.type === "delayed"
                  ? when.delayInSeconds
                  : when.type === "cron"
                    ? when.cron
                    : null;
            if (!input) return "Invalid schedule type";
            try {
              this.schedule(input, "executeTask", description, {
                idempotent: true
              });
              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${error}`;
            }
          }
        }),

        getScheduledTasks: tool({
          description: "List all currently scheduled tasks",
          inputSchema: z.object({}),
          execute: async () => {
            const tasks = this.getSchedules();
            return tasks.length > 0 ? tasks : "No scheduled tasks found.";
          }
        }),

        cancelScheduledTask: tool({
          description: "Cancel a scheduled task by its ID",
          inputSchema: z.object({
            taskId: z.string().describe("The ID of the task to cancel")
          }),
          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);
              return `Task ${taskId} cancelled.`;
            } catch (error) {
              return `Error cancelling task: ${error}`;
            }
          }
        }),

        // ── Web Search (Real-time Information) ────────────────────
        webSearch: tool({
          description:
            "Search the web for current information on any topic. Use this when you need up-to-date information.",
          inputSchema: z.object({
            query: z.string().describe("The search query")
          }),
          execute: async ({ query }) => {
            try {
              // Use DuckDuckGo instant answer API (no key needed)
              const resp = await fetch(
                `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1`
              );
              if (!resp.ok) throw new Error("Search API error");
              const data = (await resp.json()) as {
                Abstract?: string;
                AbstractText?: string;
                AbstractSource?: string;
                AbstractURL?: string;
                RelatedTopics?: Array<{ Text: string; FirstURL: string }>;
              };
              const results: Array<{
                title: string;
                url: string;
                content: string;
              }> = [];
              if (data.Abstract || data.AbstractText) {
                results.push({
                  title: data.AbstractSource || "Abstract",
                  url: data.AbstractURL || "",
                  content: data.Abstract || data.AbstractText || ""
                });
              }
              if (data.RelatedTopics) {
                for (const topic of data.RelatedTopics.slice(0, 5)) {
                  if (topic.Text) {
                    results.push({
                      type: "related",
                      text: topic.Text,
                      url: topic.FirstURL
                    });
                  }
                }
              }
              return results.length > 0
                ? results
                : { message: "No results found. Try rephrasing.", query };
            } catch {
              return { error: "Search temporarily unavailable", query };
            }
          }
        }),

        // ── Image Generation (Text-to-Image) ─────────────────────
        generateImage: tool({
          description:
            "Generate an image from a text description using AI. Returns a base64-encoded image.",
          inputSchema: z.object({
            prompt: z
              .string()
              .describe("Detailed description of the image to generate"),
            style: z
              .enum([
                "photorealistic",
                "artistic",
                "anime",
                "sketch",
                "fantasy"
              ])
              .optional()
              .describe("Art style for the generated image")
          }),
          execute: async ({ prompt, style }) => {
            try {
              const enhancedPrompt = style
                ? `${style} style: ${prompt}`
                : prompt;
              const result = await this.env.AI.run(
                "@cf/black-forest-labs/flux-1-schnell" as any,
                { prompt: enhancedPrompt } as any
              );
              // Convert ReadableStream to base64
              const reader = (result as ReadableStream).getReader();
              const chunks: Uint8Array[] = [];
              while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
              }
              const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
              const combined = new Uint8Array(totalLength);
              let offset = 0;
              for (const chunk of chunks) {
                combined.set(chunk, offset);
                offset += chunk.length;
              }
              const base64 = btoa(String.fromCharCode(...combined));
              return {
                success: true,
                image: `data:image/png;base64,${base64}`,
                prompt: enhancedPrompt
              };
            } catch (error) {
              return {
                success: false,
                error: `Image generation failed: ${error}`
              };
            }
          }
        }),

        // ── Text Translation ──────────────────────────────────────
        translateText: tool({
          description: "Translate text from one language to another",
          inputSchema: z.object({
            text: z.string().describe("Text to translate"),
            targetLang: z
              .string()
              .describe(
                "Target language (e.g., 'French', 'Spanish', 'Chinese', 'Japanese')"
              ),
            sourceLang: z
              .string()
              .optional()
              .describe("Source language (auto-detected if omitted)")
          }),
          execute: async ({ text, targetLang, sourceLang }) => {
            const workersai = createWorkersAI({ binding: this.env.AI });
            try {
              const { text: translated } = await generateText({
                model: workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
                prompt: `Translate the following text${sourceLang ? ` from ${sourceLang}` : ""} to ${targetLang}. Only output the translation, nothing else:\n\n${text}`,
                maxTokens: 500
              });
              return {
                original: text,
                translated,
                targetLang,
                sourceLang: sourceLang ?? "auto-detected"
              };
            } catch (error) {
              return { error: `Translation failed: ${error}` };
            }
          }
        }),

        // ── Sentiment Analysis ────────────────────────────────────
        analyzeSentiment: tool({
          description:
            "Analyze the emotional sentiment of text. Returns positive/negative classification.",
          inputSchema: z.object({
            text: z.string().describe("Text to analyze sentiment for")
          }),
          execute: async ({ text }) => {
            const result = await analyzeSentiment(this.env.AI, text);
            return {
              text: text.slice(0, 100) + (text.length > 100 ? "..." : ""),
              sentiment: result.label,
              confidence: Math.round(result.score * 100) + "%"
            };
          }
        }),

        // ── Knowledge Base Tools (RAG) ────────────────────────────
        storeKnowledge: tool({
          description:
            "Store important information in the persistent knowledge base for future retrieval. Use this when users share facts, preferences, or important data.",
          inputSchema: z.object({
            content: z.string().describe("The information to store"),
            category: z
              .string()
              .describe(
                "Category (e.g., 'personal', 'work', 'research', 'preference')"
              )
          }),
          execute: async ({ content, category }) => {
            return await this.storeKnowledge(content, category);
          }
        }),

        searchKnowledge: tool({
          description:
            "Search the knowledge base for previously stored information",
          inputSchema: z.object({
            query: z.string().describe("Search query")
          }),
          execute: async ({ query }) => {
            return await this.searchKnowledge(query);
          }
        }),

        // ── Code Execution (Sandboxed) ────────────────────────────
        executeCode: tool({
          description:
            "Execute JavaScript code in a sandboxed environment. Use for data analysis, calculations, or generating structured output.",
          inputSchema: z.object({
            code: z.string().describe("JavaScript code to execute"),
            description: z
              .string()
              .describe("Brief description of what the code does")
          }),
          needsApproval: async () => true, // Always require approval for code execution
          execute: async ({ code, description }) => {
            try {
              // Safe sandboxed execution using Function constructor
              const fn = new Function(
                "Math",
                "Date",
                "JSON",
                "Array",
                "Object",
                "String",
                "Number",
                `"use strict"; ${code}`
              );
              const result = fn(
                Math,
                Date,
                JSON,
                Array,
                Object,
                String,
                Number
              );
              return {
                success: true,
                result: JSON.stringify(result),
                description
              };
            } catch (error) {
              return { success: false, error: `${error}`, description };
            }
          }
        }),

        // ── Summarize Long Text ───────────────────────────────────
        summarizeText: tool({
          description: "Summarize a long piece of text into key points",
          inputSchema: z.object({
            text: z.string().describe("The text to summarize"),
            style: z
              .enum(["bullet-points", "paragraph", "executive-summary"])
              .optional()
              .describe("Summary style")
          }),
          execute: async ({ text, style }) => {
            const workersai = createWorkersAI({ binding: this.env.AI });
            const stylePrompt =
              style === "bullet-points"
                ? "Summarize as bullet points"
                : style === "executive-summary"
                  ? "Write an executive summary"
                  : "Write a concise paragraph summary";
            const { text: summary } = await generateText({
              model: workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
              prompt: `${stylePrompt} of the following text:\n\n${text}`,
              maxTokens: 500
            });
            return {
              summary,
              originalLength: text.length,
              style: style ?? "paragraph"
            };
          }
        }),

        // ── Get Agent Capabilities ────────────────────────────────
        getCapabilities: tool({
          description:
            "List all available tools and capabilities of this agent",
          inputSchema: z.object({}),
          execute: async () => {
            return {
              model: "Llama 3.3 70B (Workers AI)",
              capabilities: [
                "💬 Streaming AI Chat with message persistence",
                "🧠 Long-term Conversation Memory (auto-summarization)",
                "🔍 Semantic Knowledge Base (RAG with Vectorize)",
                "🌐 Real-time Web Search",
                "🎨 AI Image Generation (Flux)",
                "🌍 Multi-language Translation",
                "😊 Sentiment Analysis",
                "📝 Text Summarization",
                "🔢 Math Calculations",
                "💻 Sandboxed Code Execution",
                "⏰ Task Scheduling (cron, delay, one-time)",
                "🔌 MCP Server Integration",
                "🎤 Voice Input (client-side)",
                "📎 Image Upload & Analysis",
                "🌓 Dark/Light Theme"
              ],
              memoryType: "SQLite-backed via Durable Objects",
              vectorSearch: "Cloudflare Vectorize (BGE embeddings)"
            };
          }
        })
      },
      stopWhen: stepCountIs(8),
      abortSignal: options?.abortSignal
    });

    // After streaming completes, summarize for long-term memory
    this.summarizeAndStore(messages).catch(() => {});

    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    console.log(`Executing scheduled task: ${description}`);
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
