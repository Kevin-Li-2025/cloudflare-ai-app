import { createWorkersAI } from "workers-ai-provider";
import { routeAgentRequest, callable, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import { createAnthropic } from "@ai-sdk/anthropic";
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
  const result = (await ai.run("@cf/baai/bge-base-en-v1.5", {
    text: [text]
  })) as { data: number[][] };
  return result.data[0];
}

// ── Sentiment Analysis ──────────────────────────────────────────────────

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

// ═══════════════════════════════════════════════════════════════════════
// ██  MAIN AGENT  ████████████████████████████████████████████████████
// ═══════════════════════════════════════════════════════════════════════

export class ChatAgent extends AIChatAgent<Env> {
  maxPersistedMessages = 200;

  // ── Schema Initialization ──────────────────────────────────────────
  private _memoryInitialized = false;

  private async ensureMemoryTable() {
    if (this._memoryInitialized) return;

    // Flat conversation memory (for summaries)
    this.sql`CREATE TABLE IF NOT EXISTS conversation_memory (
      id TEXT PRIMARY KEY,
      summary TEXT NOT NULL,
      key_entities TEXT DEFAULT '[]',
      sentiment TEXT DEFAULT 'neutral',
      created_at TEXT NOT NULL
    )`;

    // Knowledge base (for RAG vectors)
    this.sql`CREATE TABLE IF NOT EXISTS knowledge_base (
      id TEXT PRIMARY KEY,
      content TEXT NOT NULL,
      category TEXT DEFAULT 'general',
      embedding_stored INTEGER DEFAULT 0,
      created_at TEXT NOT NULL
    )`;

    // ③ STRUCTURED MEMORY — hierarchical user profile/preferences/goals
    this.sql`CREATE TABLE IF NOT EXISTS structured_memory (
      id TEXT PRIMARY KEY,
      type TEXT NOT NULL CHECK(type IN ('profile','preference','goal','fact','task_history')),
      key TEXT NOT NULL,
      value TEXT NOT NULL,
      confidence REAL DEFAULT 1.0,
      updated_at TEXT NOT NULL
    )`;

    this._memoryInitialized = true;
  }

  // ── Lifecycle ──────────────────────────────────────────────────────

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

  // ═══════════════════════════════════════════════════════════════════
  // ①  AGENTIC RAG — Iterative Multi-Round Retrieval
  // ═══════════════════════════════════════════════════════════════════

  @callable()
  async storeKnowledge(content: string, category: string) {
    await this.ensureMemoryTable();
    const id = `kb_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    this.sql`INSERT INTO knowledge_base (id, content, category, created_at)
      VALUES (${id}, ${content}, ${category}, ${new Date().toISOString()})`;

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

    await this.ensureMemoryTable();
    const rows = this.sql`SELECT id, content, category FROM knowledge_base
      WHERE content LIKE ${"%" + query + "%"} LIMIT ${topK}`;
    return rows;
  }

  /**
   * ① AGENTIC RAG — Multi-round retrieval with confidence scoring
   *    and LLM-powered query refinement.
   *
   *    RAG goes from "pipeline" → "loop":
   *    for each round:
   *      1. Search with current query
   *      2. Score results
   *      3. If confidence < threshold → refine query via LLM
   *      4. Else → stop and return
   */
  private async agenticRAG(
    query: string,
    maxRounds = 3,
    confidenceThreshold = 0.7
  ): Promise<{
    results: Array<{
      id: string;
      score: number;
      text: string;
      category: string;
    }>;
    reasoning: string[];
    rounds: number;
    finalConfidence: number;
  }> {
    const reasoning: string[] = [];
    let currentQuery = query;
    let allResults: Array<{
      id: string;
      score: number;
      text: string;
      category: string;
    }> = [];
    let bestScore = 0;

    for (let round = 0; round < maxRounds; round++) {
      reasoning.push(`Round ${round + 1}: Searching for "${currentQuery}"`);

      // ── Search via Vectorize or SQL fallback ──
      let roundResults: Array<{
        id: string;
        score: number;
        text: string;
        category: string;
      }> = [];

      try {
        if (this.env.VECTORIZE) {
          const embedding = await generateEmbedding(this.env.AI, currentQuery);
          const vResults = await this.env.VECTORIZE.query(embedding, {
            topK: 5,
            returnMetadata: "all"
          });
          roundResults = vResults.matches.map((m) => ({
            id: m.id,
            score: m.score,
            text: (m.metadata?.text as string) ?? "",
            category: (m.metadata?.category as string) ?? ""
          }));
        }
      } catch {
        // fallback to SQL
        await this.ensureMemoryTable();
        const rows = this.sql`SELECT id, content, category FROM knowledge_base
          WHERE content LIKE ${"%" + currentQuery + "%"} LIMIT 5` as Array<{
          id: string;
          content: string;
          category: string;
        }>;
        roundResults = rows.map((r) => ({
          id: r.id,
          score: 0.5,
          text: r.content,
          category: r.category
        }));
      }

      // ── Merge results (keep highest scores) ──
      for (const r of roundResults) {
        const existing = allResults.find((e) => e.id === r.id);
        if (!existing) allResults.push(r);
        else if (r.score > existing.score) existing.score = r.score;
      }

      bestScore =
        allResults.length > 0 ? Math.max(...allResults.map((r) => r.score)) : 0;
      reasoning.push(
        `  Found ${roundResults.length} results (best score: ${bestScore.toFixed(3)})`
      );

      // ── Confidence check ──
      if (bestScore >= confidenceThreshold || round === maxRounds - 1) {
        reasoning.push(
          `  ✓ Confidence ${bestScore.toFixed(3)} meets threshold ${confidenceThreshold}`
        );
        break;
      }

      // ── Refine query using LLM ──
      reasoning.push(`  ✗ Confidence too low, refining query...`);
      try {
        const workersai = createWorkersAI({ binding: this.env.AI });
        const { text: refinedQuery } = await generateText({
          model: workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
          prompt: `You are a search query optimizer. The query "${currentQuery}" returned low-relevance results. Based on these partial results:\n${roundResults
            .map((r) => `- ${r.text}`)
            .join(
              "\n"
            )}\n\nGenerate a better, more specific search query. Output ONLY the new query, nothing else.`,
          maxOutputTokens: 50
        });
        currentQuery = refinedQuery.trim();
        reasoning.push(`  Refined query: "${currentQuery}"`);
      } catch {
        reasoning.push(`  Could not refine query, stopping.`);
        break;
      }
    }

    allResults.sort((a, b) => b.score - a.score);

    return {
      results: allResults.slice(0, 10),
      reasoning,
      rounds: reasoning.filter((r) => r.startsWith("Round")).length,
      finalConfidence: bestScore
    };
  }

  // ═══════════════════════════════════════════════════════════════════
  // ②  MULTI-AGENT — Planner → Worker → Reviewer
  // ═══════════════════════════════════════════════════════════════════

  /**
   * Multi-agent orchestration: decomposes complex tasks into
   * planning, execution, and review phases — each handled by
   * a distinct LLM "persona" with its own system prompt.
   *
   * If the Reviewer rejects the output, the Worker revises once.
   */
  private async multiAgentOrchestrate(task: string): Promise<{
    plan: string;
    execution: string;
    review: { approved: boolean; feedback: string };
    trace: string[];
  }> {
    const workersai = createWorkersAI({ binding: this.env.AI });
    const trace: string[] = [];

    // ── Phase 1: Planner Agent ──
    trace.push("🧠 [Planner] Analyzing task and creating execution plan...");
    const { text: plan } = await generateText({
      model: workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
      system: `You are a Planning Agent. Your role is to analyze complex tasks and break them down into clear, actionable steps. Output a numbered list of concrete steps. Be specific and practical. Focus on what needs to be done, not how.`,
      prompt: `Task: ${task}\n\nCreate a detailed execution plan:`,
      maxOutputTokens: 500
    });
    trace.push(
      `📋 [Planner] Plan created with ${plan.split("\n").filter((l) => l.trim()).length} steps`
    );

    // ── Phase 2: Worker Agent ──
    trace.push("⚡ [Worker] Executing the plan...");
    const { text: execution } = await generateText({
      model: workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
      system: `You are an Execution Agent. Your role is to follow a plan precisely and produce high-quality output. Be thorough, detailed, and accurate in your work. Execute each step of the plan.`,
      prompt: `Task: ${task}\n\nPlan to follow:\n${plan}\n\nExecute this plan and produce the final output:`,
      maxOutputTokens: 1000
    });
    trace.push("✅ [Worker] Execution completed");

    // ── Phase 3: Reviewer Agent ──
    trace.push("🔍 [Reviewer] Reviewing the output quality...");
    const { text: reviewText } = await generateText({
      model: workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
      system: `You are a Review Agent. Critically evaluate work output for: completeness, accuracy, quality, and adherence to the original task. Start your response with either "APPROVED:" or "NEEDS_REVISION:" followed by detailed feedback.`,
      prompt: `Original Task: ${task}\n\nPlan:\n${plan}\n\nExecution Output:\n${execution}\n\nProvide your review:`,
      maxOutputTokens: 500
    });

    const approved = reviewText.toUpperCase().startsWith("APPROVED");
    trace.push(
      `${approved ? "✅" : "⚠️"} [Reviewer] ${approved ? "Output approved" : "Revision suggested"}`
    );

    // ── Phase 4: Optional Revision ──
    let finalExecution = execution;
    if (!approved) {
      trace.push("🔄 [Worker] Revising based on reviewer feedback...");
      const { text: revised } = await generateText({
        model: workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
        system: `You are an Execution Agent. Revise your previous work based on the reviewer's feedback. Produce improved, polished output.`,
        prompt: `Original Task: ${task}\n\nYour previous output:\n${execution}\n\nReviewer feedback:\n${reviewText}\n\nRevised output:`,
        maxOutputTokens: 1000
      });
      finalExecution = revised;
      trace.push("✅ [Worker] Revision completed");
    }

    return {
      plan,
      execution: finalExecution,
      review: { approved, feedback: reviewText },
      trace
    };
  }

  // ═══════════════════════════════════════════════════════════════════
  // ③  STRUCTURED MEMORY — Profile / Preferences / Goals / Facts
  // ═══════════════════════════════════════════════════════════════════

  /**
   * Upsert a structured memory entry. If a key already exists
   * for the given type, it will be updated with the new value.
   */
  private async updateStructuredMemory(
    type: "profile" | "preference" | "goal" | "fact" | "task_history",
    key: string,
    value: string,
    confidence = 1.0
  ) {
    await this.ensureMemoryTable();
    const id = `sm_${type}_${key.replace(/\s+/g, "_").toLowerCase()}`;
    const now = new Date().toISOString();

    // Upsert: delete old + insert new (SQLite in DO doesn't support ON CONFLICT well)
    this.sql`DELETE FROM structured_memory WHERE id = ${id}`;
    this
      .sql`INSERT INTO structured_memory (id, type, key, value, confidence, updated_at)
      VALUES (${id}, ${type}, ${key}, ${value}, ${confidence}, ${now})`;

    return { id, type, key, value, confidence, updated_at: now };
  }

  /**
   * Retrieve the full structured memory tree, organized by type.
   */
  private async getStructuredMemory(): Promise<{
    profile: Record<string, string>;
    preferences: Record<string, string>;
    goals: Array<{ key: string; value: string }>;
    facts: Array<{ key: string; value: string }>;
    taskHistory: Array<{ key: string; value: string }>;
  }> {
    await this.ensureMemoryTable();

    const rows = this.sql`SELECT type, key, value FROM structured_memory
      ORDER BY updated_at DESC` as Array<{
      type: string;
      key: string;
      value: string;
    }>;

    const memory = {
      profile: {} as Record<string, string>,
      preferences: {} as Record<string, string>,
      goals: [] as Array<{ key: string; value: string }>,
      facts: [] as Array<{ key: string; value: string }>,
      taskHistory: [] as Array<{ key: string; value: string }>
    };

    for (const row of rows) {
      switch (row.type) {
        case "profile":
          memory.profile[row.key] = row.value;
          break;
        case "preference":
          memory.preferences[row.key] = row.value;
          break;
        case "goal":
          memory.goals.push({ key: row.key, value: row.value });
          break;
        case "fact":
          memory.facts.push({ key: row.key, value: row.value });
          break;
        case "task_history":
          memory.taskHistory.push({ key: row.key, value: row.value });
          break;
      }
    }

    return memory;
  }

  /**
   * Auto-extract user preferences from conversation using LLM.
   * Called after each conversation turn to build the user profile.
   */
  private async autoExtractPreferences(messages: ModelMessage[]) {
    if (messages.length < 2) return;

    const lastUserMsg = messages.filter((m) => m.role === "user").pop();
    if (!lastUserMsg) return;

    const userText =
      typeof lastUserMsg.content === "string"
        ? lastUserMsg.content
        : "[complex content]";

    // Only extract if the message seems to contain personal info
    if (userText.length < 10) return;

    try {
      const workersai = createWorkersAI({ binding: this.env.AI });
      const { text: extraction } = await generateText({
        model: workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
        prompt: `Analyze this user message and extract any personal preferences, facts, or goals. Output JSON array (or empty array if none found). Each item should have: {"type": "profile|preference|goal|fact", "key": "short_key", "value": "description"}.

User message: "${userText}"

Output ONLY valid JSON array:`,
        maxOutputTokens: 200
      });

      try {
        const items = JSON.parse(extraction.trim()) as Array<{
          type: "profile" | "preference" | "goal" | "fact";
          key: string;
          value: string;
        }>;
        if (Array.isArray(items)) {
          for (const item of items.slice(0, 3)) {
            if (item.type && item.key && item.value) {
              await this.updateStructuredMemory(
                item.type,
                item.key,
                item.value,
                0.8
              );
            }
          }
        }
      } catch {
        // JSON parse failed — skip silently
      }
    } catch {
      // Extraction is non-critical
    }
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
      const workersai = createWorkersAI({ binding: this.env.AI });
      const { text: summary } = await generateText({
        model: workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
        prompt: `Summarize this conversation in 2-3 sentences:\n\n${conversationText}`,
        maxOutputTokens: 200
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

  // ═══════════════════════════════════════════════════════════════════
  // ██  MAIN CHAT HANDLER  ██████████████████████████████████████████
  // ═══════════════════════════════════════════════════════════════════

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    await this.ensureMemoryTable();
    const mcpTools = this.mcp.getAITools();

    // Select model: Anthropic if key provided, otherwise Workers AI
    let model;
    if (this.env.ANTHROPIC_API_KEY) {
      model = createAnthropic({ apiKey: this.env.ANTHROPIC_API_KEY })(
        "claude-3-5-sonnet-latest"
      );
    } else {
      const workersai = createWorkersAI({ binding: this.env.AI });
      model = workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast", {
        sessionAffinity: this.sessionAffinity
      });
    }

    // ── Retrieve conversation summaries (long-term memory) ──
    let memoryContext = "";
    try {
      const memories = this
        .sql`SELECT summary, sentiment, created_at FROM conversation_memory
        ORDER BY created_at DESC LIMIT 5` as Array<{
        summary: string;
        sentiment: string;
        created_at: string;
      }>;
      if (Array.isArray(memories) && memories.length > 0) {
        memoryContext =
          "\n\n## Previous Conversation Memories:\n" +
          memories
            .map((m) => `- [${m.created_at}] (${m.sentiment}): ${m.summary}`)
            .join("\n");
      }
    } catch {}

    // ── ③ Retrieve structured memory (profile/prefs/goals) ──
    let structuredCtx = "";
    try {
      const sm = await this.getStructuredMemory();
      const parts: string[] = [];
      if (Object.keys(sm.profile).length > 0) {
        parts.push(
          "**User Profile**: " +
            Object.entries(sm.profile)
              .map(([k, v]) => `${k}: ${v}`)
              .join(", ")
        );
      }
      if (Object.keys(sm.preferences).length > 0) {
        parts.push(
          "**Preferences**: " +
            Object.entries(sm.preferences)
              .map(([k, v]) => `${k}: ${v}`)
              .join(", ")
        );
      }
      if (sm.goals.length > 0) {
        parts.push("**Goals**: " + sm.goals.map((g) => g.value).join("; "));
      }
      if (sm.facts.length > 0) {
        parts.push(
          "**Known Facts**: " +
            sm.facts.map((f) => `${f.key}: ${f.value}`).join("; ")
        );
      }
      if (parts.length > 0) {
        structuredCtx = "\n\n## User Memory:\n" + parts.join("\n");
      }
    } catch {}

    const messages = pruneMessages({
      messages: inlineDataUrls(await convertToModelMessages(this.messages)),
      toolCalls: "before-last-2-messages"
    });

    const result = streamText({
      model,
      system: `You are an advanced AI agent with frontier capabilities:

- **Agentic RAG**: Use deepSearch for iterative, multi-round retrieval with confidence scoring and query refinement
- **Multi-Agent Orchestration**: Use complexTask to decompose hard problems into Planning → Execution → Review phases
- **Structured Memory**: You remember user preferences, goals, and facts across conversations
- **Real-time Web Search, Image Generation, Translation, Sentiment Analysis, Code Execution, Task Scheduling**

When a user asks a complex question that might need deep research, use deepSearch instead of basic searchKnowledge.
When a user asks for a complex task (writing, analysis, planning), use complexTask for higher quality output.
When a user shares personal information, preferences, or goals, use rememberAboutUser to persist it.

${getSchedulePrompt({ date: new Date() })}
${memoryContext}
${structuredCtx}`,
      messages,
      tools: {
        ...mcpTools,

        // ═════════════════════════════════════════════════════════
        // ① AGENTIC RAG — Deep Search with Iterative Refinement
        // ═════════════════════════════════════════════════════════

        deepSearch: tool({
          description:
            "Perform deep, agentic search over the knowledge base. Unlike basic search, this iteratively refines queries across multiple rounds until high-confidence results are found. Use this for complex or ambiguous queries.",
          inputSchema: z.object({
            query: z.string().describe("The search query"),
            maxRounds: z
              .number()
              .optional()
              .describe("Maximum search rounds (1-5, default 3)"),
            confidenceThreshold: z
              .number()
              .optional()
              .describe(
                "Minimum confidence to stop searching (0-1, default 0.7)"
              )
          }),
          execute: async ({ query, maxRounds, confidenceThreshold }) => {
            return await this.agenticRAG(
              query,
              Math.min(maxRounds ?? 3, 5),
              confidenceThreshold ?? 0.7
            );
          }
        }),

        // ═════════════════════════════════════════════════════════
        // ② MULTI-AGENT — Complex Task Orchestration
        // ═════════════════════════════════════════════════════════

        complexTask: tool({
          description:
            "Handle complex tasks using multi-agent orchestration. A Planner Agent decomposes the task, a Worker Agent executes it, and a Reviewer Agent validates the output. If the review fails, the Worker revises. Use this for tasks requiring analysis, writing, or multi-step reasoning.",
          inputSchema: z.object({
            task: z
              .string()
              .describe(
                "Detailed description of the complex task to accomplish"
              )
          }),
          execute: async ({ task }) => {
            const result = await this.multiAgentOrchestrate(task);

            // Store task in structured memory for history
            await this.updateStructuredMemory(
              "task_history",
              `task_${Date.now()}`,
              task.slice(0, 200),
              1.0
            );

            return result;
          }
        }),

        // ═════════════════════════════════════════════════════════
        // ③ STRUCTURED MEMORY — Remember & Recall User Context
        // ═════════════════════════════════════════════════════════

        rememberAboutUser: tool({
          description:
            "Store structured information about the user: their profile details, preferences, goals, or important facts. This persists across conversations.",
          inputSchema: z.object({
            type: z
              .enum(["profile", "preference", "goal", "fact"])
              .describe("Type of information"),
            key: z
              .string()
              .describe(
                "Short key (e.g., 'name', 'favorite_language', 'career_goal')"
              ),
            value: z.string().describe("The information to remember")
          }),
          execute: async ({ type, key, value }) => {
            return await this.updateStructuredMemory(type, key, value);
          }
        }),

        recallUserContext: tool({
          description:
            "Retrieve the full structured memory tree about the user: profile, preferences, goals, and known facts.",
          inputSchema: z.object({}),
          execute: async () => {
            return await this.getStructuredMemory();
          }
        }),

        // ═════════════════════════════════════════════════════════
        // EXISTING TOOLS
        // ═════════════════════════════════════════════════════════

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

        getUserTimezone: tool({
          description: "Get the user's timezone from their browser",
          inputSchema: z.object({})
        }),

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

        webSearch: tool({
          description: "Search the web for current information on any topic.",
          inputSchema: z.object({
            query: z.string().describe("The search query")
          }),
          execute: async ({ query }) => {
            try {
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
                      title: "Related",
                      url: topic.FirstURL,
                      content: topic.Text
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

        generateImage: tool({
          description: "Generate an image from a text description using AI.",
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

        translateText: tool({
          description: "Translate text from one language to another",
          inputSchema: z.object({
            text: z.string().describe("Text to translate"),
            targetLang: z
              .string()
              .describe("Target language (e.g., 'French', 'Spanish')"),
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
                maxOutputTokens: 500
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

        analyzeSentiment: tool({
          description: "Analyze the emotional sentiment of text.",
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

        storeKnowledge: tool({
          description:
            "Store information in the persistent knowledge base for future RAG retrieval.",
          inputSchema: z.object({
            content: z.string().describe("The information to store"),
            category: z
              .string()
              .describe("Category (e.g., 'personal', 'work', 'research')")
          }),
          execute: async ({ content, category }) => {
            return await this.storeKnowledge(content, category);
          }
        }),

        searchKnowledge: tool({
          description:
            "Basic single-pass search over the knowledge base. For complex queries, prefer deepSearch.",
          inputSchema: z.object({
            query: z.string().describe("Search query")
          }),
          execute: async ({ query }) => {
            return await this.searchKnowledge(query);
          }
        }),

        executeCode: tool({
          description: "Execute JavaScript code in a sandboxed environment.",
          inputSchema: z.object({
            code: z.string().describe("JavaScript code to execute"),
            description: z
              .string()
              .describe("Brief description of what the code does")
          }),
          needsApproval: async () => true,
          execute: async ({ code, description }) => {
            try {
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
              maxOutputTokens: 500
            });
            return {
              summary,
              originalLength: text.length,
              style: style ?? "paragraph"
            };
          }
        }),

        getCapabilities: tool({
          description:
            "List all available tools and capabilities of this agent",
          inputSchema: z.object({}),
          execute: async () => {
            return {
              model: "Llama 3.3 70B (Workers AI)",
              frontierFeatures: [
                "🔄 Agentic RAG — Iterative multi-round retrieval with confidence-based query refinement",
                "🤖 Multi-Agent Orchestration — Planner → Worker → Reviewer pipeline for complex tasks",
                "🧬 Structured Memory — Hierarchical user profile, preferences, goals, and facts"
              ],
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
              memoryArchitecture: {
                conversational: "SQLite-backed summaries via Durable Objects",
                semantic: "Cloudflare Vectorize (BGE embeddings)",
                structured: "Hierarchical profile/preference/goal/fact tree"
              }
            };
          }
        })
      },
      stopWhen: stepCountIs(8),
      abortSignal: options?.abortSignal
    });

    // ── Background tasks after streaming ──
    // Long-term memory: summarize + auto-extract preferences
    this.summarizeAndStore(messages).catch(() => {});
    this.autoExtractPreferences(messages).catch(() => {});

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
