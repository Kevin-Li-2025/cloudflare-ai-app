# 🚀 Cloudflare AI Agent — Frontier Intelligence Platform

A world-class AI-powered agent application built on Cloudflare's platform, showcasing frontier AI capabilities with **Llama 3.3 70B** on Workers AI, persistent state via Durable Objects, semantic memory via Vectorize, and real-time user interaction.

> **Built for the Cloudflare AI Application Assignment** — demonstrating mastery of LLM integration, agentic workflows, real-time user input, and persistent memory/state.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   React Frontend                     │
│  Voice Input │ Image Upload │ Chat UI │ Tool Approval│
└──────────────────────┬──────────────────────────────┘
                       │ WebSocket (real-time)
┌──────────────────────▼──────────────────────────────┐
│              ChatAgent (Durable Object)              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Llama 3.3   │  │ SQLite State │  │ Vectorize  │ │
│  │ Workers AI  │  │ (Memory)     │  │ (RAG)      │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Flux Image  │  │ BGE Embeddings│ │ Scheduler  │ │
│  │ Generation  │  │ (Semantic)   │  │ (Cron/Delay│ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Web Search  │  │ Sentiment    │  │ MCP Tools  │ │
│  │ (DuckDuckGo)│  │ Analysis     │  │ Integration│ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
```

## ✅ Assignment Requirements Mapping

| Requirement                 | Implementation                                                                                                       |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **LLM**                     | Llama 3.3 70B (`@cf/meta/llama-3.3-70b-instruct-fp8-fast`) on Workers AI                                             |
| **Workflow / Coordination** | Durable Objects with `AIChatAgent` for stateful agent lifecycle, multi-step tool orchestration with `stepCountIs(8)` |
| **User Input (Chat/Voice)** | Real-time WebSocket chat UI + **Web Speech API** voice input with live transcription                                 |
| **Memory / State**          | SQLite-backed conversation memory + auto-summarization + Vectorize RAG knowledge base                                |

## 🧠 Frontier AI Architecture

### ① Agentic RAG — Iterative Multi-Round Retrieval

RAG goes from pipeline → **reasoning loop**. The agent autonomously decides how many search rounds to perform, refining its query each round until confidence exceeds the threshold.

```
Round 1: search("original query") → score 0.4 ✗
  → LLM refines query based on partial results
Round 2: search("refined query") → score 0.6 ✗
  → LLM refines again
Round 3: search("highly specific query") → score 0.85 ✓ → return
```

### ② Multi-Agent Orchestration — Planner → Worker → Reviewer

Complex tasks are decomposed across **3 specialized LLM personas**:

```
User Request → [Planner Agent] → Step-by-step plan
                    ↓
              [Worker Agent] → Executes each step
                    ↓
             [Reviewer Agent] → Validates output
                    ↓ (if rejected)
              [Worker Agent] → Revises based on feedback
```

### ③ Structured Memory — Hierarchical User Profile

Memory evolves from flat vectors to **structured knowledge**:

```
user_memory
 ├── profile:    { name, role, expertise }
 ├── preferences: { language, style, topics }
 ├── goals:      [ career_goals, learning_goals ]
 └── facts:      [ key_fact_1, key_fact_2 ]
```

Auto-extracted from conversations via LLM, persisted in SQLite, injected into system prompt.

### Core AI

- **Llama 3.3 70B** — Meta's frontier open-weight model on Cloudflare's serverless GPUs
- **Streaming responses** — Token-by-token streaming via WebSocket with resumable streams
- **Multi-step reasoning** — Up to 8 tool-calling steps per turn

### Semantic Memory (RAG)

- **Vectorize integration** — Store and retrieve knowledge using `@cf/baai/bge-base-en-v1.5` embeddings
- **Persistent knowledge base** — SQLite-backed storage with vector search fallback
- **Auto-summarization** — Conversations automatically summarized for long-term memory

### Tools & Capabilities

| Tool                  | Description                                                         |
| --------------------- | ------------------------------------------------------------------- |
| 🌤️ `getWeather`       | Real weather data via wttr.in API                                   |
| 🔍 `webSearch`        | Real-time web search via DuckDuckGo                                 |
| 🎨 `generateImage`    | Text-to-image with Flux (Black Forest Labs) on Workers AI           |
| 🌍 `translateText`    | Multi-language translation powered by Llama 3.3                     |
| 😊 `analyzeSentiment` | NLP sentiment classification via DistilBERT                         |
| 📝 `summarizeText`    | AI-powered text summarization (bullet points, paragraph, executive) |
| 💻 `executeCode`      | Sandboxed JavaScript execution with human-in-the-loop approval      |
| ⏰ `scheduleTask`     | Cron, delayed, and one-time task scheduling                         |
| 🧠 `storeKnowledge`   | Persist information to RAG knowledge base                           |
| 🔎 `searchKnowledge`  | Semantic search over stored knowledge                               |
| 🔢 `calculate`        | Math with approval gates for large numbers                          |
| 🔌 MCP Tools          | Dynamic external tool integration via Model Context Protocol        |

### User Experience

- **🎤 Voice Input** — Browser-native speech recognition with live transcription indicator
- **📎 Image Upload** — Drag-and-drop, paste, or click to attach images
- **🌓 Dark/Light Theme** — Persistent theme preference with smooth toggle
- **🛡️ Human-in-the-Loop** — Approval workflows for sensitive operations (code execution, large calculations)
- **⏰ Toast Notifications** — Real-time alerts for completed scheduled tasks
- **🔌 MCP Panel** — Connect external tool servers via the Model Context Protocol

## Tech Stack

| Layer       | Technology                                         |
| ----------- | -------------------------------------------------- |
| LLM         | Llama 3.3 70B on Workers AI                        |
| Image Gen   | Flux-1-schnell on Workers AI                       |
| Embeddings  | BGE Base EN v1.5 on Workers AI                     |
| Sentiment   | DistilBERT SST-2 on Workers AI                     |
| Runtime     | Cloudflare Workers (Durable Objects)               |
| State       | SQLite (built into Durable Objects)                |
| Vector DB   | Cloudflare Vectorize                               |
| Frontend    | React 19 + Vite + TailwindCSS + Kumo Design System |
| Voice       | Web Speech API (SpeechRecognition)                 |
| Protocol    | WebSocket (real-time) + Server-Sent Events         |
| Tool System | Vercel AI SDK with MCP integration                 |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Kevin-Li-2025/cloudflare-ai-app.git
cd cloudflare-ai-app

# Install dependencies
npm install

# Run locally (no API keys needed — uses Workers AI)
npm run dev

# Deploy to Cloudflare
npm run deploy
```

### Optional: Enable Vectorize (RAG)

```bash
# Create the Vectorize index for semantic memory
npx wrangler vectorize create ai-knowledge --dimensions=768 --metric=cosine
```

## 🤖 Switching to Frontier Models (Optional)

While this agent is optimized for Cloudflare Workers AI, you can easily switch to external frontier models like Claude 3.5 Sonnet by providing an API key.

```typescript
const result = streamText({
  model: anthropic("claude-sonnet-4-20250514")
  // ...
});
```

Create a `.env` file with your API key:

```
ANTHROPIC_API_KEY=your-key-here
```

## 🚀 Deploy

```bash
npm run deploy
```

Your agent is live on Cloudflare's global network. Messages persist in SQLite, streams resume on disconnect, and the agent hibernates when idle.

## 📚 Learn more

- [Agents SDK documentation](https://developers.cloudflare.com/agents/)
- [Build a chat agent tutorial](https://developers.cloudflare.com/agents/getting-started/build-a-chat-agent/)
- [Chat agents API reference](https://developers.cloudflare.com/agents/api-reference/chat-agents/)
- [Workers AI models](https://developers.cloudflare.com/workers-ai/models/)

## Project Structure

```
src/
├── server.ts     # Agent backend — LLM, tools, RAG, memory, scheduling
├── app.tsx       # React frontend — chat UI, voice input, capabilities
├── client.tsx    # React entry point
└── styles.css    # Tailwind + Kumo design system

wrangler.jsonc    # Cloudflare config — AI, Durable Objects, Vectorize
index.html        # SEO-optimized shell with Inter font
env.d.ts          # TypeScript bindings for AI, Vectorize, Durable Objects
```

## License

MIT
