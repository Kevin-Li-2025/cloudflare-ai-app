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

| Requirement | Implementation |
|---|---|
| **LLM** | Llama 3.3 70B (`@cf/meta/llama-3.3-70b-instruct-fp8-fast`) on Workers AI |
| **Workflow / Coordination** | Durable Objects with `AIChatAgent` for stateful agent lifecycle, multi-step tool orchestration with `stepCountIs(8)` |
| **User Input (Chat/Voice)** | Real-time WebSocket chat UI + **Web Speech API** voice input with live transcription |
| **Memory / State** | SQLite-backed conversation memory + auto-summarization + Vectorize RAG knowledge base |

## 🧠 Frontier Capabilities

### Core AI
- **Llama 3.3 70B** — Meta's frontier open-weight model running on Cloudflare's serverless GPUs
- **Streaming responses** — Token-by-token streaming via WebSocket with resumable streams
- **Multi-step reasoning** — Up to 8 tool-calling steps per turn for complex tasks

### Semantic Memory (RAG)
- **Vectorize integration** — Store and retrieve knowledge using `@cf/baai/bge-base-en-v1.5` embeddings
- **Persistent knowledge base** — SQLite-backed storage with vector search fallback
- **Auto-summarization** — Conversations are automatically summarized and stored for long-term memory

### Tools & Capabilities
| Tool | Description |
|---|---|
| 🌤️ `getWeather` | Real weather data via wttr.in API |
| 🔍 `webSearch` | Real-time web search via DuckDuckGo |
| 🎨 `generateImage` | Text-to-image with Flux (Black Forest Labs) on Workers AI |
| 🌍 `translateText` | Multi-language translation powered by Llama 3.3 |
| 😊 `analyzeSentiment` | NLP sentiment classification via DistilBERT |
| 📝 `summarizeText` | AI-powered text summarization (bullet points, paragraph, executive) |
| 💻 `executeCode` | Sandboxed JavaScript execution with human-in-the-loop approval |
| ⏰ `scheduleTask` | Cron, delayed, and one-time task scheduling |
| 🧠 `storeKnowledge` | Persist information to RAG knowledge base |
| 🔎 `searchKnowledge` | Semantic search over stored knowledge |
| 🔢 `calculate` | Math with approval gates for large numbers |
| 🔌 MCP Tools | Dynamic external tool integration via Model Context Protocol |

### User Experience
- **🎤 Voice Input** — Browser-native speech recognition with live transcription indicator
- **📎 Image Upload** — Drag-and-drop, paste, or click to attach images
- **🌓 Dark/Light Theme** — Persistent theme preference with smooth toggle
- **🛡️ Human-in-the-Loop** — Approval workflows for sensitive operations (code execution, large calculations)
- **⏰ Toast Notifications** — Real-time alerts for completed scheduled tasks
- **🔌 MCP Panel** — Connect external tool servers via the Model Context Protocol

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.3 70B on Workers AI |
| Image Gen | Flux-1-schnell on Workers AI |
| Embeddings | BGE Base EN v1.5 on Workers AI |
| Sentiment | DistilBERT SST-2 on Workers AI |
| Runtime | Cloudflare Workers (Durable Objects) |
| State | SQLite (built into Durable Objects) |
| Vector DB | Cloudflare Vectorize |
| Frontend | React 19 + Vite + TailwindCSS + Kumo Design System |
| Voice | Web Speech API (SpeechRecognition) |
| Protocol | WebSocket (real-time) + Server-Sent Events |
| Tool System | Vercel AI SDK with MCP integration |

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
