# 🏗️ SYSTEM ARCHITECTURE: Aether Frontier

This project is a reference implementation of a **Reasoning-First Autonomous Agent** built on Cloudflare's serverless stack.

## 🧱 Component Overview

### 1. 🧬 STATE ENGINE (Durable Objects)
- **Class**: `ChatAgent`
- **Persistence**: SQLite (via `this.sql`)
- **Role**: Maintains the "Entity Context" (User Profile, Goals, Facts) and "Conversation Context" (Summaries).
- **Benefit**: Zero-latency state access and strong consistency for agentic operations.

### 2. 🧠 RETRIEVAL ENGINE (Vectorize)
- **Index**: `ai-knowledge`
- **Model**: `@cf/baai/bge-base-en-v1.5`
- **Pattern**: **Agentic RAG**. Instead of a static search, the agent uses an iterative `deepSearch` loop to refine its own queries based on previous results.

### 3. 🤖 ORCHESTRATION ENGINE (Multi-Agent)
- **Flow**: `Planner` → `Worker` → `Reviewer`.
- **Logic**: Complex tasks are never handled in a single pass. The system decomposes, executes, and then critically reviews its own output.
- **Model**: `meta/llama-3.3-70b-instruct-fp8-fast`.

### 4. 🗃️ SCHEMA DESIGN
The SQLite-backed memory is structured for hierarchical retrieval:
- `conversation_memory`: Summaries and sentiments.
- `knowledge_base`: Indexed content for RAG.
- `structured_memory`: Entity-relationship key-values (User Profile/Prefs).

## 🔄 Lifecycle of a Request
1. **REASON**: The agent analyzes the user's input against the System Prompt (ReAct framework).
2. **ACT**: The agent chooses a tool (e.g., `deepSearch`, `webSearch`, `complexTask`).
3. **REFLECT**: The agent verifies the result and either continues or responds.
4. **EVOLVE**: After the turn, the agent auto-extracts information to update the `structured_memory`.
