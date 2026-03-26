# 📜 ARCHITECTURAL PROMPTS: The Vision

This document details the high-level "System Directives" (Prompts) that define the **Aether Frontier** architecture. These are the master prompts that tell the AI how to think, build, and evolve this specific system.

## 🏗️ Prompt 1: The Foundation (Durable Objects & State)
> "As a Lead Cloud Architect specializing in Cloudflare Workers, design a Durable Object-based AI Agent. The system MUST persist long-term conversation memory and user-specific structured state (profile, goals, facts). Use DO's storage for SQLite-backed memory persistence, ensuring low-latency access and session affinity."

## 🧠 Prompt 2: Agentic RAG (Vectorize & Iterative Loops)
> "Implement an 'Agentic RAG' module using Cloudflare Vectorize and `@cf/baai/bge-base-en-v1.5` embeddings. The retrieval pipeline must NOT be a simple one-pass search; it must use an iterative *ReAct loop*. If search confidence is below 0.7, the model must autonomously refine the query and search again up to 3 times."

## 🤖 Prompt 3: Multi-Agent Orchestration (Pipeline)
> "Create a Multi-Agent 'Planner-Worker-Reviewer' system. Decompose complex tasks into a directed graph:
> 1. **Planner**: Break the task into discrete steps.
> 2. **Worker**: Execute the steps with high-fidelity output.
> 3. **Reviewer**: Critically evaluate the output against original constraints.
> If the Reviewer rejects, the Worker MUST revise. All agents use `llama-3.3-70b-instruct`."

## 🧬 Prompt 4: Hierarchical Structured Memory
> "Define a hierarchical memory schema for user personalization. Categorize information into:
> - **Profile**: Static user attributes.
> - **Preferences**: Working styles and format choices.
> - **Goals**: Long-term objectives.
> - **Facts**: Verified context about the user's world.
> Auto-extract these from conversations using a dedicated extraction LLM pass after every turn."
