# AI Persona Agent with RAG (LangGraph)

## Overview

This project builds an AI agent that generates persona-driven social media posts by combining large language models, structured workflows, and memory. The system simulates how different personas interpret and react to real-world events using contextual reasoning and past interactions.

Unlike simple prompt-based systems, this agent integrates retrieval and structured execution to produce more consistent and context-aware outputs.

---

## Problem Statement

Standard LLM outputs are often inconsistent and lack memory. This project addresses that by:

* Introducing structured execution using LangGraph
* Incorporating memory using Retrieval-Augmented Generation (RAG)
* Generating reliable, structured JSON outputs

---

## Features

* Persona-based routing using semantic embeddings (ChromaDB)
* Query generation using LLM reasoning
* Context simulation via tool-based retrieval
* Graph-based execution pipeline using LangGraph
* Memory integration using vector search (RAG)
* Structured JSON output with error handling

---

## Architecture

Persona
→ Query Generation (LLM)
→ Context Retrieval (Tool)
→ Memory Retrieval (Vector DB)
→ Post Generation (LLM)
→ Output Parsing
→ Storage (RAG Memory)

---

## Tech Stack

* Python
* LangChain / LangGraph
* ChromaDB (Vector Database)
* Sentence Transformers
* Groq (LLM API)

---

## Setup Instructions

1. Clone the repository:

```
git clone <your-repo-link>
cd grid07_ai_assignment
```

2. Create environment:

```
conda create -n ai-agent python=3.10
conda activate ai-agent
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Add `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

## Usage

Run the notebook:

```
Main.ipynb
```

Execute cells step-by-step to observe the full agent pipeline.

---

## Example Output

```
{
  "bot_id": "A",
  "topic": "AI Impact on Developers",
  "post_content": "OpenAI's new model is a game-changer..."
}
```

---

## Key Learnings

* Designing AI agents with tool-based reasoning
* Handling unreliable LLM outputs using parsing and validation
* Implementing memory with vector databases (RAG)
* Structuring AI workflows using LangGraph

---

## Future Improvements

* Real-time data integration (news/search APIs)
* Multi-agent interaction system
* Thread-aware conversation memory (deep RAG)
* Frontend interface for interaction

---

## Author

Vismay
