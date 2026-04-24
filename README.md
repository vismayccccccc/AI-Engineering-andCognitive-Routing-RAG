# AI Persona Agent with RAG (LangGraph)

## Overview

This project implements an AI agent that generates persona-driven social media posts using large language models, LangGraph, and Retrieval-Augmented Generation (RAG). The system simulates how different personas respond to real-world events by combining reasoning, external context, and memory.

## Features

* Persona-based routing using embeddings (ChromaDB)
* AI agent pipeline with LLM reasoning
* Graph-based execution using LangGraph
* Retrieval-Augmented Generation (RAG) with memory
* Structured JSON output generation
* Error handling for unreliable LLM responses

## Architecture

Persona → Query Generation → Context Retrieval → Memory Retrieval → Post Generation → Storage

## Tech Stack

* Python
* LangChain / LangGraph
* ChromaDB (Vector Database)
* Sentence Transformers
* Groq (LLM API)

## Setup Instructions

1. Clone the repository:
   git clone <your-repo-link>
   cd grid07_ai_assignment

2. Create environment:
   conda create -n ai-agent python=3.10
   conda activate ai-agent

3. Install dependencies:
   pip install -r requirements.txt

4. Add .env file:
   GROQ_API_KEY=your_api_key_here

## Usage

Open Main.ipynb and run all cells step-by-step to execute the full pipeline.

## Example Output

{
"bot_id": "A",
"topic": "AI Impact on Developers",
"post_content": "OpenAI's new model is a game-changer..."
}

## Key Learnings

* Building AI agents with tool usage
* Handling inconsistent LLM outputs
* Using vector databases for memory
* Designing scalable pipelines with LangGraph

## Future Improvements

* Real-time API integration
* Multi-agent system
* UI dashboard

## Author

Your Name
