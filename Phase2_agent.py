"""
Phase 2: The Autonomous Content Engine (LangGraph)
Builds a proper state machine with 3 nodes:
  Node 1: decide_search  → LLM picks a topic and formats a search query
  Node 2: web_search     → mock_searxng_search fetches context
  Node 3: draft_post     → LLM generates a structured JSON post
"""

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv
import os, json, re

load_dotenv()

# ── LLM Setup ──────────────────────────────────────────────────────────────────
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

# ── State Schema ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    bot_id: str
    persona: str
    query: str
    context: str
    output: dict

# ── Mock Search Tool ───────────────────────────────────────────────────────────
def mock_searxng_search(query: str) -> str:
    """
    Simulates a real search engine (SearXNG) by returning
    hardcoded headlines based on keywords in the query.
    """
    q = query.lower()
    if "ai" in q or "model" in q or "developer" in q:
        return "OpenAI releases GPT-5, feared to replace junior developers worldwide."
    elif "crypto" in q or "bitcoin" in q or "blockchain" in q:
        return "Bitcoin hits new all-time high amid regulatory ETF approvals."
    elif "market" in q or "rate" in q or "fed" in q or "stock" in q:
        return "Fed holds interest rates steady; S&P 500 surges on strong earnings."
    elif "tech" in q or "monopoly" in q or "privacy" in q or "data" in q:
        return "EU fines Meta $1.3B for privacy violations; calls for Big Tech breakup."
    elif "space" in q or "elon" in q or "musk" in q:
        return "SpaceX successfully launches Starship on its 5th integrated flight test."
    else:
        return "Tech industry disruption continues at unprecedented pace."

# ── Node 1: Decide Search ──────────────────────────────────────────────────────
def decide_search_node(state: AgentState) -> dict:
    """
    LLM reads the bot's persona and decides what topic
    it wants to post about today, formatted as a search query.
    """
    prompt = f"""
    You are this persona:
    {state['persona']}

    What topic do you want to post about today?
    Generate ONLY a short 3-5 word search query. Nothing else.
    No explanation, no punctuation, just the query.
    """
    query = llm.invoke(prompt).content.strip()
    print(f"🔍 [Node 1 - Decide Search] Query: {query}")
    return {"query": query}

# ── Node 2: Web Search ─────────────────────────────────────────────────────────
def web_search_node(state: AgentState) -> dict:
    """
    Executes the mock search tool to retrieve real-world context
    based on the query generated in Node 1.
    """
    context = mock_searxng_search(state["query"])
    print(f"📰 [Node 2 - Web Search] Context: {context}")
    return {"context": context}

# ── Node 3: Draft Post ─────────────────────────────────────────────────────────
def draft_post_node(state: AgentState) -> dict:
    """
    LLM combines the bot persona + search context to generate
    a highly opinionated 280-character post in strict JSON format.
    """
    prompt = f"""
    You are this persona:
    {state['persona']}

    Based on this real-world news:
    {state['context']}

    Write a strong, opinionated tweet (max 280 characters) that reflects your persona.

    Return ONLY valid JSON. No markdown, no explanation, no extra text.
    Format:
    {{
        "bot_id": "{state['bot_id']}",
        "topic": "short topic label",
        "post_content": "your tweet here"
    }}
    """
    raw = llm.invoke(prompt).content

    try:
        json_str = re.search(r"\{.*\}", raw, re.DOTALL).group()
        parsed = json.loads(json_str)
    except Exception:
        parsed = {"error": "Invalid JSON", "raw": raw}

    print(f"✅ [Node 3 - Draft Post] Output: {parsed}")
    return {"output": parsed}

# ── Build LangGraph ────────────────────────────────────────────────────────────
builder = StateGraph(AgentState)

builder.add_node("decide_search", decide_search_node)
builder.add_node("web_search", web_search_node)
builder.add_node("draft_post", draft_post_node)

builder.set_entry_point("decide_search")
builder.add_edge("decide_search", "web_search")
builder.add_edge("web_search", "draft_post")
builder.add_edge("draft_post", END)

graph = builder.compile()
print("✅ LangGraph compiled: decide_search → web_search → draft_post\n")

# ── Run All 3 Bots ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bots = {
        "A": (
            "I believe AI and crypto will solve all human problems. "
            "I am highly optimistic about technology, Elon Musk, and space exploration. "
            "I dismiss regulatory concerns."
        ),
        "B": (
            "I believe late-stage capitalism and tech monopolies are destroying society. "
            "I am highly critical of AI, social media, and billionaires. "
            "I value privacy and nature."
        ),
        "C": (
            "I strictly care about markets, interest rates, trading algorithms, and making money. "
            "I speak in finance jargon and view everything through the lens of ROI."
        ),
    }

    for bot_id, persona in bots.items():
        print(f"\n{'='*60}")
        print(f"🤖 Running Bot {bot_id}")
        print('='*60)
        result = graph.invoke({
            "bot_id": bot_id,
            "persona": persona,
            "query": "",
            "context": "",
            "output": {}
        })
        print(f"\n📤 Final JSON Output:\n{json.dumps(result['output'], indent=2)}")
