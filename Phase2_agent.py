from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

def decide_search(persona):
    prompt = f"""
    Based on this persona:
    {persona}

    Generate ONLY a short 3-5 word search query.
    """
    return llm.invoke(prompt).content.strip()


def mock_search(query):
    if "ai" in query.lower():
        return "OpenAI released a new AI model affecting developers."
    elif "crypto" in query.lower():
        return "Bitcoin is rising due to institutional investment."
    else:
        return "Tech industry is evolving rapidly."


def generate_post(bot_id, persona, context):
    prompt = f"""
    You are this persona:
    {persona}

    Based on this news:
    {context}

    Write a strong opinionated tweet (max 280 characters).

    Return ONLY valid JSON.

    Format:
    {{
        "bot_id": "{bot_id}",
        "topic": "short topic",
        "post_content": "tweet"
    }}
    """
    return llm.invoke(prompt).content