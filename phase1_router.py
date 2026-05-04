
    """
Phase 1: Vector-Based Persona Matching (The Router)
Uses cosine similarity (not L2 distance) as specified in the assignment.
Function signature matches: route_post_to_bots(post_content, threshold=0.85)
"""

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Use cosine similarity space in ChromaDB
client = chromadb.Client(Settings())
collection = client.create_collection(
    name="persona_router",
    metadata={"hnsw:space": "cosine"}  # <-- KEY FIX: use cosine, not L2
)

# Full persona descriptions as specified in the assignment
personas = {
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

# Embed and store all personas in the vector DB
for bot_id, persona_text in personas.items():
    embedding = model.encode(persona_text).tolist()
    collection.add(
        documents=[persona_text],
        embeddings=[embedding],
        ids=[bot_id]
    )

print("✅ Personas embedded and stored in ChromaDB (cosine space).\n")


def route_post_to_bots(post_content: str, threshold: float = 0.85) -> list[str]:
    """
    Embeds the incoming post and finds which bot personas are similar enough
    to 'care' about it, using cosine similarity.

    In ChromaDB cosine space, distances are returned as (1 - cosine_similarity),
    so distance = 0 means identical, distance = 1 means opposite.

    To get cosine_similarity > threshold, we filter for distance < (1 - threshold).

    Args:
        post_content: The social media post to route.
        threshold: Minimum cosine similarity required (default 0.85).

    Returns:
        List of bot IDs (e.g. ["A", "C"]) whose personas match the post.
    """
    post_embedding = model.encode(post_content).tolist()

    results = collection.query(
        query_embeddings=[post_embedding],
        n_results=3  # fetch all 3 bots
    )

    matched_bots = []
    max_distance = 1 - threshold  # Convert similarity threshold to distance threshold

    print(f"📨 Post: \"{post_content}\"")
    print(f"📏 Threshold: cosine similarity > {threshold} (distance < {max_distance:.2f})\n")
    print(f"{'Bot':<6} {'Cosine Distance':<20} {'Cosine Similarity':<20} {'Matched?'}")
    print("-" * 60)

    for i, (bot_id, distance) in enumerate(
        zip(results["ids"][0], results["distances"][0])
    ):
        cosine_similarity = 1 - distance
        matched = cosine_similarity > threshold
        print(
            f"Bot {bot_id:<3} {distance:<20.4f} {cosine_similarity:<20.4f} {'✅ YES' if matched else '❌ NO'}"
        )
        if matched:
            matched_bots.append(bot_id)

    print(f"\n🎯 Routed to bots: {matched_bots if matched_bots else 'None (try lowering threshold)'}\n")
    return matched_bots


# --- Test it ---
if __name__ == "__main__":
    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits all time high as ETF approvals flood the market.",
        "Big Tech companies are exploiting user data and should be broken up.",
    ]

    for post in test_posts:
        route_post_to_bots(post, threshold=0.85)
        print("=" * 60 + "\n")
