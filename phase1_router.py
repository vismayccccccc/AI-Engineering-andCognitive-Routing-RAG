from sentence_transformers import SentenceTransformer
import chromadb

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create DB
client = chromadb.Client()
collection = client.create_collection("persona_router")

# Sample personas
personas = {
    "A": "I believe AI and crypto will solve all human problems.",
    "B": "I believe capitalism and tech monopolies are harmful.",
    "C": "I care about markets, trading, and ROI."
}

# Store personas
for pid, text in personas.items():
    embedding = model.encode(text).tolist()
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[pid]
    )

def route_post(post, max_distance=1.5):
    post_embedding = model.encode(post).tolist()

    results = collection.query(
        query_embeddings=[post_embedding],
        n_results=3
    )

    matched = []
    for i, dist in enumerate(results["distances"][0]):
        if dist < max_distance:
            matched.append(results["ids"][0][i])

    return matched