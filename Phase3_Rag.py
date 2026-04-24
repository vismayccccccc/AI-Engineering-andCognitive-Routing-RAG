import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
memory = client.create_collection("post_memory")


def store_memory(post_id, text):
    embedding = model.encode(text).tolist()

    memory.add(
        documents=[text],
        embeddings=[embedding],
        ids=[post_id]
    )


def retrieve_memory(query):
    embedding = model.encode(query).tolist()

    results = memory.query(
        query_embeddings=[embedding],
        n_results=2
    )

    return results["documents"][0]