import os
from uuid import uuid4
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import requests

model = SentenceTransformer("intfloat/e5-large-v2")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")

client = QdrantClient(host=QDRANT_HOST, port=6333)

COLLECTION_NAME = "devops_docs"

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


def initialize_collection():
    if not client.collection_exists(COLLECTION_NAME):
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    data_dir = Path("data")
    md_files = list(data_dir.rglob("*.md"))

    all_text = ""
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as file:
            all_text += file.read() + "\n\n"

    chunks = splitter.split_text(all_text)

    points = []

    for chunk in chunks:
        vector = model.encode(f"passage: {chunk}").tolist()
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={"text": chunk},
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)


try:
    client.get_collection(COLLECTION_NAME)
except:
    initialize_collection()


def query_qdrant(query, top_k=3):
    query_vector = model.encode(f"query: {query}").tolist()
    results = client.search(
        collection_name=COLLECTION_NAME, query_vector=query_vector, limit=top_k
    )
    return "\n".join([r.payload["text"] for r in results])


def query_ollama(prompt):
    res = requests.post(
        f"http://{OLLAMA_HOST}:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False},
    )
    return res.json()["response"]


def get_rag_response(query):
    context = query_qdrant(query)
    prompt = f"""
        You are a helpful assistant. Use the following context to answer:
        
        {context}

        Question: {query}

        Answer:
    """
    return query_ollama(prompt)
