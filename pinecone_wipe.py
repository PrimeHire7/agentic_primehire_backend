import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "primehire-production"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

print("🔥 Deleting all vectors from __default__ namespace...")
index.delete(delete_all=True, namespace="__default__")

print("🔥 Deleting all vectors from candidates namespace...")
index.delete(delete_all=True, namespace="candidates")

print("✅ Completed cleanup.")
