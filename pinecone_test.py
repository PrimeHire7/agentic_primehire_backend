import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone

# ✅ Load .env file
load_dotenv()

# ✅ Verify key loaded
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("❌ Missing PINECONE_API_KEY — check your .env file or export it")

print(f"[DEBUG] Pinecone API Key Loaded: {api_key[:12]}...")

# ✅ Initialize client
pc = Pinecone(api_key=api_key)

# ✅ List indexes safely
indexes = pc.list_indexes()

# Convert safely (SDK objects → plain dict)
if hasattr(indexes, "to_dict"):
    indexes = indexes.to_dict()
elif hasattr(indexes, "__iter__"):
    indexes = [i.to_dict() if hasattr(i, "to_dict") else str(i) for i in indexes]

print("\n✅ Pinecone Index List:")
print(json.dumps(indexes, indent=2))

# ✅ Verify connectivity to your index
index_name = "primehire-production"
print(f"\n[DEBUG] Connecting to index: {index_name}")
index = pc.Index(index_name)

stats = index.describe_index_stats()
stats = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)

print("\n✅ Index Stats:")
print(json.dumps(stats, indent=2))
