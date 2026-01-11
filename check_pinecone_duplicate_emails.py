import os
from collections import defaultdict
from dotenv import load_dotenv
from pinecone import Pinecone

# ----------------------------
# Load .env
# ----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "primehire-production")
PINECONE_NAMESPACE = "__default__"

if not PINECONE_API_KEY:
    raise RuntimeError("❌ PINECONE_API_KEY not found in .env")

# ----------------------------
# Init Pinecone
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

print(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")

# ----------------------------
# 1️⃣ List all vector IDs
# ----------------------------
all_ids = []
for batch in index.list(namespace=PINECONE_NAMESPACE):
    # batch is a list[str]
    all_ids.extend(batch)

print(f"🔍 Total vector IDs found: {len(all_ids)}")

# ----------------------------
# 2️⃣ Fetch metadata in batches
# ----------------------------
email_map = defaultdict(list)

BATCH_SIZE = 100  # safe limit

for i in range(0, len(all_ids), BATCH_SIZE):
    batch_ids = all_ids[i:i + BATCH_SIZE]

    res = index.fetch(
        ids=batch_ids,
        namespace=PINECONE_NAMESPACE
    )

    # ✅ FetchResponse object
    for vec_id, vec in res.vectors.items():
        meta = vec.metadata or {}
        email = meta.get("email")

        if isinstance(email, str):
            email = email.strip().lower()

        if not email:
            continue

        email_map[email].append(vec_id)

# ----------------------------
# 3️⃣ Detect duplicates
# ----------------------------
duplicates = {
    email: ids
    for email, ids in email_map.items()
    if len(ids) > 1
}

print("\n================ DUPLICATE EMAILS IN PINECONE ================")

if not duplicates:
    print("🎉 No duplicate emails found in Pinecone")
else:
    print(f"⚠️ Found {len(duplicates)} duplicate emails:\n")
    for email, ids in duplicates.items():
        print(f"{email} → {ids}")

print("==============================================================")
