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
# 1️⃣ Collect all vector IDs
# ----------------------------
all_ids = []
for batch in index.list(namespace=PINECONE_NAMESPACE):
    all_ids.extend(batch)

print(f"🔍 Total vectors found: {len(all_ids)}")

# ----------------------------
# 2️⃣ Fetch metadata & group by email
# ----------------------------
email_map = defaultdict(list)
BATCH_SIZE = 100

for i in range(0, len(all_ids), BATCH_SIZE):
    batch_ids = all_ids[i:i + BATCH_SIZE]

    res = index.fetch(
        ids=batch_ids,
        namespace=PINECONE_NAMESPACE
    )

    for vec_id, vec in res.vectors.items():
        meta = vec.metadata or {}
        email = meta.get("email")

        if isinstance(email, str):
            email = email.strip().lower()

        if not email:
            continue

        email_map[email].append(vec_id)

# ----------------------------
# 3️⃣ Identify deletions (delete FIRST ID only)
# ----------------------------
to_delete = []

for email, ids in email_map.items():
    if len(ids) > 1:
        delete_id = ids[0]  # 👈 DELETE FIRST ONE
        to_delete.append(delete_id)
        print(f"🗑️ Marked for delete: {email} → {delete_id}")

print(f"\n⚠️ Total vectors to delete: {len(to_delete)}")

# ----------------------------
# 4️⃣ Confirm before delete
# ----------------------------
confirm = input("\nType YES to proceed with deletion: ").strip()

if confirm != "YES":
    print("❌ Aborted. No vectors deleted.")
    exit(0)

# ----------------------------
# 5️⃣ Delete from Pinecone
# ----------------------------
DELETE_BATCH = 100

for i in range(0, len(to_delete), DELETE_BATCH):
    batch = to_delete[i:i + DELETE_BATCH]

    index.delete(
        ids=batch,
        namespace=PINECONE_NAMESPACE
    )

print("✅ Pinecone duplicate cleanup completed successfully.")
