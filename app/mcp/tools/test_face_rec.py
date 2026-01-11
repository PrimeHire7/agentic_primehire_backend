import cv2
import sys
import numpy as np
from pathlib import Path

# Import your functions
from app.mcp.tools.interview_bot import (

    get_face_embedding,
    load_registered_face,
    compute_similarity
)

def test(candidate_name, candidate_id):
    print("\n========== FACE RECOGNITION TEST ==========")
    print("Candidate:", candidate_name)
    print("Candidate ID:", candidate_id)

    # Try loading the registered embedding
    reg = load_registered_face(candidate_name, candidate_id)
    if reg is None:
        print("❌ Registered face embedding NOT FOUND")
        return

    print("✅ Registered embedding loaded! Shape:", reg.shape)

    # Locate saved face file
    exts = ["png", "jpg", "jpeg"]
    saved_file = None
    for ext in exts:
        p = Path("saved_faces") / f"{candidate_name}_{candidate_id}.{ext}"
        if p.exists():
            saved_file = p
            break

    if not saved_file:
        print("❌ Saved face file not found")
        return

    print("📂 Saved face file found:", saved_file)

    # Load and embed
    img = cv2.imread(str(saved_file))
    if img is None:
        print("❌ cv2 failed to read image")
        return

    emb = get_face_embedding(img)
    if emb is None:
        print("❌ InsightFace failed to extract embedding")
        return

    print("✅ Live embedding computed. Shape:", emb.shape)

    # Compare similarity
    sim = compute_similarity(emb, reg)
    print("🔍 Similarity (saved vs saved):", sim)

    if sim is None:
        print("❌ Similarity computation failed")
    elif sim < 0.60:
        print("⚠ FACE MISMATCH - similarity too low!")
    else:
        print("👍 Face MATCH - similarity high enough")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_face_rec.py <candidate_name> <candidate_id>")
    else:
        test(sys.argv[1], sys.argv[2])
