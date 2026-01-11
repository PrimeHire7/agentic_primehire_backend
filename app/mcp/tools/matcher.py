# app/mcp/tools/matcher.py
import os
import logging
import json
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# OpenAI / Pinecone
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger("matcher")
logger.setLevel(logging.DEBUG)

router = APIRouter()

# ---------- CONFIG ----------
DEBUG_MATCHER = os.getenv("DEBUG_MATCHER", "true").lower() in ("1", "true", "yes")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "primehire-production-v2")
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "20"))  # lowered to 20 for testing
TOP_K = int(os.getenv("MATCH_TOP_K", "20"))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None

if pc is None:
    logger.warning("[MATCHER INIT] Pinecone client not configured; matcher will fail.")
else:
    try:
        if INDEX_NAME not in pc.list_indexes().names():
            logger.info("[MATCHER INIT] Creating pinecone index: %s", INDEXNAME)
            pc.create_index(name=INDEX_NAME, dimension=3072, metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    except Exception as e:
        logger.exception("[MATCHER INIT] Pinecone init/check failed: %s", e)

index = pc.Index(INDEX_NAME) if pc is not None else None


# ---------- Request Model ----------
class MatchRequest(BaseModel):
    jd_text: str


# ---------- Helpers ----------
def debug_print(*args, **kwargs):
    if DEBUG_MATCHER:
        print("[MATCHER]", *args, **kwargs)


def get_embedding(text: str):
    if client is None:
        raise RuntimeError("OpenAI client not configured")
    resp = client.embeddings.create(model="text-embedding-3-large", input=text)
    return resp.data[0].embedding


# ---------- JD metadata extraction ----------
def extract_jd_metadata_via_llm(jd_text: str) -> dict:
    if client is None:
        debug_print("[JD META] OpenAI client not configured")
        return {"role": jd_text.splitlines()[0], "location": "", "years_experience": 0, "skills": []}

    prompt = f"""
Extract JSON with keys: role, location, years_experience, skills (list).
JD:
\"\"\"{jd_text}\"\"\" 
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        jd_meta = {
            "role": parsed.get("role", "") or "",
            "location": parsed.get("location", "") or "",
            "years_experience": int(parsed.get("years_experience") or 0),
            "skills": parsed.get("skills", []) or []
        }
        debug_print("[JD META] Parsed:", jd_meta)
        return jd_meta
    except Exception as e:
        logger.exception("[JD META] FAILED: %s", e)
        return {"role": "", "location": "", "years_experience": 0, "skills": []}


# ---------- Scoring ----------
def compute_skill_score(jd_skills, cand_skills):
    jd = set([s.lower().strip() for s in jd_skills or [] if s])
    cd = set([s.lower().strip() for s in cand_skills or [] if s])
    if not jd:
        return 0
    return round((len(jd & cd) / len(jd)) * 100, 2)


def compute_experience_score(req, got):
    try:
        req = float(req or 0)
        got = float(got or 0)
    except:
        return 0
    if req <= 0:
        return 50
    ratio = got / req
    if ratio >= 1:
        return min(100, 70 + (ratio - 1) * 15)
    return max(0, 70 * ratio)


def compute_designation_score(jd_role, title):
    jd = (jd_role or "").lower()
    tt = (title or "").lower()
    if not jd or not tt:
        return 0
    if jd in tt:
        return 100
    if tt in jd:
        return 70
    if jd.split()[0] in tt:
        return 50
    return 0


def compute_location_score(jd_loc, cand_loc):
    if not jd_loc or not cand_loc:
        return 0
    return 100 if jd_loc.lower().strip() in cand_loc.lower().strip() else 0


def semantic_to_percent(v):
    try:
        return round(float(v) * 100, 2)
    except:
        return 0


WEIGHTS = {
    "skill": 0.35,
    "semantic": 0.25,
    "experience": 0.15,
    "designation": 0.15,
    "location": 0.10,
}


def score_resume_vs_jd(meta, jd_meta, match):
    skill = compute_skill_score(jd_meta["skills"], meta["skills"])
    semantic = semantic_to_percent(match.get("score", 0))
    exp = compute_experience_score(jd_meta["years_experience"], meta["experience_years"])
    desig = compute_designation_score(jd_meta["role"], meta["designation"] or meta["current_title"])
    loc = compute_location_score(jd_meta["location"], meta["location"])

    final = (
        WEIGHTS["skill"] * skill +
        WEIGHTS["semantic"] * semantic +
        WEIGHTS["experience"] * exp +
        WEIGHTS["designation"] * desig +
        WEIGHTS["location"] * loc
    )

    return {
        "skill_score": round(skill, 2),
        "semantic_score": round(semantic, 2),
        "experience_score": round(exp, 2),
        "designation_score": round(desig, 2),
        "location_score": round(loc, 2),
        "final_score": round(final, 2),
    }


# ---------- MAIN MATCHER ----------
async def match_candidates_tool(jd_text: str, top_k: int = TOP_K):
    start = time.time()
    debug_print("[MATCHER] JD:", jd_text[:160])

    jd_meta = extract_jd_metadata_via_llm(jd_text)

    emb = get_embedding(jd_text)
    debug_print("[MATCHER] Embedding OK:", len(emb))

    if index is None:
        raise RuntimeError("Pinecone index missing")

    result = index.query(
        vector=emb,
        top_k=top_k,
        include_metadata=True,
        namespace="__default__"
    )

    matches = result.get("matches", [])
    debug_print(f"[MATCHER] Pinecone returned: {len(matches)}")

    final = []
    seen = set()
    total_scanned = 0

    for m in matches:
        total_scanned += 1
        meta = m.get("metadata") or {}

        cid = meta.get("candidate_id") or meta.get("email") or meta.get("full_name")
        if not cid:
            continue
        if cid in seen:
            debug_print("[MATCHER] duplicate skip:", cid)
            continue
        seen.add(cid)

        # normalize:
        try:
            exp = float(str(meta.get("years_of_experience") or meta.get("experience_years") or "0").replace("+", ""))
        except:
            exp = 0

        skills_raw = meta.get("top_skills") or meta.get("skills") or ""
        skills = [s.strip() for s in skills_raw.split(",") if s.strip()]

        resume = {
            "name": meta.get("full_name") or meta.get("name") or "",
            "full_name": meta.get("full_name") or meta.get("name") or "",
            "designation": meta.get("designation") or meta.get("current_title") or "",
            "current_title": meta.get("current_title") or "",
            "current_company": meta.get("current_company") or "",
            "experience_years": exp,
            "location": meta.get("location") or "",
            "skills": skills,
            "email": meta.get("email") or "",
            "phone": meta.get("phone") or "",
            "linkedin": meta.get("linkedin_url") or "",
            "candidate_id": cid,
        }

        breakdown = score_resume_vs_jd(resume, jd_meta, m)
        debug_print("[SCORE]", json.dumps(breakdown))

        final.append({
            **resume,
            "scores": breakdown
        })

    final.sort(key=lambda x: x["scores"]["final_score"], reverse=True)

    filtered = [c for c in final if c["scores"]["final_score"] >= MATCH_THRESHOLD]

    # -------------------------------
    # NEW → Create matched_candidate_ids for DB save
    # -------------------------------
    matched_candidate_ids = [c.get("candidate_id") for c in filtered if c.get("candidate_id")]
    debug_print("[MATCHER] Matched candidate IDs:", matched_candidate_ids)

    elapsed = round(time.time() - start, 2)
    debug_info = {
        "elapsed_seconds": elapsed,
        "threshold": MATCH_THRESHOLD,
        "total_scanned": total_scanned,
        "raw_count": len(final),
        "filtered_count": len(filtered),
    }

    return {
        "jd_meta": jd_meta,
        "raw_candidates_count": len(final),
        "candidates": filtered,
        "matched_candidate_ids": matched_candidate_ids,   # <= important
        "debug": debug_info
    }


# ---------- HTTP Endpoint ----------
@router.post("/profile/match")
async def profile_match(req: MatchRequest):
    try:
        result = await match_candidates_tool(req.jd_text)
        return {
            "jd_meta": result["jd_meta"],
            "candidates": result["candidates"],
            "debug": result["debug"],
        }
    except Exception as e:
        logger.exception("[PROFILE_MATCH] FAILED: %s", e)
        raise HTTPException(status_code=500, detail=str(e))



# ================================
# JD CLARIFIER ENDPOINT
# ================================
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
import json
import re

class JDClarifyRequest(BaseModel):
    jd_text: str


@router.post("/clarify")
async def clarify_jd(request: JDClarifyRequest):
    """
    Clarifies incomplete JD prompts by returning missing questions.
    Response example:
    {
        "complete": false,
        "questions": [
            "What skills should the candidate have?",
            "How many years of experience?",
            "Preferred location?"
        ]
    }
    """
    jd_text = request.jd_text.strip()
    if not jd_text:
        raise HTTPException(status_code=400, detail="JD text is empty.")

    prompt = f"""
    You are a JD clarifier bot.
    User wrote this job request:

    \"\"\"{jd_text}\"\"\"

    Your job:
    1. Detect if the job request includes:
       - role
       - required skills
       - years of experience
       - location
    2. If missing, ask questions to fill only the missing info.
    3. Return STRICT JSON ONLY:
    {{
        "complete": boolean,
        "questions": ["question1", "question2", ...]
    }}
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        content = resp.choices[0].message.content.strip()
        content = re.sub(r"^```(json)?", "", content)
        content = re.sub(r"```$", "", content).strip()

        parsed = json.loads(content)

        # Guarantee schema
        return {
            "complete": bool(parsed.get("complete", False)),
            "questions": parsed.get("questions", []),
        }

    except Exception as e:
        print("JD clarifier error:", e)
        # Fall back: assume JD is complete
        return {"complete": True, "questions": []}
