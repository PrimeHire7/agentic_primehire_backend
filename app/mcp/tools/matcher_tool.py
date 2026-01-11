
# matcher_tool.py — Full: No soft filters, GPT-only scoring (parallel async), dedupe, moderate skill inference
import os
import re
import json
import logging
import asyncio
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from app.mcp.server_core import register_mcp_tool


from fastapi import APIRouter, HTTPException, Query, Body, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
import re, json
from collections import defaultdict
import smtplib
from email.message import EmailMessage
import imaplib
import email
from email.header import decode_header
import os
import tempfile
import re
import traceback
from datetime import datetime, timedelta
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from threading import Lock

from app.mcp.tools.jd_history import GeneratedJDHistory, _ensure_matches_json

load_dotenv()
logger = logging.getLogger(__name__)
DEBUG_MATCHER = True

from app.mcp.tools.matcher_history import save_match_to_db
RESPONSES_FILE = os.path.join(os.path.dirname(__file__), "whatsapp_responses.json")
# ---------------- Configuration ----------------
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "primehire-production-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

TOP_K = 500                  # Option B style coverage
EMBEDDING_DIM = 3072
INFER_SKILLS_COUNT = (5, 8)  # moderate inference 5-8 skills

# ---------------- Clients ----------------
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure Pinecone index exists (best-effort)
try:
    idx_names = pc.list_indexes().names()
except Exception as e:
    logger.exception("Failed to list pinecone indexes: %s", e)
    idx_names = []

if PINECONE_INDEX not in idx_names:
    try:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    except Exception as e:
        logger.exception("Pinecone create_index failed: %s", e)

index = pc.Index(PINECONE_INDEX)

# ---------------- FastAPI Router ----------------
router = APIRouter()

class MatchRequest(BaseModel):
    jd_text: str

# ---------------- Utilities ----------------
def debug_log(*args):
    if DEBUG_MATCHER:
        print("[MATCHER DEBUG]", *args)

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\bllms\b", "llm", s)
    s = re.sub(r"\bgen\s*ai\b", "generative ai", s)
    s = re.sub(r"[\_\|\/,;]+", " ", s)
    s = re.sub(r"[^\w\s\-\+\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str):
    return [t for t in normalize_text(s).split() if t]

def skill_tokenize(skill: str):
    s = normalize_text(skill)
    toks = [t for t in re.split(r"[\s\-\_\.\/]+", s) if t]
    return toks

# ---------------- Embedding ----------------
def get_embedding(text: str):
    resp = client.embeddings.create(model="text-embedding-3-large", input=text)
    return resp.data[0].embedding

# ---------------- JD extraction + moderate skill inference (5-8 skills) ----------------
async def extract_jd_metadata(jd_text: str) -> dict:
    extraction_prompt = f"""
    You are a structured data extractor. Return ONLY valid JSON with keys:
    role (string), location (string or empty), years_experience (integer or 0), skills (list of strings).

    Job description:
    \"\"\"{jd_text}\"\"\"
    """
    jd_meta = {"role": "", "location": "", "years_experience": 0, "skills": []}
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        content = re.sub(r"^```(json)?", "", content)
        content = re.sub(r"```$", "", content).strip()
        parsed = json.loads(content)
        jd_meta["role"] = parsed.get("role", "") or ""
        jd_meta["location"] = parsed.get("location", "") or ""
        try:
            jd_meta["years_experience"] = int(parsed.get("years_experience") or 0)
        except:
            jd_meta["years_experience"] = 0
        jd_meta["skills"] = [normalize_text(s) for s in parsed.get("skills", []) if s]
    except Exception as e:
        debug_log("JD extractor failed or returned non-json:", e)
        # Heuristic fallback
        lines = [ln.strip() for ln in jd_text.splitlines() if ln.strip()]
        if lines:
            jd_meta["role"] = lines[0][:120]
        m = re.search(r"(\d{1,2})(?:\+)?\s+years?", jd_text, flags=re.I)
        if m:
            try:
                jd_meta["years_experience"] = int(m.group(1))
            except:
                jd_meta["years_experience"] = 0
        mloc = re.search(r"in\s+([A-Za-z\s,]{2,60})", jd_text)
        if mloc:
            jd_meta["location"] = mloc.group(1).strip()
        jd_meta["skills"] = []

    # If skills empty or too few, infer moderate set (5-8)
    if not jd_meta.get("skills") or len(jd_meta.get("skills", [])) < INFER_SKILLS_COUNT[0]:
        inferred = await infer_skills_from_role_moderate(jd_meta.get("role", "") or jd_text)
        if inferred:
            if not jd_meta.get("skills"):
                jd_meta["skills"] = inferred
            else:
                combined = list(dict.fromkeys(jd_meta["skills"] + inferred))
                jd_meta["skills"] = combined[:INFER_SKILLS_COUNT[1]]

    jd_meta["skills"] = list({normalize_text(s) for s in jd_meta["skills"] if s})
    return jd_meta

async def infer_skills_from_role_moderate(role_or_text: str) -> list:
    """
    Produce a moderate list of 5-8 skills for the given role or JD text.
    """
    prompt = f"""
    You are an expert job-skill generator.
    Given this job title or description, return ONLY a JSON array of 5-8 concise skills (strings).
    Example: ["python","pytorch","nlp","transformers","docker"]
    Input: \"\"\"{role_or_text}\"\"\"
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        content = re.sub(r"^```(json)?", "", content)
        content = re.sub(r"```$", "", content).strip()
        parsed = json.loads(content)
        skills = [normalize_text(s) for s in parsed if isinstance(s, str) and s.strip()]
        if len(skills) < INFER_SKILLS_COUNT[0]:
            return skills
        return skills[:INFER_SKILLS_COUNT[1]]
    except Exception as e:
        debug_log("infer_skills_from_role_moderate failed:", e)
        return []

def extract_skills_from_text(text: str, limit=8) -> list:
    common_skills = [
        "python", "java", "c++", "react", "node", "sql", "aws", "docker",
        "fastapi", "django", "pytorch", "tensorflow", "nlp", "computer vision",
        "llm", "transformers", "langchain", "spark", "kubernetes", "mlops"
    ]
    t = text.lower()
    found = [s for s in common_skills if s in t]
    if found:
        return found[:limit]
    tokens = re.findall(r"\b[a-zA-Z\-\+]{4,}\b", text)
    unique = []
    for tok in tokens:
        nt = normalize_text(tok)
        if nt and nt not in unique:
            unique.append(nt)
        if len(unique) >= limit:
            break
    return unique

# ---------------- Dedup helpers ----------------
def candidate_unique_key_from_metadata(meta: dict, pine_id: str) -> str:
    """
    Construct robust unique key priority:
    1) phone (digits only)
    2) email (lower)
    3) name + location (lower)
    4) pinecone id fallback
    """
    phone = meta.get("phone") or meta.get("mobile") or ""
    phone_digits = re.sub(r"\D+", "", str(phone)) if phone else ""
    if phone_digits:
        return f"phone:{phone_digits}"

    email = meta.get("email") or ""
    if email:
        return f"email:{normalize_text(email)}"

    name = meta.get("full_name") or meta.get("name") or ""
    location = meta.get("location") or ""
    if name:
        key = f"name:{normalize_text(name)}"
        if location:
            key += f"|loc:{normalize_text(location)}"
        return key

    return f"pineid:{pine_id}"

# ---------------- GPT scoring (parallel) ----------------
def _clean_json_from_model(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(json)?", "", t)
    t = re.sub(r"```$", "", t).strip()
    return t

def gpt_score_candidate_sync(jd_meta: dict, resume_meta: dict) -> dict:
    """
    Blocking GPT call.
    Weights:
    - Designation semantic match: 60%
    - Skills match: 30%
    - Experience relevance: 10%
    """

    prompt = f"""
You are an expert hiring assistant.

Your task is to rate how well the candidate matches the job.

PRIORITY & WEIGHTS:
1. Designation / Role semantic match — 60%
2. Skills match — 30%
3. Experience relevance — 10%

Rules:
- Designation is most important, but closely related roles (e.g. Backend Engineer vs Software Engineer) should still score moderately.

- Skills are secondary.
- Experience is tertiary and only fine-tunes the score.

Return STRICT JSON ONLY:
{{
  "designation_score": 0-100,
  "skills_score": 0-100,
  "experience_score": 0-100,
  "final_score": 0-100,
  "reason": "max 150 chars"
}}

JOB:
Role: {jd_meta.get('role')}
Skills: {jd_meta.get('skills')}
Years Experience: {jd_meta.get('years_experience')}

CANDIDATE:
Designation: {resume_meta.get('designation')}
Skills: {resume_meta.get('skills')}
Years Experience: {resume_meta.get('experience_years')}

Compute:
final_score = designation_score*0.60 + skills_score*0.30 + experience_score*0.10

Return JSON only.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        raw = resp.choices[0].message.content.strip()
        raw = _clean_json_from_model(raw)
        parsed = json.loads(raw)

        def clamp(v):
            try:
                return max(0, min(100, int(round(float(v)))))
            except:
                return 0

        designation_score = clamp(parsed.get("designation_score"))
        skills_score = clamp(parsed.get("skills_score"))
        experience_score = clamp(parsed.get("experience_score"))

        final_score = clamp(
            designation_score * 0.60 +
            skills_score * 0.30 +
            experience_score * 0.10
        )

        reason = str(parsed.get("reason", ""))[:150]

        return {
            "designation_score": designation_score,
            "skills_score": skills_score,
            "experience_score": experience_score,
            "final_score": final_score,
            "reason": reason,
            "raw": raw,
        }

    except Exception as e:
        debug_log("gpt scoring failed:", resume_meta.get("name"), e)
        return {
            "designation_score": 0,
            "skills_score": 0,
            "experience_score": 0,
            "final_score": 0,
            "reason": f"gpt_error: {str(e)[:120]}",
            "raw": str(e),
        }


async def gpt_score_candidate(jd_meta: dict, resume_meta: dict) -> dict:
    # run blocking GPT call in thread to enable concurrency with blocking client
    return await asyncio.to_thread(gpt_score_candidate_sync, jd_meta, resume_meta)


# ---------------- Optimization helpers ----------------

# TOP_K_OPT = 150
# MIN_SEMANTIC_SCORE = 0.25
# MAX_GPT_PARALLEL = 6
# JD_CACHE = {}
# _gpt_semaphore = asyncio.Semaphore(MAX_GPT_PARALLEL)

# def _jd_cache_key(jd_text: str) -> str:
#     return re.sub(r"\s+", " ", jd_text.strip().lower())[:500]

# async def extract_jd_metadata_cached(jd_text: str):
#     key = _jd_cache_key(jd_text)
#     if key in JD_CACHE:
#         return JD_CACHE[key]
#     meta = await extract_jd_metadata(jd_text)
#     JD_CACHE[key] = meta
#     return meta

# async def gpt_score_candidate_limited(jd_meta, resume_meta):
#     async with _gpt_semaphore:
#         return await gpt_score_candidate(jd_meta, resume_meta)

# # @register_mcp_tool(
# #     name="matcher.match_candidates",
# #     description="Match JD text against candidate resumes from Pinecone (GPT-only scoring, async)."
# # )
# # async def match_candidates_tool(jd_text: str):
# #     jd_meta = await extract_jd_metadata(jd_text)
# #     debug_log("JD Meta:", jd_meta)

# #     emb = get_embedding(jd_text)
# #     debug_log("Embedding length:", len(emb))

# #     # --------------------------------------------------
# #     # PINECONE QUERY
# #     # --------------------------------------------------
# #     results = index.query(
# #         vector=emb,
# #         top_k=TOP_K,
# #         include_metadata=True,
# #         namespace="__default__"
# #     )
# #     matches = results.get("matches", [])
# #     debug_log("Pinecone matches:", len(matches))

# #     # --------------------------------------------------
# #     # DEDUPLICATION
# #     # --------------------------------------------------
# #     dedup_map = {}
# #     for m in matches:
# #         pine_id = m.get("id")
# #         meta = m.get("metadata") or {}
# #         unique_key = candidate_unique_key_from_metadata(meta, pine_id)

# #         existing = dedup_map.get(unique_key)
# #         if not existing or (m.get("score", 0) or 0) > (existing.get("score", 0) or 0):
# #             dedup_map[unique_key] = m

# #     debug_log("Deduped candidates count:", len(dedup_map))

# #     # --------------------------------------------------
# #     # PREPARE CANDIDATES
# #     # --------------------------------------------------
# #     candidates = []
# #     for _, m in dedup_map.items():
# #         meta = m.get("metadata") or {}

# #         exp_raw = (
# #             meta.get("years_of_experience")
# #             or meta.get("experience_years")
# #             or meta.get("total_experience_years")
# #             or 0
# #         )
# #         try:
# #             exp_candidate = float(str(exp_raw).replace("+", "").strip())
# #         except:
# #             exp_candidate = 0.0

# #         skills_raw = meta.get("top_skills") or meta.get("skills") or ""
# #         if isinstance(skills_raw, str):
# #             cand_skills = [normalize_text(s) for s in skills_raw.split(",") if s.strip()]
# #         elif isinstance(skills_raw, list):
# #             cand_skills = [normalize_text(s) for s in skills_raw if s]
# #         else:
# #             cand_skills = []

# #         designation = meta.get("current_title") or meta.get("designation") or ""

# #         resume_meta = {
# #             "name": meta.get("full_name") or meta.get("name") or "",
# #             "designation": normalize_text(designation),
# #             "location": meta.get("location") or "",
# #             "skills": cand_skills,
# #             "email": meta.get("email") or "",
# #             "phone": meta.get("phone") or "",
# #             "linkedin": meta.get("linkedin_url") or "",
# #             "experience_years": exp_candidate,
# #             "_semantic_score": m.get("score", 0),
# #             "_pine_id": m.get("id"),
# #         }

# #         candidates.append((m, resume_meta))

# #     debug_log("Candidates prepared for GPT scoring:", len(candidates))

# #     # --------------------------------------------------
# #     # GPT SCORING (UNCHANGED)
# #     # --------------------------------------------------
# #     tasks = [gpt_score_candidate(jd_meta, resume_meta) for (_, resume_meta) in candidates]

# #     final_candidates = []
# #     if tasks:
# #         scored_results = await asyncio.gather(*tasks, return_exceptions=False)

# #         for ((m, resume_meta), score_res) in zip(candidates, scored_results):
# #             final_score = int(score_res.get("final_score", 0))
# #             reason = score_res.get("reason", "") or ""
# #             raw = score_res.get("raw", "")

# #             final_candidates.append({
# #                 "name": resume_meta["name"],
# #                 "designation": resume_meta["designation"],
# #                 "experience_years": resume_meta["experience_years"],
# #                 "location": resume_meta["location"],
# #                 "skills": resume_meta["skills"],
# #                 "email": resume_meta["email"],
# #                 "phone": resume_meta["phone"],
# #                 "linkedin": resume_meta["linkedin"],
# #                 "candidate_id": (
# #                     resume_meta.get("email")
# #                     or resume_meta.get("phone")
# #                     or resume_meta.get("name")
# #                     or resume_meta.get("_pine_id")
# #                 ),
# #                 "semantic": m.get("score"),
# #                 "scores": {"final_score": final_score},
# #                 "finalScore": final_score,
# #                 "reason": reason,
# #                 "gpt_raw": raw,
# #             })

# #     # --------------------------------------------------
# #     # SORT + FILTER
# #     # --------------------------------------------------
# #     final_candidates.sort(key=lambda x: x["finalScore"], reverse=True)

# #     MIN_SCORE = 35  # allow partial but relevant matches

# #     all_scored = final_candidates.copy()  # keep full list BEFORE filtering

# #     # Debug BEFORE filtering
# #     debug_log("Sample scores (top 5 before filter):", [
# #         {"name": c["name"], "score": c["finalScore"], "semantic": c.get("semantic")}
# #         for c in all_scored[:5]
# #     ])

# #     # Apply threshold
# #     final_candidates = [c for c in final_candidates if c["finalScore"] >= MIN_SCORE]

# #     if not final_candidates:
# #         debug_log("⚠ No candidates >= threshold, falling back to top semantic matches")
# #         final_candidates = sorted(
# #             all_scored,
# #             key=lambda x: (x.get("semantic") or 0),
# #             reverse=True
# #         )[:10]

# #     debug_log("Final shortlisted candidates:", len(final_candidates))


# #     # ==================================================
# #     # ✅ SAVE INTO GENERATED_JD_HISTORY (NEW PART)
# #     # ==================================================
# #     from app.db import SessionLocal
# #     from app.mcp.tools.jd_history import GeneratedJDHistory, _ensure_matches_json

# #     session = SessionLocal()
# #     try:
# #         # 🔍 Reuse JD if exists
# #         jd_row = (
# #             session.query(GeneratedJDHistory)
# #             .filter(GeneratedJDHistory.designation == jd_meta.get("role"))
# #             .order_by(GeneratedJDHistory.created_at.desc())
# #             .first()
# #         )

# #         # ➕ Create JD if not exists
# #         if not jd_row:
# #             jd_row = GeneratedJDHistory(
# #                 designation=jd_meta.get("role") or "Unknown Role",
# #                 skills=", ".join(jd_meta.get("skills", [])),
# #                 jd_text=jd_text,
# #                 matches_json={"profile_matches": []},
# #                 matched_candidate_ids=[],
# #             )
# #             session.add(jd_row)
# #             session.commit()
# #             session.refresh(jd_row)

# #         # 💾 Save matches
# #         mj = _ensure_matches_json(jd_row)
# #         mj["profile_matches"] = final_candidates
# #         jd_row.matches_json = mj
# #         jd_row.matched_candidate_ids = [
# #             c["candidate_id"] for c in final_candidates if c.get("candidate_id")
# #         ]
        

# #         session.commit()

# #         debug_log(f"✅ Saved matches to JD_HISTORY (jd_id={jd_row.id})")

# #     finally:
# #         session.close()

# #     # --------------------------------------------------
# #     # RETURN (BACKWARD COMPATIBLE)
# #     # --------------------------------------------------
# #     return {
# #         "jd_id": jd_row.id,
# #         "jd_meta": jd_meta,
# #         "candidates": final_candidates,
# #         "match_count": len(final_candidates),
# #     }

# @register_mcp_tool(
#     name="matcher.match_candidates",
#     description="Match JD text against candidate resumes from Pinecone (GPT-only scoring, async)."
# )
# async def match_candidates_tool(jd_text: str):
#     start_time = datetime.utcnow()

#     jd_meta = await extract_jd_metadata_cached(jd_text)
#     debug_log("JD Meta:", jd_meta)

#     emb = get_embedding(jd_text)

#     results = index.query(
#         vector=emb,
#         top_k=TOP_K_OPT,
#         include_metadata=True,
#         namespace="__default__"
#     )
#     matches = results.get("matches", [])
#     debug_log("Pinecone matches:", len(matches))

#     # -------- Dedup + semantic prune --------
#     dedup_map = {}
#     for m in matches:
#         if (m.get("score") or 0) < MIN_SEMANTIC_SCORE:
#             continue

#         meta = m.get("metadata") or {}
#         key = candidate_unique_key_from_metadata(meta, m.get("id"))

#         if key not in dedup_map or (m.get("score") or 0) > (dedup_map[key].get("score") or 0):
#             dedup_map[key] = m

#     # -------- Hard cap BEFORE GPT --------
#     deduped = list(dedup_map.values())
#     deduped.sort(key=lambda x: x.get("score") or 0, reverse=True)
#     deduped = deduped[:40]   # 👈 HARD LIMIT (major speedup)

#     debug_log("Candidates after dedup+cap:", len(deduped))

#     # -------- Prepare candidates --------
#     candidates = []
#     for m in deduped:
#         meta = m.get("metadata") or {}

#         try:
#             exp = float(str(
#                 meta.get("years_of_experience")
#                 or meta.get("experience_years")
#                 or meta.get("total_experience_years")
#                 or 0
#             ).replace("+", ""))
#         except:
#             exp = 0.0

#         skills_raw = meta.get("top_skills") or meta.get("skills") or ""
#         if isinstance(skills_raw, str):
#             skills = [normalize_text(s) for s in skills_raw.split(",") if s.strip()]
#         elif isinstance(skills_raw, list):
#             skills = [normalize_text(s) for s in skills_raw if s]
#         else:
#             skills = []

#         resume_meta = {
#             "name": meta.get("full_name") or meta.get("name") or "",
#             "designation": normalize_text(meta.get("current_title") or meta.get("designation") or ""),
#             "location": meta.get("location") or "",
#             "skills": skills,
#             "email": meta.get("email") or "",
#             "phone": meta.get("phone") or "",
#             "linkedin": meta.get("linkedin_url") or "",
#             "experience_years": exp,
#             "_semantic_score": m.get("score"),
#             "_pine_id": m.get("id"),
#         }

#         candidates.append((m, resume_meta))

#     # -------- GPT scoring (limited parallel) --------
#     TOP_GPT = 15  # only GPT score top 15 semantic matches

#     candidates.sort(key=lambda x: x[1]["_semantic_score"] or 0, reverse=True)

#     gpt_candidates = candidates[:TOP_GPT]
#     rest_candidates = candidates[TOP_GPT:]

#     tasks = [gpt_score_candidate_limited(jd_meta, rm) for (_, rm) in gpt_candidates]
#     gpt_results = await asyncio.gather(*tasks)

#     scored_results = []
#     for ((m, rm), score_res) in zip(gpt_candidates, gpt_results):
#         scored_results.append((m, rm, score_res))

#     # cheap fallback scoring for rest
#     for (m, rm) in rest_candidates:
#         semantic = rm.get("_semantic_score") or 0
#         approx_score = int(min(100, max(30, semantic * 100)))
#         scored_results.append((
#             m,
#             rm,
#             {"final_score": approx_score, "reason": "semantic fallback"}
#         ))


#     final_candidates = []
#     for (m, resume_meta, score_res) in scored_results:

#         final_score = int(score_res.get("final_score", 0))
#         final_candidates.append({
#             "name": rm["name"],
#             "designation": rm["designation"],
#             "experience_years": rm["experience_years"],
#             "location": rm["location"],
#             "skills": rm["skills"],
#             "email": rm["email"],
#             "phone": rm["phone"],
#             "linkedin": rm["linkedin"],
#             "candidate_id": rm.get("email") or rm.get("phone") or rm.get("_pine_id"),
#             "semantic": rm["_semantic_score"],
#             "scores": {"final_score": final_score},
#             "finalScore": final_score,
#             "reason": score_res.get("reason", ""),
#         })

#     # -------- Sort + filter --------
#     final_candidates.sort(key=lambda x: x["finalScore"], reverse=True)
#     MIN_SCORE = 35
#     all_scored = final_candidates.copy()

#     final_candidates = [c for c in final_candidates if c["finalScore"] >= MIN_SCORE]

#     if not final_candidates:
#         final_candidates = sorted(all_scored, key=lambda x: x.get("semantic") or 0, reverse=True)[:10]

#     # ---------------- SAVE TO DB ----------------
#     from app.db import SessionLocal
#     session = SessionLocal()

#     try:
#         jd_row = (
#             session.query(GeneratedJDHistory)
#             .filter(GeneratedJDHistory.designation == jd_meta.get("role"))
#             .order_by(GeneratedJDHistory.created_at.desc())
#             .first()
#         )

#         if not jd_row:
#             jd_row = GeneratedJDHistory(
#                 designation=jd_meta.get("role") or "Unknown Role",
#                 skills=", ".join(jd_meta.get("skills", [])),
#                 jd_text=jd_text,
#                 matches_json={"profile_matches": []},
#                 matched_candidate_ids=[],
#             )
#             session.add(jd_row)
#             session.commit()
#             session.refresh(jd_row)

#         mj = _ensure_matches_json(jd_row)
#         mj["profile_matches"] = final_candidates
#         jd_row.matches_json = mj
#         jd_row.matched_candidate_ids = [
#             c["candidate_id"] for c in final_candidates if c.get("candidate_id")
#         ]
#         session.commit()

#         jd_id = jd_row.id  # ✅ store while session is alive

#     finally:
#         session.close()

#     logger.info(f"[MATCHER] Took {(datetime.utcnow()-start_time).total_seconds():.2f}s")

#     return {
#         "jd_id": jd_id,
#         "jd_meta": jd_meta,
#         "candidates": final_candidates,
#         "match_count": len(final_candidates),
#     }



# # ---------------- FastAPI endpoint ----------------
# @router.post("/profile/match")
# async def profile_match(req: MatchRequest):
#     """
#     Thin endpoint.
#     Delegates everything to match_candidates_tool:
#     - JD creation / reuse
#     - Matching
#     - DB persistence
#     """

#     logger.info("PROFILE_MATCH received")

#     try:
#         result = await match_candidates_tool(req.jd_text)

#         # result already contains:
#         # {
#         #   jd_id,
#         #   jd_meta,
#         #   candidates,
#         #   match_count
#         # }

#         return result

#     except Exception as e:
#         logger.exception("Error in /profile/match")
#         raise HTTPException(status_code=500, detail=str(e))

# ===================== MATCHER TOOL =====================

TOP_K_OPT = 150
MIN_SEMANTIC_SCORE = 0.25
TOP_GPT = 15
MIN_SCORE = 35

JD_CACHE = {}
_gpt_semaphore = asyncio.Semaphore(6)


def _jd_cache_key(jd_text: str) -> str:
    return re.sub(r"\s+", " ", jd_text.strip().lower())[:500]


async def extract_jd_metadata_cached(jd_text: str):
    key = _jd_cache_key(jd_text)
    if key in JD_CACHE:
        return JD_CACHE[key]
    meta = await extract_jd_metadata(jd_text)
    JD_CACHE[key] = meta
    return meta


async def gpt_score_candidate_limited(jd_meta, resume_meta):
    async with _gpt_semaphore:
        return await gpt_score_candidate(jd_meta, resume_meta)


@register_mcp_tool(
    name="matcher.match_candidates",
    description="Match JD text against candidate resumes from Pinecone (fast hybrid)."
)
async def match_candidates_tool(jd_text: str):
    start_time = datetime.utcnow()

    jd_meta = await extract_jd_metadata_cached(jd_text)
    debug_log("JD Meta:", jd_meta)

    emb = get_embedding(jd_text)

    results = index.query(
        vector=emb,
        top_k=TOP_K_OPT,
        include_metadata=True,
        namespace="__default__"
    )
    matches = results.get("matches", [])
    debug_log("Pinecone matches:", len(matches))

    # ---------- Dedup + prune ----------
    dedup_map = {}
    for m in matches:
        if (m.get("score") or 0) < MIN_SEMANTIC_SCORE:
            continue

        meta = m.get("metadata") or {}
        key = candidate_unique_key_from_metadata(meta, m.get("id"))

        if key not in dedup_map or (m.get("score") or 0) > (dedup_map[key].get("score") or 0):
            dedup_map[key] = m

    deduped = list(dedup_map.values())
    deduped.sort(key=lambda x: x.get("score") or 0, reverse=True)
    deduped = deduped[:40]   # HARD CAP

    debug_log("Candidates after dedup+cap:", len(deduped))

    # ---------- Prepare ----------
    candidates = []
    for m in deduped:
        meta = m.get("metadata") or {}

        try:
            exp = float(str(
                meta.get("years_of_experience")
                or meta.get("experience_years")
                or meta.get("total_experience_years")
                or 0
            ).replace("+", ""))
        except:
            exp = 0.0

        skills_raw = meta.get("top_skills") or meta.get("skills") or ""
        if isinstance(skills_raw, str):
            skills = [normalize_text(s) for s in skills_raw.split(",") if s.strip()]
        elif isinstance(skills_raw, list):
            skills = [normalize_text(s) for s in skills_raw if s]
        else:
            skills = []

        resume_meta = {
            "name": meta.get("full_name") or meta.get("name") or "",
            "designation": normalize_text(meta.get("current_title") or meta.get("designation") or ""),
            "location": meta.get("location") or "",
            "skills": skills,
            "email": meta.get("email") or "",
            "phone": meta.get("phone") or "",
            "linkedin": meta.get("linkedin_url") or "",
            "experience_years": exp,
            "_semantic_score": m.get("score"),
            "_pine_id": m.get("id"),
        }

        candidates.append((m, resume_meta))

    candidates.sort(key=lambda x: x[1]["_semantic_score"] or 0, reverse=True)

    gpt_candidates = candidates[:TOP_GPT]
    rest_candidates = candidates[TOP_GPT:]

    # ---------- GPT on top only ----------
    tasks = [gpt_score_candidate_limited(jd_meta, rm) for (_, rm) in gpt_candidates]
    gpt_results = await asyncio.gather(*tasks)

    scored = []

    for ((m, rm), score_res) in zip(gpt_candidates, gpt_results):
        scored.append((m, rm, score_res))

    # ---------- Cheap fallback for rest ----------
    for (m, rm) in rest_candidates:
        semantic = rm.get("_semantic_score") or 0
        approx_score = int(min(100, max(30, semantic * 100)))
        scored.append((m, rm, {"final_score": approx_score, "reason": "semantic fallback"}))

    # ---------- Build final ----------
    final_candidates = []
    for (m, rm, score_res) in scored:
        final_score = int(score_res.get("final_score", 0))

        final_candidates.append({
            "name": rm["name"],
            "designation": rm["designation"],
            "experience_years": rm["experience_years"],
            "location": rm["location"],
            "skills": rm["skills"],
            "email": rm["email"],
            "phone": rm["phone"],
            "linkedin": rm["linkedin"],
            "candidate_id": rm.get("email") or rm.get("phone") or rm.get("_pine_id"),
            "semantic": rm["_semantic_score"],
            "scores": {"final_score": final_score},
            "finalScore": final_score,
            "reason": score_res.get("reason", ""),
        })

    final_candidates.sort(key=lambda x: x["finalScore"], reverse=True)

    all_scored = final_candidates.copy()
    final_candidates = [c for c in final_candidates if c["finalScore"] >= MIN_SCORE]

    if not final_candidates:
        final_candidates = sorted(all_scored, key=lambda x: x.get("semantic") or 0, reverse=True)[:10]

    # ---------- Save ----------
    from app.db import SessionLocal
    session = SessionLocal()
    try:
        jd_row = (
            session.query(GeneratedJDHistory)
            .filter(GeneratedJDHistory.designation == jd_meta.get("role"))
            .order_by(GeneratedJDHistory.created_at.desc())
            .first()
        )

        if not jd_row:
            jd_row = GeneratedJDHistory(
                designation=jd_meta.get("role") or "Unknown Role",
                skills=", ".join(jd_meta.get("skills", [])),
                jd_text=jd_text,
                matches_json={"profile_matches": []},
                matched_candidate_ids=[],
            )
            session.add(jd_row)
            session.commit()
            session.refresh(jd_row)

        mj = _ensure_matches_json(jd_row)
        mj["profile_matches"] = final_candidates
        jd_row.matches_json = mj
        jd_row.matched_candidate_ids = [c["candidate_id"] for c in final_candidates if c.get("candidate_id")]
        session.commit()

        jd_id = jd_row.id

    finally:
        session.close()

    logger.info(f"[MATCHER] Took {(datetime.utcnow()-start_time).total_seconds():.2f}s")

    return {
        "jd_id": jd_id,
        "jd_meta": jd_meta,
        "candidates": final_candidates,
        "match_count": len(final_candidates),
    }


# ---------------- FastAPI endpoint ----------------

@router.post("/profile/match")
async def profile_match(req: MatchRequest):
    logger.info("PROFILE_MATCH received")
    try:
        return await match_candidates_tool(req.jd_text)
    except Exception as e:
        logger.exception("Error in /profile/match")
        raise HTTPException(status_code=500, detail=str(e))
# ===================== WHATSAPP TOOL =====================
import requests

# Use the latest long-lived token here
WHATSAPP_ACCESS_TOKEN = "EAAU2bZAblWIABP9fKFiinI0uH7asjWMKVzZBqWWsWn4ANNHR9o5wvNeLJa7vZAbSqg6OIW2EqTYgqEVzbip3Cfka48rVeaHObHvxthLY9JSZAsZB1dIUdVH1DzwaDSA7YnFjdh2Swttc3ZBo36ZC6OU0s1ZAwKFrlIZAKxJt3PvTWiH3LV6lCYlkCUXMrXz8v"
WHATSAPP_PHONE_NUMBER_ID = "830820360117790"  # your WhatsApp business phone ID



class WhatsAppPayload(BaseModel):
    phone: str
    candidate_name: str  # used as template variable


@router.post("/send_whatsapp")
async def send_whatsapp_message(payload: WhatsAppPayload):
    # Normalize phone number
    phone = payload.phone.strip().replace("+", "").replace(" ", "")
    candidate_name = payload.candidate_name.strip()

    if not phone or not candidate_name:
        raise HTTPException(status_code=400, detail="Phone or candidate name missing")

    print(f"📤 Sending WhatsApp message to {phone} ({candidate_name})")

    url = f"https://graph.facebook.com/v24.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"

    data = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "template",
        "template": {
            "name": "primehire_test_bot",  # your template name
            "language": {"code": "en"},
            "components": [
                {
                    "type": "body",
                    "parameters": [
                        {
                            "type": "text",
                            "parameter_name": "name",  # ✅ use parameter_name
                            "text": candidate_name,
                        }
                    ],
                }
            ],
        },
    }

    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=data, headers=headers)
    print(f"📨 WhatsApp API response {response.status_code}: {response.text}")

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return {"status": "sent", "response": response.json()}


VERIFY_TOKEN = "primehire123"


@router.get("/webhook/whatsapp")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(content=challenge)
    return PlainTextResponse(content="Verification failed", status_code=403)


# ---------------------------
# Helper: Load & Save JSON
# ---------------------------
def load_responses() -> dict:
    """Load responses from disk (safe)."""
    try:
        if os.path.exists(RESPONSES_FILE):
            with open(RESPONSES_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    print(f"📂 Loaded {len(data)} WhatsApp responses from disk.")
                    return data
                else:
                    print("⚠️ Invalid JSON structure. Resetting.")
    except Exception as e:
        print(f"⚠️ Failed to load responses: {e}")
    return {}


def save_responses():
    """Safely write responses to disk."""
    global whatsapp_responses
    try:
        with _lock:
            with open(RESPONSES_FILE, "w") as f:
                json.dump(whatsapp_responses, f, indent=2)
        print(
            f"💾 Saved {len(whatsapp_responses)} WhatsApp responses to {RESPONSES_FILE}"
        )
    except Exception as e:
        print(f"⚠️ Error saving WhatsApp responses: {e}")


# Helper function to normalize phone numbers
def normalize_phone(phone: str):
    return re.sub(r"\D", "", phone)  # remove +, spaces, non-digits


# ---------------------------
# Global in-memory storage (auto-loaded)
# ---------------------------
whatsapp_responses = load_responses()
print(f"📂 Loaded WhatsApp responses from disk: {len(whatsapp_responses)} entries")




# ---------------------------
# Webhook Endpoint (Receives Messages)
# ---------------------------
@router.post("/webhook/whatsapp")
async def receive_message(request: Request):
    try:
        body = await request.json()
        entry_list = body.get("entry", [])
        if not entry_list:
            return JSONResponse(content={"status": "no entry"}, status_code=400)

        entry = entry_list[0]
        changes_list = entry.get("changes", [])
        if not changes_list:
            return JSONResponse(content={"status": "no changes"}, status_code=400)

        change = changes_list[0]
        value = change.get("value", {})

        if "messages" in value:
            for message in value["messages"]:
                msg_type = message.get("type")
                from_number = normalize_phone(message.get("from"))

                if msg_type == "button":
                    button_payload = message["button"]["payload"]
                    button_text = message["button"]["text"]
                    whatsapp_responses[from_number] = {
                        "type": "button",
                        "payload": button_payload,
                        "text": button_text,
                    }
                    print(f"🟢 {from_number} clicked: {button_text} ({button_payload})")

                elif msg_type == "text":
                    user_text = message["text"]["body"]
                    whatsapp_responses[from_number] = {
                        "type": "text",
                        "text": user_text,
                    }
                    print(f"💬 Text reply from {from_number}: {user_text}")

            # ✅ save to disk right after any update
            save_responses()

    except Exception as e:
        return JSONResponse(
            content={"status": "error", "detail": str(e)}, status_code=500
        )

    return JSONResponse(content={"status": "received"})


@router.get("/whatsapp/responses")
async def get_whatsapp_responses():
    """
    Returns the latest WhatsApp responses.
    Example response:
    {
        "918885999458": {"type": "button", "payload": "Available", "text": "Available"}
    }
    """
    print("📥 Fetching WhatsApp responses", json.dumps(whatsapp_responses, indent=2))

    return JSONResponse(content=whatsapp_responses)




class MailRequest(BaseModel):
    email: str
    candidate_name: str
    message: str


@router.post("/send_mail")
def send_mail(request: MailRequest):
    sender_email = "naresh@primehire.ai"
    sender_password = "Techdeveloper$"
    smtp_server = "smtpout.secureserver.net"
    smtp_port = 465  # SSL port for GoDaddy

    try:
        msg = EmailMessage()
        msg["From"] = sender_email
        msg["To"] = request.email
        msg["Subject"] = "Interview Availability Check"
        msg.set_content(request.message)

        with smtplib.SMTP_SSL(smtp_server, smtp_port) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)

        return {"message": f"Email sent successfully to {request.email}"}

    except smtplib.SMTPAuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid SMTP credentials")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")


VERIFY_TOKEN = "primehire123"


@router.get("/verify_webhook/whatsapp")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge
    return "Verification failed"


# # ================================
# SEND CANDIDATE DETAILS TO CLIENT
# ================================
from .matcher_history import SessionLocal, MatcherHistory
from .jd_history import GeneratedJDHistory
from app.db import SessionLocal as DBSession  # candidate fallback
from app.mcp.tools.resume_tool import Candidate     # only if you have it

from typing import Optional
from pydantic import BaseModel
from fastapi import HTTPException, APIRouter
from email.message import EmailMessage
import smtplib
import traceback

# router = APIRouter()

import re

def strip_html(html: str) -> str:
    if not html:
        return ""

    # Remove the Copy JD button (entire block)
    html = re.sub(r"📋\s*Copy JD", "", html, flags=re.I)
    html = re.sub(r"<button[\s\S]*?<\/button>", "", html, flags=re.I)

    # Remove "How to Apply" section
    html = re.sub(r"<h2>How to Apply[\s\S]*?<\/p>", "", html, flags=re.I)

    # Remove script & style tags
    html = re.sub(r"<script[\s\S]*?<\/script>", "", html, flags=re.I)
    html = re.sub(r"<style[\s\S]*?<\/style>", "", html, flags=re.I)

    # Replace list items
    html = re.sub(r"<li>", "- ", html, flags=re.I)

    # Replace breaks & paragraphs with newline
    html = re.sub(r"<br\s*\/?>", "\n", html, flags=re.I)
    html = re.sub(r"</p>", "\n", html, flags=re.I)

    # Add newline after headings
    html = re.sub(r"</h[1-6]>", "\n", html, flags=re.I)

    # Strip all remaining HTML
    html = re.sub(r"<[^>]+>", "", html)

    # Remove extra newlines
    html = re.sub(r"\n\s*\n\s*\n+", "\n\n", html)

    return html.strip()


class ClientMailPayload(BaseModel):
    client_email: str
    candidate_id: str
    jd_id: Optional[int] = None


@router.post("/send_to_client")
async def send_to_client(payload: ClientMailPayload):

    session = SessionLocal()

    try:
        print("🔥 Received payload:", payload.dict())

        # ================================================
        # CASE 1 — WITH JD ID (existing full flow)
        # ================================================
        if payload.jd_id is not None:
            jd_row = (
                session.query(GeneratedJDHistory)
                .filter(GeneratedJDHistory.id == payload.jd_id)
                .first()
            )
            if not jd_row:
                raise HTTPException(status_code=404, detail="JD not found")

            jd_text = jd_row.jd_text or "Job description unavailable"
            clean_jd = strip_html(jd_text)
            # jd_summary = f"\n\nJD Summary:\n-----------------------------------------\n{jd_text}\n-----------------------------------------"
            jd_summary = (
    "\n\nJD Summary:"
    "\n-----------------------------------------\n"
    f"{clean_jd}\n"
    "-----------------------------------------"
)
        else:
            # ================================================
            # CASE 2 — NO JD ID → direct candidate mode
            # ================================================
            jd_summary = "\n(Note: Sent without JD — direct candidate mode)"

        # ==================================================
        # SEARCH CANDIDATE IN MATCHER HISTORY (primary)
        # ==================================================
        all_matches = (
            session.query(MatcherHistory)
            .order_by(MatcherHistory.created_at.desc())
            .all()
        )

        candidate = None
        for match in all_matches:
            for c in (match.candidates or []):
                if (
                    c.get("candidate_id") == payload.candidate_id
                    or c.get("email") == payload.candidate_id
                    or c.get("phone") == payload.candidate_id
                ):
                    candidate = c
                    break
            if candidate:
                break

        # ==================================================
        # FALLBACK: FETCH FROM candidates TABLE
        # ==================================================
        if not candidate:
            print("⚠ Candidate not found in MatcherHistory — checking candidates table")
            db2 = DBSession()
            row = db2.query(Candidate).filter(Candidate.candidate_id == payload.candidate_id).first()
            if row:
                candidate = {
                    "name": row.full_name,
                    "email": row.email,
                    "phone": row.phone,
                    "designation": row.current_title,
                    "location": row.location,
                    "experience_years": row.years_of_experience,
                    "skills": row.top_skills.split(",") if row.top_skills else [],
                    "finalScore": row.rating_score or 0,
                }
            db2.close()

        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")

        # ==================================================
        # EXTRACT CANDIDATE FIELDS
        # ==================================================
        cname = candidate.get("name", "Candidate")
        cemail = candidate.get("email", "-")
        cphone = candidate.get("phone", "-")
        cloc = candidate.get("location", "-")
        crole = candidate.get("designation", "-")
        cexp = candidate.get("experience_years", "-")
        cskills = ", ".join(candidate.get("skills", []))
        cscore = candidate.get("finalScore", 0)

        # ==================================================
        # BUILD EMAIL
        # ==================================================
        sender_email = "naresh@primehire.ai"
        sender_password = "Techdeveloper$"
        smtp_server = "smtpout.secureserver.net"
        smtp_port = 465

        msg = EmailMessage()
        msg["From"] = sender_email
        msg["To"] = payload.client_email
        msg["Subject"] = f"Candidate Submission – {cname}"

        content = f"""
Hello,

Here are the candidate details shortlisted by PrimeHire:

===============================
       Candidate Summary
===============================

Name: {cname}
Email: {cemail}
Phone: {cphone}
Designation: {crole}
Location: {cloc}
Experience: {cexp} years
Skills: {cskills}
Match Score: {cscore}/100

{jd_summary}

Let us know if you'd like to schedule this candidate.

Regards,
PrimeHire Team
"""

        msg.set_content(content)

        # ==================================================
        # SEND EMAIL
        # ==================================================
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)

        return {
            "ok": True,
            "message": "Client mail sent successfully",
            "mode": "with_jd" if payload.jd_id else "direct_candidate",
            "candidate_id": payload.candidate_id,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Mail failed: {e}")

    finally:
        session.close()

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
