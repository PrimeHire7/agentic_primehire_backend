# app/mcp/tools/jd_history.py

import os
import json
import uuid
import logging
from datetime import datetime, date, time

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import Column, Integer, Text, JSON, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy import text

from sqlalchemy import text
from datetime import datetime, timedelta, timezone
import uuid
import httpx
from fastapi import HTTPException
from app.db import SessionLocal
from app.mcp.tools.interview_attempts_model import InterviewAttempts

from app.db import SessionLocal, engine, Base
from sqlalchemy.orm import joinedload
from collections import defaultdict
from sqlalchemy import Column, Integer, Text, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableList, MutableDict
from datetime import datetime
from app.db import Base
# 👉 Use InterviewAttempts from the external model (correct)
from app.mcp.tools.interview_attempts_model import InterviewAttempts
from app.mcp.tools.resume_tool import Candidate
logger = logging.getLogger(__name__)
router = APIRouter()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------------------------------------------------------------------
# DB MODEL
# --------------------------------------------------------------------
# class GeneratedJDHistory(Base):
#     __tablename__ = "generated_jd_history"

#     id = Column(Integer, primary_key=True)
#     designation = Column(Text, nullable=False)
#     skills = Column(Text, nullable=True)
#     jd_text = Column(Text, nullable=False)
#     created_at = Column(DateTime, default=datetime.utcnow)

#     matches_json = Column(JSON, nullable=True, default=lambda: {"profile_matches": []})
#     matched_candidate_ids = Column(JSON, nullable=True, default=list)

#     ai_questions = Column(JSON, nullable=True, default=list)
#     manual_questions = Column(JSON, nullable=True, default=list)

class GeneratedJDHistory(Base):
    __tablename__ = "generated_jd_history"

    id = Column(Integer, primary_key=True)
    designation = Column(Text, nullable=False)
    skills = Column(Text, nullable=True)
    jd_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    matches_json = Column(MutableDict.as_mutable(JSONB), nullable=True, default=lambda: {
        "ai_questions": [],
        "profile_matches": [],
        "manual_questions": []
    })

    matched_candidate_ids = Column(MutableList.as_mutable(JSONB), nullable=True, default=list)

    ai_questions = Column(MutableList.as_mutable(JSONB), nullable=True, default=list)
    manual_questions = Column(MutableList.as_mutable(JSONB), nullable=True, default=list)

    total_matched = Column(Integer, default=0)
    status = Column(String, default="Active")
    availability = Column(String, default="Open")

Base.metadata.create_all(bind=engine)

# ----------------------------
# Helpers
# ----------------------------
def _ensure_matches_json(row: GeneratedJDHistory) -> dict:
    raw = row.matches_json
    try:
        if raw is None:
            return {"profile_matches": []}
        if isinstance(raw, dict):
            raw.setdefault("profile_matches", [])
            return raw
        if isinstance(raw, list):
            return {"profile_matches": raw}
        if isinstance(raw, str):
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                parsed.setdefault("profile_matches", [])
                return parsed
            return {"profile_matches": parsed}
        return {"profile_matches": []}
    except:
        return {"profile_matches": []}

# --------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------
class SaveJDRequest(BaseModel):
    designation: str
    skills: str | None = None
    jd_text: str

class ManualQuestionModel(BaseModel):
    questions: list[str]

# --------------------------------------------------------------------
# SAVE JD
# --------------------------------------------------------------------
@router.post("/jd/save")
async def save_jd(payload: SaveJDRequest):
    session = SessionLocal()
    try:
        entry = GeneratedJDHistory(
            designation=payload.designation.strip(),
            skills=payload.skills or "",
            jd_text=payload.jd_text,
            # matches_json={"profile_matches": []},
            matches_json={
        "profile_matches": [],
        "ai_questions": [],
        "manual_questions": []
    },
            matched_candidate_ids=[],
            ai_questions=[],
            manual_questions=[],
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return {"ok": True, "jd_id": entry.id}

    except Exception:
        session.rollback()
        logger.exception("Failed to save JD")
        raise HTTPException(status_code=500, detail="Failed to save JD")

    finally:
        session.close()

# --------------------------------------------------------------------
# LIST ALL JDs
# --------------------------------------------------------------------
@router.get("/jd/history")
async def list_jd_history():
    session = SessionLocal()
    try:
        rows = session.query(GeneratedJDHistory).order_by(
            GeneratedJDHistory.created_at.desc()
        ).all()

        history = []
        for r in rows:
            history.append({
                "id": r.id,
                "designation": r.designation,
                "skills": r.skills,
                "created_at": r.created_at.isoformat(),
                "match_count": len(r.matched_candidate_ids or []),
                "preview": (r.jd_text[:150] + "..."),
            })

        return {"history": history}
    finally:
        session.close()

# --------------------------------------------------------------------
# GET SINGLE JD
# --------------------------------------------------------------------
@router.get("/jd/history/{jd_id}")
async def get_single_jd(jd_id: int):
    session = SessionLocal()
    try:
        row = session.query(GeneratedJDHistory).filter_by(id=jd_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="JD not found")

        mj = _ensure_matches_json(row)

        return {
            "id": row.id,
            "designation": row.designation,
            "skills": row.skills,
            "jd_text": row.jd_text,
            "created_at": row.created_at.isoformat(),
            "matches": mj.get("profile_matches", []),
            "matched_candidate_ids": row.matched_candidate_ids or [],
            "manual_questions": row.manual_questions or [],
            "ai_questions": row.ai_questions or [],
        }
    finally:
        session.close()

# --------------------------------------------------------------------
# UPDATE JD
# --------------------------------------------------------------------
@router.post("/jd/update/{jd_id}")
async def update_jd(jd_id: int, payload: dict):
    session = SessionLocal()
    try:
        row = session.query(GeneratedJDHistory).filter_by(id=jd_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="JD not found")

        row.designation = payload.get("designation", row.designation)
        row.skills = payload.get("skills", row.skills)
        row.jd_text = payload.get("jd_text", row.jd_text)
        row.created_at = datetime.utcnow()
        session.commit()

        return {"ok": True}

    except Exception:
        session.rollback()
        raise HTTPException(status_code=500, detail="Failed to update JD")

    finally:
        session.close()

# --------------------------------------------------------------------
# SAVE MANUAL QUESTIONS
# --------------------------------------------------------------------
@router.post("/jd/save_manual_questions/{jd_id}")
async def save_manual_questions(jd_id: int, payload: ManualQuestionModel):
    session = SessionLocal()
    try:
        row = session.query(GeneratedJDHistory).filter_by(id=jd_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="JD not found")

        row.manual_questions = payload.questions
        row.created_at = datetime.utcnow()
        session.commit()

        return {"ok": True, "questions": payload.questions}

    except Exception:
        session.rollback()
        raise HTTPException(status_code=500, detail="Failed to save manual questions")

    finally:
        session.close()

# --------------------------------------------------------------------
# GENERATE AI QUESTIONS
# --------------------------------------------------------------------
@router.post("/jd/generate_ai_questions/{jd_id}")
async def generate_ai_questions(jd_id: int):
    session = SessionLocal()
    try:
        row = session.query(GeneratedJDHistory).filter_by(id=jd_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="JD not found")

        jd_text = row.jd_text.strip()
        if not jd_text:
            raise HTTPException(status_code=400, detail="JD text empty")

        prompt = f"Generate EXACTLY 10 technical interview questions.\nJD:\n{jd_text}"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )

        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail="OpenAI error")

        raw = resp.json()["choices"][0]["message"]["content"]
        q = [line.strip() for line in raw.splitlines() if line.strip()][:10]

        row.ai_questions = q
        row.created_at = datetime.utcnow()
        session.commit()

        return {"ok": True, "questions": q}

    finally:
        session.close()

# --------------------------------------------------------------------
# MATCH PROFILES
# --------------------------------------------------------------------
# @router.post("/jd/match_profiles/{jd_id}")
# async def match_profiles(jd_id: int):

#     print("\n=============== JD MATCH PROFILES START ===============")

#     session = SessionLocal()
#     try:
#         row = session.query(GeneratedJDHistory).filter_by(id=jd_id).first()
#         if not row:
#             raise HTTPException(status_code=404, detail="JD not found")

#         from app.mcp.tools import matcher
#         result = await matcher.match_candidates_tool(row.jd_text)

#         all_candidates = result.get("candidates", [])
#         matched_ids = [c["candidate_id"] for c in all_candidates]

#         row.matches_json = {"profile_matches": all_candidates}
#         row.matched_candidate_ids = matched_ids
#         row.created_at = datetime.utcnow()
#         session.commit()

#         return {
#             "ok": True,
#             "jd_id": jd_id,
#             "match_count": len(all_candidates),
#             "matched_candidate_ids": matched_ids,
#             "matches": all_candidates,
#             "jd_meta": result.get("jd_meta", {})
#         }

#     except Exception as e:
#         session.rollback()
#         raise HTTPException(status_code=500, detail=str(e))

#     finally:
#         session.close()
@router.post("/jd/match_profiles/{jd_id}")
async def match_profiles(jd_id: int):

    print("\n=============== JD MATCH PROFILES START ===============")

    session = SessionLocal()
    try:
        row = session.query(GeneratedJDHistory).filter_by(id=jd_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="JD not found")

        # ✅ IMPORT THE NEW GPT-ONLY MATCHER
        from app.mcp.tools.matcher_tool import match_candidates_tool

        # ✅ CALL THE NEW MATCHER
        result = await match_candidates_tool(row.jd_text)

        all_candidates = result.get("candidates", [])
        matched_ids = [c["candidate_id"] for c in all_candidates]

        # ✅ SAVE INTO DB
        row.matches_json = {"profile_matches": all_candidates}
        row.matched_candidate_ids = matched_ids
        row.created_at = datetime.utcnow()
        session.commit()

        return {
            "ok": True,
            "jd_id": jd_id,
            "match_count": len(all_candidates),
            "matched_candidate_ids": matched_ids,
            "matches": all_candidates,
            "jd_meta": result.get("jd_meta", {})
        }

    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        session.close()

# --------------------------------------------------------------------
# SCHEDULER — CREATE ATTEMPT + SEND EMAIL
# --------------------------------------------------------------------
# @router.post("/scheduler/schedule")
# async def schedule_interview(payload: dict):

#     session = SessionLocal()
#     try:
#         candidate_id = payload.get("candidate_id")
#         # try:
#         #     jd_id = int(payload.get("jd_id"))
#         # except:
#         #     raise HTTPException(status_code=400, detail="Invalid jd_id or missing")
#         jd_id_raw = payload.get("jd_id")

#         try:
#             jd_id = int(jd_id_raw)
#         except:
#             jd_id = None  # ⭐ allow JD-less mode

#         start_iso = payload.get("start_iso")
#         end_iso = payload.get("end_iso")
#         slot_minutes = int(payload.get("slot_minutes", 20))

#         if not candidate_id or not start_iso or not end_iso:
#             raise HTTPException(status_code=400, detail="Missing fields")

#         slot_start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
#         slot_end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))

#         progress_status = "Screening Scheduled"
#         token = str(uuid.uuid4())

#         attempt = InterviewAttempts(
#             candidate_id=candidate_id,
#             jd_id=jd_id,
#             scheduled_date=slot_start.date(),
#             slot_start=slot_start,
#             slot_end=slot_end,
#             slot_duration_min=slot_minutes,
#             progress=progress_status,
#             interview_token=token,
#         )
#         session.add(attempt)
#         session.commit()
#         session.refresh(attempt)

#         print("[SCHEDULER] Created attempt:", attempt.attempt_id)

#         # ------------------------------------------------
#         # Lookup candidate email
#         # ------------------------------------------------
#         try:
#             cand = session.execute(
#     text("SELECT full_name, email FROM candidates WHERE candidate_id=:cid"),
#     {"cid": candidate_id}
# ).fetchone()

#             candidate_name = cand[0] if cand else candidate_id
#             candidate_email = cand[1] if cand else None

#         except Exception as e:
#             print("[SCHEDULER] Email lookup failed:", e)
#             candidate_name = candidate_id
#             candidate_email = None

#         interview_link = (
#             f"https://primehire-beta-ui.vercel.app/validation_panel?"
#             f"candidateId={candidate_id}&jd_id={jd_id}&token={token}"
#         )

#         message_text = (
#     f"Hi {candidate_name},\n\n"
#     f"Your interview is scheduled from {slot_start} to {slot_end} (UTC).\n\n"
#     f"Start your interview here:\n{interview_link}\n\n"
#     f"DO NOT share this link — it has your personal interview token.\n\n"
#     "Thanks,\nPrimeHire Team"
# )


#         # Send email
#         if candidate_email:
#             async with httpx.AsyncClient(timeout=10.0) as client:
#                 await client.post(
#                     "http://127.0.0.1:8000/mcp/tools/match/send_mail",
#                     json={
#                         "email": candidate_email,
#                         "candidate_name": candidate_name,
#                         "message": message_text,
#                     }
#                 )
#         else:
#             print("[SCHEDULER] No email for candidate.")

#         return {
#             "ok": True,
#             "attempt_id": attempt.attempt_id,
#             "interview_token": token,
#             "start_iso": start_iso,
#             "end_iso": end_iso
#         }

#     except Exception as e:
#         session.rollback()
#         print("[SCHEDULER] ERROR:", e)
#         raise HTTPException(status_code=500, detail=str(e))

#     finally:
#         session.close()

@router.post("/scheduler/schedule")
async def schedule_interview(payload: dict):

    session = SessionLocal()
    try:
        print("\n\n================ SCHEDULER START ================")
        print("📩 Incoming Payload:", payload)

        candidate_id = payload.get("candidate_id")
        candidate_name = payload.get("candidate_name") or candidate_id
        candidate_email = payload.get("candidate_email")  # might be email or empty

        jd_id_raw = payload.get("jd_id")

        # ------------------------------------------------
        # Normalize JD ID
        # ------------------------------------------------
        try:
            jd_id = int(jd_id_raw)
        except:
            jd_id = None   # allow JD-less scheduling

        # ------------------------------------------------
        # Required fields
        # ------------------------------------------------
        start_iso = payload.get("start_iso")
        end_iso = payload.get("end_iso")
        slot_minutes = int(payload.get("slot_minutes", 20))

        if not candidate_id or not start_iso or not end_iso:
            raise HTTPException(status_code=400, detail="Missing required fields")

        # Convert ISO UTC → datetime
        slot_start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        slot_end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))

        # ------------------------------------------------
        # Create InterviewAttempt
        # ------------------------------------------------
        token = str(uuid.uuid4())

        attempt = InterviewAttempts(
            candidate_id=candidate_id,
            jd_id=jd_id,
            scheduled_date=slot_start.date(),
            slot_start=slot_start,
            slot_end=slot_end,
            slot_duration_min=slot_minutes,
            progress="Screening Scheduled",
            interview_token=token,
        )

        session.add(attempt)
        session.commit()
        session.refresh(attempt)

        print("✅ Created attempt:", attempt.attempt_id)

        # =====================================================
        # UNIVERSAL EMAIL RESOLUTION LOGIC
        # =====================================================

        print("\n--- EMAIL RESOLUTION ---")
        print("candidate_id:", candidate_id)
        print("candidate_email(payload):", candidate_email)

        # CASE 1 → frontend already sent candidate_email
        if candidate_email and "@" in candidate_email:
            print("✔ Using candidate_email from payload")

        # CASE 2 → candidate_id itself is an EMAIL
        elif "@" in candidate_id:
            candidate_email = candidate_id
            print("✔ candidate_id is email →", candidate_email)

        # CASE 3 → Lookup by candidate_id in DB
        else:
            print("🔍 Looking up candidate in DB using UUID candidate_id")
            try:
                row = session.execute(
                    text("SELECT full_name, email FROM candidates WHERE candidate_id=:cid"),
                    {"cid": candidate_id}
                ).fetchone()

                if row:
                    candidate_name = row[0] or candidate_name
                    candidate_email = row[1]
                    print("✔ DB email found:", candidate_email)
                else:
                    print("⚠ No DB match for candidate_id")

            except Exception as e:
                print("❌ DB email lookup error:", e)
                candidate_email = None

        print("📧 FINAL EMAIL:", candidate_email)
        print("-----------------------------")

        # =====================================================
        # Convert UTC → IST for sending email
        # =====================================================
        ist = timezone(timedelta(hours=5, minutes=30))
        slot_start_ist = slot_start.astimezone(ist)
        slot_end_ist = slot_end.astimezone(ist)

        # =====================================================
        # Email Message
        # =====================================================
        jd_token = payload.get("jd_token")  # <-- add this above

        interview_link = (
            f"https://primehire-beta-ui.vercel.app/validation_panel?"
            f"candidateId={candidate_id}&jd_id={jd_id}&jd_token={jd_token}"
        )


        message_text = (
            f"Hi {candidate_name},\n\n"
            f"Your interview is scheduled from {slot_start_ist} to {slot_end_ist} (IST).\n\n"
            f"Start interview here:\n{interview_link}\n\n"
            "⚠ DO NOT share this link — it contains your unique interview token.\n\n"
            "Thanks,\nPrimeHire Team"
        )

        # =====================================================
        # Send Email
        # =====================================================
        if candidate_email:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        "http://127.0.0.1:8000/mcp/tools/match/send_mail",
                        json={
                            "email": candidate_email,
                            "candidate_name": candidate_name,
                            "message": message_text,
                        }
                    )
                print("📨 Email sent:", resp.status_code)

            except Exception as e:
                print("❌ EMAIL SEND ERROR:", e)

        else:
            print("⚠ No candidate email found → skipping email.")

        print("=============== END SCHEDULER ===============\n")

        return {
            "ok": True,
            "attempt_id": attempt.attempt_id,
            "interview_token": token,
            "start_iso": start_iso,
            "end_iso": end_iso
        }

    except Exception as e:
        session.rollback()
        print("❌ [SCHEDULER ERROR]:", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        session.close()

# --------------------------------------------------------------------
# FETCH ATTEMPT BY TOKEN
# --------------------------------------------------------------------
@router.get("/scheduler/get_attempt/{token}")
async def get_attempt_by_token(token: str):
    session = SessionLocal()
    try:
        row = session.query(InterviewAttempts).filter_by(interview_token=token).first()
        if not row:
            raise HTTPException(status_code=404, detail="Attempt not found")

        return {
            "ok": True,
            "attempt": {
                "attempt_id": row.attempt_id,
                "candidate_id": row.candidate_id,
                "jd_id": row.jd_id,
                "slot_start": row.slot_start.isoformat(),
                "slot_end": row.slot_end.isoformat(),
                "progress": row.progress,
                "interview_token": row.interview_token,
            }
        }

    finally:
        session.close()

# --------------------------------------------------------------------
# FETCH ALL INTERVIEW ATTEMPTS (for dashboard table)
# --------------------------------------------------------------------
@router.get("/scheduler/attempts/all")
async def list_all_attempts():
    session = SessionLocal()
    try:
        rows = session.execute(text("""
            SELECT 
                ia.attempt_id,
                ia.candidate_id,
                ia.jd_id,
                ia.progress,
                ia.slot_start,
                ia.slot_end,
                ia.slot_duration_min,
                ia.ai_score,
                ia.manual_score,
                ia.skill_score,
                ia.interview_score,
                c.full_name,
                c.email
            FROM interview_attempts ia
            LEFT JOIN candidates c ON ia.candidate_id = c.candidate_id
            ORDER BY ia.created_at DESC
        """)).fetchall()

        attempts = []
        for r in rows:
            total_score = (r.interview_score or 0.0)

            attempts.append({
                "attempt_id": r.attempt_id,
                "candidate_id": r.candidate_id,
                "jd_id": r.jd_id,
                "name": r.full_name or r.candidate_id,
                "email": r.email,
                "progress": r.progress,
                "slot_start": r.slot_start.isoformat() if r.slot_start else None,
                "slot_end": r.slot_end.isoformat() if r.slot_end else None,
                "totalScore": round(total_score, 1),
                "sectionScores": [
                    round(r.ai_score or 0.0, 1),
                    round(r.manual_score or 0.0, 1),
                    round(r.skill_score or 0.0, 1),
                    round(r.interview_score or 0.0, 1),
                ],
            })

        return {"ok": True, "attempts": attempts}

    finally:
        session.close()

# --------------------------------------------------------------------
# DASHBOARD STATISTICS (status + performance buckets)
# --------------------------------------------------------------------
@router.get("/scheduler/statistics")
async def scheduler_statistics():
    session = SessionLocal()
    try:
        rows = session.execute(text("""
            SELECT 
                ia.progress,
                ia.interview_score,
                ia.candidate_id
            FROM interview_attempts ia
        """)).fetchall()

        total_attempts = len(rows)
        distinct_candidates = len({r.candidate_id for r in rows if r.candidate_id})

        # status counts by DB progress
        status_counts = {}
        completed_count = 0

        # performance buckets based on interview_score
        perf_buckets = {
            "excellent": 0,  # 80–100
            "good": 0,       # 60–79
            "average": 0,    # 40–59
            "poor": 0,       # 1–39
            "no_score": 0,   # 0 or None
        }

        for r in rows:
            prog = (r.progress or "Applied").strip()
            status_counts[prog] = status_counts.get(prog, 0) + 1

            score = r.interview_score
            if score is None or score <= 0:
                perf_buckets["no_score"] += 1
            elif score >= 80:
                perf_buckets["excellent"] += 1
            elif score >= 60:
                perf_buckets["good"] += 1
            elif score >= 40:
                perf_buckets["average"] += 1
            else:
                perf_buckets["poor"] += 1

            if prog.lower().startswith("completed") or prog.lower().startswith("completed_round1"):
                completed_count += 1

        # Map DB progress → UI categories
        ui_status = {
            "Disconnected": status_counts.get("disconnected", 0),
            "Not Started": status_counts.get("Applied", 0),
            "Blocked": status_counts.get("blocked", 0),
            "Completed": sum(
                v for k, v in status_counts.items()
                if k.lower().startswith("completed")
            ),
            "In Progress": sum(
                v for k, v in status_counts.items()
                if "progress" in k.lower()
            ),
            "Test Stopped": status_counts.get("test_stopped", 0),
            "Yet To Start": status_counts.get("slot_booked_round1", 0),
        }

        return {
            "ok": True,
            "totals": {
                "total_links": len({(r.candidate_id, ) for r in rows}),
                "total_takers": distinct_candidates,
            },
            "reports": {
                "completed": completed_count,
                "total": total_attempts,
            },
            "status_counts": ui_status,
            "performance": perf_buckets,
        }

    finally:
        session.close()

# --------------------------------------------------------------------
# FETCH SINGLE ATTEMPT DETAIL (for CandidateOverview)
# --------------------------------------------------------------------
# @router.get("/scheduler/attempt_detail/{attempt_id}")
# async def attempt_detail(attempt_id: int):
#     session = SessionLocal()
#     try:
#         attempt = session.query(InterviewAttempts).filter_by(attempt_id=attempt_id).first()
#         if not attempt:
#             raise HTTPException(status_code=404, detail="Attempt not found")

#         # candidate
#         cand = session.execute(
#             text("SELECT * FROM candidates WHERE candidate_id = :cid"),
#             {"cid": attempt.candidate_id},
#         ).fetchone()

#         # JD
#         jd = session.query(GeneratedJDHistory).filter_by(id=attempt.jd_id).first()

#         return {
#             "ok": True,
#             "attempt": {
#                 "attempt_id": attempt.attempt_id,
#                 "candidate_id": attempt.candidate_id,
#                 "jd_id": attempt.jd_id,
#                 "slot_start": attempt.slot_start.isoformat() if attempt.slot_start else None,
#                 "slot_end": attempt.slot_end.isoformat() if attempt.slot_end else None,
#                 "progress": attempt.progress,
#                 "ai_score": attempt.ai_score,
#                 "manual_score": attempt.manual_score,
#                 "skill_score": attempt.skill_score,
#                 "interview_score": attempt.interview_score,
#                 "interview_round": attempt.interview_round,
#                 "interview_token": attempt.interview_token,
#                 "created_at": attempt.created_at.isoformat() if attempt.created_at else None,
#                 "updated_at": attempt.updated_at.isoformat() if attempt.updated_at else None,
#                 "anomalies": attempt.anomalies or [],
#             },
#             "candidate": {
#                 "full_name": cand.full_name if cand else attempt.candidate_id,
#                 "email": cand.email if cand else None,
#                 "phone": getattr(cand, "phone", None) if cand else None,
#                 "current_title": getattr(cand, "current_title", None) if cand else None,
#                 "current_company": getattr(cand, "current_company", None) if cand else None,
#                 "resume_link": getattr(cand, "resume_link", None) if cand else None,
#             } if cand else None,
#             "jd": {
#                 "id": jd.id,
#                 "designation": jd.designation,
#                 "skills": jd.skills,
#                 "jd_text": jd.jd_text,
#                 "created_at": jd.created_at.isoformat() if jd.created_at else None,
#             } if jd else None,
#         }

#     finally:
#         session.close()
# @router.get("/scheduler/attempt_detail/{attempt_id}")
# async def get_attempt_detail(attempt_id: int):
#     session = SessionLocal()
#     try:
#         row = session.execute(text("""
#             SELECT
#                 ia.attempt_id,
#                 ia.candidate_id,
#                 ia.jd_id,
#                 ia.status,
#                 ia.slot_start,
#                 ia.slot_end,
#                 ia.interview_score,
#                 ia.ai_score,
#                 ia.manual_score,
#                 ia.skill_score,
#                 ia.anomalies,
#                 ia.interview_token,
#                 ia.created_at,
#                 ia.updated_at,

#                 c.full_name,
#                 c.email,
#                 c.phone,
#                 c.current_title,
#                 c.current_company,
#                 c.years_of_experience,
#                 c.top_skills,
#                 c.location,
#                 c.resume_link,

#                 jd.id as jd_id,
#                 jd.designation,
#                 jd.jd_text

#             FROM interview_attempts ia

#             LEFT JOIN candidates c
#               ON c.email = ia.candidate_id
#               OR c.candidate_id = ia.candidate_id

#             LEFT JOIN generated_jd_history jd
#               ON jd.id = ia.jd_id

#             WHERE ia.attempt_id = :aid
#         """), {"aid": attempt_id}).fetchone()

#         if not row:
#             raise HTTPException(status_code=404, detail="Attempt not found")

#         # ---------------------------
#         # RESOLVE EMAIL + NAME
#         # ---------------------------
#         email = row.email or (row.candidate_id if "@" in row.candidate_id else None)
#         name = row.full_name or (email.split("@")[0] if email else "Candidate")

#         return {
#             "ok": True,

#             "attempt": {
#                 "attempt_id": row.attempt_id,
#                 "candidate_id": row.candidate_id,
#                 "status": row.status,
#                 "slot_start": row.slot_start,
#                 "slot_end": row.slot_end,
#                 "interview_score": row.interview_score,
#                 "ai_score": row.ai_score,
#                 "manual_score": row.manual_score,
#                 "skill_score": row.skill_score,
#                 "anomalies": row.anomalies or [],
#                 "interview_token": row.interview_token,
#                 "created_at": row.created_at,
#                 "updated_at": row.updated_at,
#             },

#             "candidate": {
#                 "full_name": name,
#                 "email": email,
#                 "phone": row.phone,
#                 "current_title": row.current_title,
#                 "current_company": row.current_company,
#                 "years_of_experience": row.years_of_experience,
#                 "top_skills": row.top_skills,
#                 "location": row.location,
#                 "resume_link": row.resume_link,
#             },

#             "jd": {
#                 "id": row.jd_id,
#                 "designation": row.designation,
#                 "jd_text": row.jd_text,
#             }
#         }

#     finally:
#         session.close()
@router.get("/scheduler/attempt_detail/{attempt_id}")
async def get_attempt_detail(attempt_id: int):
    session = SessionLocal()
    try:
        row = session.execute(text("""
            SELECT
                ia.attempt_id,
                ia.candidate_id,
                ia.jd_id,
                ia.status,
                ia.slot_start,
                ia.slot_end,
                ia.interview_score,
                ia.ai_score,
                ia.manual_score,
                ia.technical_reason,
                ia.communication_reason,
                ia.behaviour_reason,
                ia.skill_score,
                ia.anomalies,
                ia.anomaly_summary,
                ia.mcq,
                ia.coding,
                ia.per_question,
                ia.feedback,
                ia.interview_token,
                ia.created_at,
                ia.updated_at,

                c.full_name,
                c.email,
                c.phone,
                c.current_title,
                c.current_company,
                c.years_of_experience,
                c.top_skills,
                c.location,
                c.resume_link,

                jd.id AS jd_id,
                jd.designation,
                jd.jd_text

            FROM interview_attempts ia

            LEFT JOIN candidates c
              ON c.email = ia.candidate_id
              OR c.candidate_id = ia.candidate_id

            LEFT JOIN generated_jd_history jd
              ON jd.id = ia.jd_id

            WHERE ia.attempt_id = :aid
        """), {"aid": attempt_id}).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Attempt not found")

        # ---------------------------
        # RESOLVE EMAIL + NAME
        # ---------------------------
        email = row.email or (row.candidate_id if "@" in row.candidate_id else None)
        name = row.full_name or (email.split("@")[0] if email else "Candidate")

        return {
            "ok": True,

            # ---------------------------
            # ATTEMPT (STATE + SUMMARY)
            # ---------------------------
            "attempt": {
                "attempt_id": row.attempt_id,
                "candidate_id": row.candidate_id,
                "status": row.status,
                "slot_start": row.slot_start,
                "slot_end": row.slot_end,
                "interview_score": row.interview_score,
                "ai_score": row.ai_score,
                "manual_score": row.manual_score,
                "skill_score": row.skill_score,
                "anomalies": row.anomalies or [],
                "anomaly_summary": row.anomaly_summary or {},
                "interview_token": row.interview_token,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            },

            # ---------------------------
            # EVALUATION (FULL DETAILS)
            # ---------------------------
            "evaluation": {
                "technical": row.ai_score or 0,
                "technical_reason": row.technical_reason,
                "communication": row.manual_score or 0,
                "communication_reason": row.communication_reason,
                "behaviour": row.skill_score or 0,
                "behaviour_reason": row.behaviour_reason,
                "overall": row.interview_score or 0,
                "feedback": row.feedback,
                "mcq": row.mcq,
                "coding": row.coding,
                "per_question": row.per_question or [],
            },

            # ---------------------------
            # CANDIDATE
            # ---------------------------
            "candidate": {
                "full_name": name,
                "email": email,
                "phone": row.phone,
                "current_title": row.current_title,
                "current_company": row.current_company,
                "years_of_experience": row.years_of_experience,
                "top_skills": row.top_skills,
                "location": row.location,
                "resume_link": row.resume_link,
            },

            # ---------------------------
            # JOB DESCRIPTION
            # ---------------------------
            "jd": {
                "id": row.jd_id,
                "designation": row.designation,
                "jd_text": row.jd_text,
            }
        }

    finally:
        session.close()

# # --------------------------------------------------------------------
# # FETCH INTERVIEW ATTEMPTS FOR A SPECIFIC JD
# # --------------------------------------------------------------------
# @router.get("/scheduler/attempts/{jd_id}")
# async def list_attempts_by_jd(jd_id: int):
#     session = SessionLocal()
#     try:
#         rows = session.execute(text("""
#             SELECT 
#                 ia.attempt_id,
#                 ia.candidate_id,
#                 ia.jd_id,
#                 ia.progress,
#                 ia.slot_start,
#                 ia.slot_end,
#                 ia.slot_duration_min,
#                 ia.ai_score,
#                 ia.manual_score,
#                 ia.skill_score,
#                 ia.interview_score,
#                 c.full_name,
#                 c.email,
#                 jd.designation
#             FROM interview_attempts ia
#             LEFT JOIN candidates c ON ia.candidate_id = c.candidate_id
#             LEFT JOIN generated_jd_history jd ON ia.jd_id = jd.id
#             WHERE ia.jd_id = :jid
#             ORDER BY ia.created_at DESC
#         """), {"jid": jd_id}).fetchall()

#         attempts = []
#         for r in rows:
#             total_score = (r.interview_score or 0.0)

#             attempts.append({
#                 "attempt_id": r.attempt_id,
#                 "candidate_id": r.candidate_id,
#                 "jd_id": r.jd_id,
#                 "designation": r.designation,  # ⭐ new field
#                 "name": r.full_name or r.candidate_id,
#                 "email": r.email,
#                 "progress": r.progress,
#                 "slot_start": r.slot_start.isoformat() if r.slot_start else None,
#                 "slot_end": r.slot_end.isoformat() if r.slot_end else None,
#                 "totalScore": round(total_score, 1),
#             })

#         return {"ok": True, "attempts": attempts}

#     finally:
#         session.close()

# --------------------------------------------------------------------
# FETCH INTERVIEW ATTEMPTS FOR A SPECIFIC JD (UPDATED)
# --------------------------------------------------------------------
# @router.get("/scheduler/attempts/{jd_id}")
# async def list_attempts_by_jd(jd_id: int):
#     session = SessionLocal()
#     try:
#         rows = session.execute(text("""
#             SELECT 
#                 ia.attempt_id,
#                 ia.candidate_id,
#                 ia.jd_id,
#                 ia.status,              -- ✅ NEW
#                 ia.slot_start,
#                 ia.slot_end,
#                 ia.ai_score,
#                 ia.manual_score,
#                 ia.skill_score,
#                 ia.interview_score,
#                 c.full_name,
#                 c.email,
#                 jd.designation
#             FROM interview_attempts ia
#             LEFT JOIN candidates c 
#                 ON ia.candidate_id = c.candidate_id
#             LEFT JOIN generated_jd_history jd 
#                 ON ia.jd_id = jd.id
#             WHERE ia.jd_id = :jid
#             ORDER BY ia.created_at DESC
#         """), {"jid": jd_id}).fetchall()

#         attempts = []
#         for r in rows:
#             total_score = r.interview_score if r.interview_score is not None else None

#             attempts.append({
#                 "attempt_id": r.attempt_id,
#                 "candidate_id": r.candidate_id,
#                 "jd_id": r.jd_id,
#                 "designation": r.designation,
#                 "name": r.full_name or r.candidate_id,
#                 "email": r.email,

#                 # ✅ USE STATUS (NOT PROGRESS)
#                 "status": r.status,

#                 "slot_start": r.slot_start.isoformat() if r.slot_start else None,
#                 "slot_end": r.slot_end.isoformat() if r.slot_end else None,

#                 "totalScore": round(total_score, 1) if total_score is not None else None,
#             })

#         return {"ok": True, "attempts": attempts}

#     finally:
#         session.close()

# --------------------------------------------------------------------
# FETCH INTERVIEW ATTEMPTS FOR A SPECIFIC JD (FIXED)
# --------------------------------------------------------------------
# @router.get("/scheduler/attempts/{jd_id}")
# async def list_attempts_by_jd(jd_id: int):
#     session = SessionLocal()
#     try:
#         rows = session.execute(text("""
#             SELECT 
#                 ia.attempt_id,
#                 ia.candidate_id,
#                 ia.jd_id,
#                 ia.status,
#                 ia.slot_start,
#                 ia.slot_end,
#                 ia.ai_score,
#                 ia.manual_score,
#                 ia.skill_score,
#                 ia.interview_score,
#                 c.full_name,
#                 c.email,
#                 jd.designation
#             FROM interview_attempts ia
#             LEFT JOIN candidates c 
#                 ON c.email = ia.candidate_id   -- ✅ IMPORTANT FIX
#             LEFT JOIN generated_jd_history jd 
#                 ON ia.jd_id = jd.id
#             WHERE ia.jd_id = :jid
#             ORDER BY ia.created_at DESC
#         """), {"jid": jd_id}).fetchall()

#         attempts = []
#         for r in rows:
#             total_score = r.interview_score or 0.0

#             resolved_email = r.email or r.candidate_id
#             resolved_name = r.full_name or resolved_email.split("@")[0]

#             attempts.append({
#                 "attempt_id": r.attempt_id,
#                 "candidate_id": r.candidate_id,
#                 "jd_id": r.jd_id,
#                 "designation": r.designation,
#                 "name": resolved_name,              # ✅ FIXED
#                 "email": resolved_email,            # ✅ FIXED
#                 "status": r.status,                 # ✅ FIXED
#                 "slot_start": r.slot_start.isoformat() if r.slot_start else None,
#                 "slot_end": r.slot_end.isoformat() if r.slot_end else None,
#                 "totalScore": round(total_score, 1),
#             })

#         return {"ok": True, "attempts": attempts}

#     finally:
#         session.close()

# @router.get("/scheduler/attempts/{jd_id}")
# async def list_attempts_by_jd(jd_id: int):
#     session = SessionLocal()
#     try:
#         rows = session.execute(text("""
#             SELECT 
#                 ia.attempt_id,
#                 ia.candidate_id,
#                 ia.jd_id,
#                 ia.status,
#                 ia.slot_start,
#                 ia.slot_end,
#                 ia.interview_score,
#                 c.full_name,
#                 c.email,
#                 jd.designation
#             FROM interview_attempts ia
#             LEFT JOIN candidates c 
#                 ON ia.candidate_id = c.candidate_id
#                 OR ia.candidate_id = c.email
#             LEFT JOIN generated_jd_history jd 
#                 ON ia.jd_id = jd.id
#             WHERE ia.jd_id = :jid
#             ORDER BY ia.created_at DESC
#         """), {"jid": jd_id}).fetchall()

#         attempts = []
#         for r in rows:
#             # -------------------------------
#             # RESOLVE EMAIL
#             # -------------------------------
#             email = r.email
#             if not email and r.candidate_id and "@" in r.candidate_id:
#                 email = r.candidate_id

#             # -------------------------------
#             # RESOLVE NAME
#             # -------------------------------
#             name = r.full_name
#             if not name:
#                 name = email or r.candidate_id

#             attempts.append({
#                 "attempt_id": r.attempt_id,
#                 "candidate_id": r.candidate_id,
#                 "jd_id": r.jd_id,
#                 "designation": r.designation,
#                 "name": name,          # ✅ FIXED
#                 "email": email,        # ✅ FIXED
#                 "status": r.status,
#                 "slot_start": r.slot_start.isoformat() if r.slot_start else None,
#                 "slot_end": r.slot_end.isoformat() if r.slot_end else None,
#                 "totalScore": round(r.interview_score or 0, 1),
#             })

#         return {"ok": True, "attempts": attempts}

#     finally:
#         session.close()




@router.get("/scheduler/attempts/{jd_id}")
async def list_attempts_by_jd(jd_id: int):
    session = SessionLocal()
    try:
        attempts_q = (
            session.query(InterviewAttempts)
            .filter(InterviewAttempts.jd_id == jd_id)
            .order_by(InterviewAttempts.created_at.desc())
            .all()
        )

        attempts = []
        status_counts = defaultdict(int)

        for a in attempts_q:
            status = (a.status or "UNKNOWN").upper()
            status_counts[status] += 1

            # -------------------------------
            # Resolve Candidate
            # -------------------------------
            candidate = (
                session.query(Candidate)
                .filter(
                    (Candidate.candidate_id == a.candidate_id) |
                    (Candidate.email == a.candidate_id)
                )
                .first()
            )

            name = (
                candidate.full_name
                if candidate and candidate.full_name
                else candidate.email
                if candidate
                else a.candidate_id
            )

            email = (
                candidate.email
                if candidate and candidate.email
                else a.candidate_id
                if "@" in a.candidate_id
                else None
            )

            # -------------------------------
            # Resolve JD
            # -------------------------------
            jd = (
                session.query(GeneratedJDHistory)
                .filter(GeneratedJDHistory.id == a.jd_id)
                .first()
            )

            attempts.append({
                "attempt_id": a.attempt_id,
                "candidate_id": a.candidate_id,
                "jd_id": a.jd_id,
                "designation": jd.designation if jd else None,
                "name": name,
                "email": email,
                "status": status,
                "slot_start": a.slot_start.isoformat() if a.slot_start else None,
                "slot_end": a.slot_end.isoformat() if a.slot_end else None,
                "totalScore": round(a.interview_score or 0, 1),
            })

        return {
            "ok": True,
            "total_candidates": len(attempts),
            "counts_by_status": dict(status_counts),
            "attempts": attempts,
        }

    finally:
        session.close()


@router.get("/scheduler/jd_stats/{jd_id}")
async def jd_attempt_stats(jd_id: int):
    session = SessionLocal()
    try:
        rows = session.execute(text("""
            SELECT status, COUNT(*) 
            FROM interview_attempts
            WHERE jd_id = :jid
            GROUP BY status
        """), {"jid": jd_id}).fetchall()

        stats = {r[0]: r[1] for r in rows}

        return {
            "ok": True,
            "stats": {
                "SCHEDULED": stats.get("SCHEDULED", 0),
                "IN_PROGRESS": stats.get("IN_PROGRESS", 0),
                "COMPLETED": stats.get("COMPLETED", 0),
                "EXPIRED": stats.get("EXPIRED", 0),
            }
        }
    finally:
        session.close()


@router.get("/scheduler/latest_attempt/{candidate_id}/{jd_id}")
async def latest_attempt(candidate_id: str, jd_id: int):
    """
    Correct, stable version:
    - Uses REAL candidate_id (NOT phone)
    - Filters by BOTH candidate_id + jd_id
    - Returns last attempt with status, round, time
    """

    session = SessionLocal()
    try:
        row = (
            session.query(InterviewAttempts)
            .filter(
                InterviewAttempts.candidate_id == candidate_id,
                InterviewAttempts.jd_id == jd_id,
            )
            .order_by(InterviewAttempts.created_at.desc())
            .first()
        )

        if not row:
            return {"ok": False, "status": "No Attempt"}

        return {
            "ok": True,
            "progress": row.progress or "Not Started",
            "interview_round": row.interview_round or 1,
            "slot_start": row.slot_start.isoformat() if row.slot_start else None,
            "slot_end": row.slot_end.isoformat() if row.slot_end else None,
        }

    finally:
        session.close()

__all__ = ["router"]
