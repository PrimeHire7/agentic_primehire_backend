# ============================================================
# INTERVIEW BOT BETA
# - Handles audio upload, transcription, analysis, question generation
# - New confidence + superficial answer detectors
# - Transcript evaluation with per-question breakdown
# - Interview scheduling endpoint
# - MCQ question generator from JD
# ============================================================
import re
import uuid
import json
import os
import asyncio
import requests
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import assemblyai as aai
from dotenv import load_dotenv

from app.db import SessionLocal
from app.mcp.tools.interview_attempts_model import InterviewAttempts
from sqlalchemy import text
from datetime import datetime


from sqlalchemy import text
from datetime import datetime, timedelta, timezone
import uuid
import httpx
from fastapi import HTTPException
from app.db import SessionLocal
from app.mcp.tools.interview_attempts_model import InterviewAttempts

from sqlalchemy import text
from datetime import datetime, timedelta, timezone
import uuid
import httpx
from fastapi import HTTPException
from app.db import SessionLocal
from app.mcp.tools.interview_attempts_model import InterviewAttempts

import re
import numpy as np
# ----------------------------
# INIT
# ----------------------------
load_dotenv()
router = APIRouter()

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

TRANSCRIPTS_DIR = "logs/interviews"
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# ----------------------------
# UTILITIES
# ----------------------------
def safe_name(s: str):
    return re.sub(r"[^A-Za-z0-9_]", "_", s)

def transcript_file(attempt_id: int):
    return os.path.join(
        TRANSCRIPTS_DIR,
        f"attempt_{attempt_id}.json"
    )


# ----------------------------
# AUDIO TRANSCRIPTION
# ----------------------------
async def transcribe_audio(file_path):
    print("🎙 Transcribing audio:", file_path)

    headers = {"authorization": aai.settings.api_key}
    with open(file_path, "rb") as f:
        upload_res = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers=headers,
            data=f
        )
        audio_url = upload_res.json()["upload_url"]

    transcript_res = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        json={"audio_url": audio_url},
        headers=headers
    )

    transcript_id = transcript_res.json()["id"]
    print("📝 AssemblyAI transcript ID:", transcript_id)

    # Poll until done
    while True:
        res = requests.get(
            f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
            headers=headers
        ).json()

        if res["status"] == "completed":
            print("✅ Transcription completed:", res["text"])
            return res["text"]

        if res["status"] == "error":
            print("❌ Transcription error:", res)
            return None

        await asyncio.sleep(2)

# =============================================
# CONFIDENCE ANALYSIS + FAKE ANSWER DETECTION
# =============================================

FILLERS = ["um", "uh", "like", "you know", "basically", "actually"]

def analyze_confidence(transcript_text):
    words = transcript_text.split()
    wpm = len(words) / 0.5  # approx 30-sec answer → tune later

    filler_count = sum(transcript_text.lower().count(f) for f in FILLERS)

    confidence = 100
    if filler_count >= 3:
        confidence -= (filler_count * 5)
    if wpm < 60:
        confidence -= 20
    if wpm > 180:
        confidence -= 15

    return {
        "wpm": wpm,
        "filler_count": filler_count,
        "confidence_score": max(0, min(100, confidence))
    }


def detect_superficial_answer(answer_text):
    superficial_patterns = [
        r"\bteam player\b",
        r"\bfast learner\b",
        r"\bsynergy\b",
        r"\bpassionate\b",
        r"\bhard worker\b",
        r"\bresults driven\b"
    ]

    buzzword_hits = sum(1 for p in superficial_patterns if re.search(p, answer_text.lower()))

    depth_score = 100 - (buzzword_hits * 10)
    if len(answer_text.split()) < 20:
        depth_score -= 20

    return {
        "buzzword_hits": buzzword_hits,
        "depth_score": max(0, depth_score),
        "is_superficial": buzzword_hits > 2 or depth_score < 50
    }

# ----------------------------
# NEXT QUESTION GENERATOR (GPT)
# ----------------------------
from openai import AsyncOpenAI
client = AsyncOpenAI()

async def generate_next_question(conversation, job_description):
    history_text = "\n".join(
        [f"{msg['sender'].capitalize()}: {msg['text']}" for msg in conversation]
    )

    print("🤖 Generating next question with history:")
    print(history_text)

    system_prompt = f"""
    You are a professional technical interviewer for PrimeHire AI.
    Interview based on:

    {job_description}

    Generate the next best question.
    """

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": history_text}
            ],
            temperature=0.7,
            max_tokens=100
        )
        q = response.choices[0].message.content.strip()
        if not q.endswith("?"):
            q += "?"

        print("➡️ Next question:", q)
        return q

    except Exception as e:
        print("❌ GPT error:", e)
        return "Can you explain one of your recent projects?"

from fastapi import UploadFile, File, Form, HTTPException
from datetime import datetime
import uuid, os, json
from app.db import SessionLocal
from app.mcp.tools.interview_attempts_model import InterviewAttempts
from datetime import timezone

@router.post("/process-answer")
async def process_answer(
    audio: UploadFile = File(None),
    candidate_id: str = Form(...),
    candidate_name: str = Form(...),
    attempt_id: str = Form(...),
    token: str | None = Form(None),
    jd_id: int | None = Form(None),

    job_description: str = Form("N/A"),
    init: str = Form("false"),
):
    print("\n\n================ PROCESS ANSWER ================")
    print("🟡 RAW FORM VALUES")
    print("  init:", init, type(init))
    print("  token:", token)
    print("  candidate:", candidate_name, candidate_id)
    print("  jd_id:", jd_id)
    print("  audio_present:", bool(audio))

    # =========================================================
    # 🔐 1. VALIDATE INTERVIEW ATTEMPT
    # =========================================================
    db = SessionLocal()
    try:
        attempt = (
            db.query(InterviewAttempts)
            .filter(
                InterviewAttempts.candidate_id == candidate_id,
                InterviewAttempts.jd_id == jd_id,
                InterviewAttempts.interview_token == token,
            )
            .order_by(InterviewAttempts.created_at.desc())
            .first()
        )

        print("🟡 ATTEMPT FOUND:", bool(attempt))

        if not attempt:
            print("❌ INVALID ATTEMPT — AUTH FAIL")
            raise HTTPException(403, "Invalid interview link")

        attempt_id = attempt.attempt_id
        print("🟢 ATTEMPT ID:", attempt_id)

        now = datetime.now(timezone.utc)

        # Ensure timezone-safe DB values
        if attempt.slot_start.tzinfo is None:
            attempt.slot_start = attempt.slot_start.replace(tzinfo=timezone.utc)
        if attempt.slot_end.tzinfo is None:
            attempt.slot_end = attempt.slot_end.replace(tzinfo=timezone.utc)

        print("🟡 STATUS:", attempt.status)
        print("🟡 NOW:", now)
        print("🟡 WINDOW:", attempt.slot_start, "→", attempt.slot_end)

        if attempt.status == "LOCKED":
            raise HTTPException(403, "Interview already evaluated")

        if attempt.status == "COMPLETED":
            raise HTTPException(403, "Interview already completed")

        if now < attempt.slot_start:
            raise HTTPException(403, "Interview has not started yet")

        if now > attempt.slot_end:
            attempt.status = "EXPIRED"
            attempt.updated_at = now
            db.commit()
            raise HTTPException(403, "Interview window expired")

        if attempt.status == "SCHEDULED":
            print("🟢 TRANSITION → IN_PROGRESS")
            attempt.status = "IN_PROGRESS"
            attempt.token_used_at = now
            db.commit()

    finally:
        db.close()

    # =========================================================
    # 📁 2. LOAD / INIT TRANSCRIPT (PER ATTEMPT)
    # =========================================================
    json_path = transcript_file(attempt_id)
    print("🟡 TRANSCRIPT PATH:", json_path)

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            interview_data = json.load(f)
        print("🟡 EXISTING TRANSCRIPT LOADED")
    else:
        interview_data = {
            "attempt_id": attempt_id,
            "candidate_id": candidate_id,
            "jd_id": jd_id,
            "job_description": job_description,
            "conversation": [],
        }
        print("🟢 NEW TRANSCRIPT CREATED")

    # =========================================================
    # 🟢 3. INIT CALL → FIRST QUESTION (NO AUDIO)
    # =========================================================
    if init == "true":
        print("🟢 INIT CALL RECEIVED")

        ai_count = sum(1 for x in interview_data["conversation"] if x["sender"] == "ai")
        print("🟡 AI MESSAGE COUNT:", ai_count)

        if ai_count > 0:
            print("🟡 INIT SKIPPED — ALREADY INITIALIZED")
            return {
                "ok": True,
                "candidate_id": candidate_id,
                "completed": False,
            }

        first_q = "Tell me about yourself."
        print("🟢 FIRST QUESTION:", first_q)

        interview_data["conversation"].append({
            "sender": "ai",
            "text": first_q,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

        with open(json_path, "w") as f:
            json.dump(interview_data, f, indent=2)

        print("🟢 FIRST QUESTION SAVED")

        return {
            "ok": True,
            "candidate_id": candidate_id,
            "next_question": first_q,
            "completed": False,
        }

    # =========================================================
    # 🎙 4. AUDIO REQUIRED AFTER INIT
    # =========================================================
    if not audio:
        print("❌ AUDIO MISSING AFTER INIT")
        raise HTTPException(400, "Audio answer is required")

    print("🟢 AUDIO RECEIVED")

    tmp_file = f"temp_{attempt_id}_{uuid.uuid4().hex}.webm"
    with open(tmp_file, "wb") as f:
        f.write(await audio.read())

    transcript_text = await transcribe_audio(tmp_file)
    os.remove(tmp_file)

    print("🟢 TRANSCRIBED TEXT:", transcript_text)

    if not transcript_text:
        raise HTTPException(500, "Transcription failed")

    # =========================================================
    # 🧠 5. ANALYSIS
    # =========================================================
    confidence_data = analyze_confidence(transcript_text)
    superficial_data = detect_superficial_answer(transcript_text)

    print("🟢 ANALYSIS DONE")

    # =========================================================
    # 💾 6. SAVE USER ANSWER
    # =========================================================
    interview_data["conversation"].append({
        "sender": "user",
        "text": transcript_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "analysis": {
            "confidence": confidence_data,
            "superficial_detection": superficial_data,
        },
    })

    # =========================================================
    # 🤖 7. NEXT QUESTION OR COMPLETE
    # =========================================================
    MAX_QUESTIONS = 10
    ai_questions = sum(1 for x in interview_data["conversation"] if x["sender"] == "ai")
    completed = ai_questions >= MAX_QUESTIONS

    if not completed:
        next_q = await generate_next_question(
            interview_data["conversation"],
            job_description,
        )
        print("🟢 NEXT QUESTION:", next_q)

        interview_data["conversation"].append({
            "sender": "ai",
            "text": next_q,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        final_msg = None
    else:
        next_q = None
        final_msg = "Thank you for completing the interview."
        print("🟢 INTERVIEW COMPLETED")

        db = SessionLocal()
        try:
            attempt = (
                db.query(InterviewAttempts)
                .filter(InterviewAttempts.attempt_id == attempt_id)
                .first()
            )
            if attempt:
                attempt.status = "COMPLETED"
                attempt.updated_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()

    with open(json_path, "w") as f:
        json.dump(interview_data, f, indent=2)

    # =========================================================
    # 📤 8. RESPONSE
    # =========================================================
    return {
        "ok": True,
        "candidate_id": candidate_id,
        "transcribed_text": transcript_text,
        "next_question": next_q,
        "completed": completed,
        "final_message": final_msg,
        "analysis": {
            "confidence": confidence_data,
            "superficial": superficial_data,
        },
    }


@router.post("/evaluate-transcript")
async def evaluate_transcript(
    attempt_id: int = Form(...),
    candidate_id: str = Form(...),
    candidate_name: str = Form(...),
    job_description: str = Form("N/A"),
    jd_id: int | None = Form(None),
    designation: str | None = Form(None),
    pass_threshold: int = Form(60),

    # From frontend
    
    mcq_result: str | None = Form(None),
    coding_result: str | None = Form(None),
    anomaly_counts: str | None = Form(None)
):
    """
    FULL FINAL EVALUATION (ATTEMPT-BASED)

    Includes:
    - AI Interview transcript
    - Per-question analysis
    - Face-monitor anomalies
    - MCQ result
    - Coding result
    - Holistic GPT evaluation
    - Locks interview attempt
    """

    print("\n===============================")
    print("EVALUATE TRANSCRIPT")
    print("===============================")
    print("Attempt:", attempt_id)
    print("Candidate:", candidate_name, candidate_id)
    print("mcq_result:", "present" if mcq_result else "none")
    print("coding_result:", "present" if coding_result else "none")
    
    try:
        parsed_anomaly_counts = json.loads(anomaly_counts) if anomaly_counts else {}
    except Exception:
        parsed_anomaly_counts = {}

    debug = {}

    # --------------------------------------------------
    # 1️⃣ LOAD TRANSCRIPT FILE (ATTEMPT-BASED)
    # --------------------------------------------------
    json_path = transcript_file(attempt_id)
    debug["transcript_path"] = json_path

    if not os.path.exists(json_path):
        return {"ok": False, "error": "Transcript not found", "debug": debug}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversation = data.get("conversation", [])
    print("📄 Transcript entries loaded:", len(conversation))
    print("---------------------------------------------------")    
    print(  "Conversation Preview:"     )
    for i, msg in enumerate(conversation[:5]):
        print(f"  {i+1}. {msg['sender'].upper()}: {msg.get('text','')}")
    print("---------------------------------------------------")            
    # --------------------------------------------------
    # 2️⃣ PARSE MCQ & CODING RESULTS
    # --------------------------------------------------
    mcq_data = None
    coding_data = None

    if mcq_result:
        try:
            mcq_data = json.loads(mcq_result)
        except Exception:
            mcq_data = None

    if coding_result:
        try:
            coding_data = json.loads(coding_result)
        except Exception:
            coding_data = None

    # Persist them into transcript
    data["mcq"] = mcq_data
    data["coding"] = coding_data

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # --------------------------------------------------
    # 3️⃣ BUILD TRANSCRIPT TEXT
    # --------------------------------------------------
    convo_text = "\n".join(
        [f"{m['sender'].upper()}: {m.get('text','')}" for m in conversation]
    )

    # --------------------------------------------------
    # 4️⃣ PER-QUESTION ANALYSIS
    # --------------------------------------------------
    FILLERS = ["um", "uh", "like", "you know", "basically", "actually"]

    def analyze_confidence(text: str):
        if not text:
            return 0
        filler_hits = sum(text.lower().count(f) for f in FILLERS)
        score = 100 - filler_hits * 6
        if len(text.split()) < 10:
            score -= 25
        return max(0, min(100, score))

    def analyze_depth(text: str):
        buzz = sum(
            1 for p in [
                r"\bteam player\b",
                r"\bfast learner\b",
                r"\bsynergy\b",
                r"\bpassionate\b",
                r"\bhard worker\b",
            ]
            if re.search(p, text.lower())
        )
        depth = max(0, 100 - buzz * 12)
        return depth, buzz

    per_question = []
    last_ai_q = None

    for msg in conversation:
        if msg.get("sender") == "ai":
            last_ai_q = msg.get("text", "")
        elif msg.get("sender") == "user":
            conf = analyze_confidence(msg.get("text", ""))
            depth, buzz = analyze_depth(msg.get("text", ""))
            auth = int((conf + depth) / 2)

            per_question.append({
                "question": last_ai_q or "",
                "answer": msg.get("text", ""),
                "analysis": {
                    "confidence_score": conf,
                    "depth_score": depth,
                    "buzzword_hits": buzz,
                    "authenticity_score": auth,
                }
            })
            last_ai_q = None

    # --------------------------------------------------
    # 5️⃣ ANOMALY SUMMARY (FROM TRANSCRIPT SYSTEM EVENTS)
    # --------------------------------------------------
    # anomaly_counts = {}
    # 5️⃣ ANOMALY SUMMARY (MERGE FRONTEND + TRANSCRIPT)
    anomaly_counts = dict(parsed_anomaly_counts or {})

    
    # for msg in conversation:
    #     if msg.get("sender") == "system":
    #         key = (
    #             msg.get("text", "")
    #             .replace("⚠", "")
    #             .strip()
    #             .lower()
    #             .replace(" ", "_")
    #         )
    #         anomaly_counts[key] = anomaly_counts.get(key, 0) + 1

    # --------------------------------------------------
    # 6️⃣ GPT HOLISTIC EVALUATION
    # --------------------------------------------------
    prompt = f"""
You are a STRICT, REAL-TIME TECHNICAL PROCTOR and SENIOR INTERVIEWER.

Your role is NOT to be polite.
Your role is to ACCURATELY evaluate competence, authenticity, and depth.

You are evaluating a candidate in a monitored interview environment with:
- live proctoring
- anomaly detection
- time pressure
- no external help allowed

You MUST cross-check consistency across:
1) AI interview transcript
2) MCQ answers
3) Coding solution

--------------------------------------------------
EVALUATION RULES (VERY IMPORTANT)
--------------------------------------------------

1. Penalize superficial answers, buzzwords, and evasion.
2. Penalize contradictions between MCQ, coding, and verbal answers.
3. Penalize hesitation, filler-heavy responses, and vague explanations.
4. Reward precise reasoning, concrete examples, and correct mental models.
5. If coding is incorrect but explanation is strong, assign partial credit.
6. If MCQ answers are correct but transcript shows guessing, penalize.
7. If anomalies suggest possible external help, be conservative in scoring.
8. Assume the candidate has no access to internet, IDE help, or documentation.

DO NOT inflate scores.
Scores above 85 should be RARE and ONLY for truly strong candidates.

--------------------------------------------------
INPUT DATA
--------------------------------------------------

MCQ RESULTS:
{json.dumps(mcq_data, indent=2)}

CODING SUBMISSION:
{json.dumps(coding_data, indent=2)}

AI INTERVIEW TRANSCRIPT:
{convo_text}

--------------------------------------------------
OUTPUT FORMAT (STRICT JSON ONLY — NO MARKDOWN)
--------------------------------------------------

Return EXACTLY this structure:

{{
  "technical": {{
    "score": 0-100,
    "reason": "Concrete justification referencing MCQ, coding, and transcript"
  }},
  "communication": {{
    "score": 0-100,
    "reason": "Clarity, structure, confidence, and precision of explanations"
  }},
  "behaviour": {{
    "score": 0-100,
    "reason": "Authenticity, honesty, composure, and exam integrity"
  }},
  "feedback": "Blunt, professional feedback. Mention strengths, weaknesses, and hire-readiness."
}}

DO NOT include any extra keys.
DO NOT include explanations outside JSON.
DO NOT wrap in markdown.
VERIFY the JSON is complete and valid before responding.
"""


    raw = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.05,
        max_tokens=900,
    )

    raw_text = raw.choices[0].message.content
    debug["gpt_raw"] = raw_text

    try:
        scores = json.loads(raw_text)
    except Exception:
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        scores = json.loads(cleaned)

    def clamp(v):
        return max(0, min(100, int(float(v))))

    technical = clamp(scores["technical"]["score"])
    communication = clamp(scores["communication"]["score"])
    behaviour = clamp(scores["behaviour"]["score"])

    overall = round((technical + communication + behaviour) / 3)
    result = "PASS" if overall >= pass_threshold else "FAIL"

    # --------------------------------------------------
    # 7️⃣ SAVE INTO DB & LOCK ATTEMPT
    # --------------------------------------------------
    db = SessionLocal()
    try:
        attempt = (
            db.query(InterviewAttempts)
            .filter(InterviewAttempts.attempt_id == attempt_id)
            .first()
        )

        if not attempt:
            raise HTTPException(404, "Interview attempt not found")

        if attempt.status == "LOCKED":
            raise HTTPException(403, "Already evaluated")

        attempt.ai_score = technical
        attempt.manual_score = communication
        attempt.skill_score = behaviour
        attempt.interview_score = overall
        if mcq_data is not None:
            attempt.mcq = mcq_data
        if coding_data is not None:
            attempt.coding = coding_data
        attempt.per_question = per_question
        attempt.feedback = scores.get("feedback")
        attempt.anomaly_summary = anomaly_counts

        attempt.status = "LOCKED"
        attempt.completed_at = datetime.utcnow()
        attempt.updated_at = datetime.utcnow()
        attempt.technical_reason = scores["technical"].get("reason")
        attempt.communication_reason = scores["communication"].get("reason")
        attempt.behaviour_reason = scores["behaviour"].get("reason")

        db.commit()

    finally:
        db.close()

    # --------------------------------------------------
    # 8️⃣ FINAL RESPONSE (CERTIFICATE-READY)
    # --------------------------------------------------
    return {
        "ok": True,
        "attemptId": attempt_id,
        "candidateName": candidate_name,
        "candidateId": candidate_id,
        "designation": designation,

        "overall": overall,
        "result": result,
        "feedback": scores.get("feedback"),

        "scores": [
        {
            "title": "technical",
            "score": technical,
            "description": scores["technical"].get("reason"),
        },
        {
            "title": "communication",
            "score": communication,
            "description": scores["communication"].get("reason"),
        },
        {
            "title": "behaviour",
            "score": behaviour,
            "description": scores["behaviour"].get("reason"),
        },
    ],

        "per_question": per_question,
        "anomalyCounts": anomaly_counts,
        "anomaly_counts": anomaly_counts,
        "mcq": mcq_data,
        "coding": coding_data,
    }

# ============================================================
# 📜 TRANSCRIPT APIs
# ============================================================

from fastapi import HTTPException
from fastapi.responses import FileResponse


# ----------------------------
# 1) GET full transcript file
# ----------------------------
@router.get("/transcript/get/{candidate_name}/{candidate_id}")
async def get_transcript(candidate_name: str, candidate_id: str, token: str, jd_id: int = None):
    db = SessionLocal()
    try:
        attempt = (
            db.query(InterviewAttempts)
            .filter(
                InterviewAttempts.candidate_id == candidate_id,
                InterviewAttempts.jd_id == jd_id,
                InterviewAttempts.interview_token == token
            )
            .first()
        )

        if not attempt or attempt.status not in ("COMPLETED", "LOCKED"):
            raise HTTPException(403, "Unauthorized access")
    finally:
        db.close()


# ------------------------------------------
# 2) GET transcript summary (clean messages)
# ------------------------------------------
@router.get("/transcript/summary/{candidate_name}/{candidate_id}")
async def get_transcript_summary(candidate_name: str, candidate_id: str):
    path = transcript_file(candidate_name, candidate_id)
    print("📄 GET TRANSCRIPT SUMMARY:", path)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Transcript not found")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversation = data.get("conversation", [])

    summary = [
        {
            "sender": msg["sender"],
            "text": msg["text"],
            "timestamp": msg.get("timestamp")
        }
        for msg in conversation
    ]

    return {
        "ok": True,
        "candidateName": candidate_name,
        "candidateId": candidate_id,
        "messageCount": len(summary),
        "summary": summary
    }


# ----------------------------
# 3) DELETE transcript
# ----------------------------
@router.delete("/transcript/delete/{candidate_name}/{candidate_id}")
async def delete_transcript(candidate_name: str, candidate_id: str):
    path = transcript_file(candidate_name, candidate_id)
    print("🗑 DELETE TRANSCRIPT:", path)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Transcript not found")

    os.remove(path)

    return {
        "ok": True,
        "message": "Transcript deleted successfully",
        "path": path
    }


# ----------------------------
# 4) LIST all transcripts
# ----------------------------
@router.get("/transcript/list")
async def list_transcripts():
    print("📂 LISTING ALL TRANSCRIPTS")

    files = [
        f for f in os.listdir(TRANSCRIPTS_DIR)
        if f.endswith(".json")
    ]

    return {
        "ok": True,
        "count": len(files),
        "files": files
    }


FRONTEND_BASE = os.getenv("FRONTEND_BASE", "https://primehire-beta-ui.vercel.app")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
@router.post("/scheduler/schedule")
async def schedule_interview(payload: dict):
    session = SessionLocal()
    try:
        print("\n\n================ SCHEDULER START ================")
        print("📩 Incoming Payload:", payload)

        candidate_id = payload.get("candidate_id")
        candidate_name = payload.get("candidate_name") or candidate_id
        candidate_email = payload.get("candidate_email")

        try:
            jd_id = int(payload.get("jd_id"))
        except Exception:
            jd_id = None

        start_iso = payload.get("start_iso")
        end_iso = payload.get("end_iso")

        if not candidate_id or not start_iso or not end_iso:
            raise HTTPException(status_code=400, detail="Missing required fields")

        slot_start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        slot_end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))

        # 🔎 FIND EXISTING SCHEDULED ATTEMPT
        existing = (
            session.query(InterviewAttempts)
            .filter(
                InterviewAttempts.candidate_id == candidate_id,
                InterviewAttempts.jd_id == jd_id,
                InterviewAttempts.status.in_(["SCHEDULED"]),
            )
            .first()
        )

        # 🔁 CANCEL IF EXISTS (AUTO RESCHEDULE)
        if existing:
            print("♻ Existing interview found — auto rescheduling:", existing.attempt_id)
            existing.status = "CANCELLED"
            existing.updated_at = datetime.utcnow()
            session.commit()

        # 🔐 CREATE NEW ATTEMPT
        token = str(uuid.uuid4())

        attempt = InterviewAttempts(
            candidate_id=candidate_id,
            jd_id=jd_id,
            slot_start=slot_start,
            slot_end=slot_end,
            interview_token=token,
            status="SCHEDULED",
        )

        session.add(attempt)
        session.commit()
        session.refresh(attempt)

        print("✅ Created attempt:", attempt.attempt_id)

        # ---------------- EMAIL RESOLUTION ----------------
        if not candidate_email and "@" in candidate_id:
            candidate_email = candidate_id

        ist = timezone(timedelta(hours=5, minutes=30))
        slot_start_ist = slot_start.astimezone(ist)
        slot_end_ist = slot_end.astimezone(ist)

        interview_link = (
            f"{FRONTEND_BASE}/validation_panel?"
            f"candidateId={candidate_id}&jd_id={jd_id}&token={token}"
        )

        message_text = (
            f"Hi {candidate_name},\n\n"
            f"Your interview is scheduled from {slot_start_ist} to {slot_end_ist} (IST).\n\n"
            f"Start interview here:\n{interview_link}\n\n"
            "⚠ DO NOT share this link.\n\n"
            "Thanks,\nPrimeHire Team"
        )

        if candidate_email:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{API_BASE_URL}/mcp/tools/match/send_mail",
                    json={
                        "email": candidate_email,
                        "candidate_name": candidate_name,
                        "message": message_text,
                    },
                )

        return {
            "ok": True,
            "attempt_id": attempt.attempt_id,
            "interview_token": token,
            "start_iso": start_iso,
            "end_iso": end_iso,
        }

    except HTTPException:
        session.rollback()
        raise

    except Exception as e:
        session.rollback()
        print("❌ Scheduler exception:", e)
        raise HTTPException(status_code=500, detail="Failed to schedule interview")

    finally:
        session.close()

# @router.post("/scheduler/schedule")
# async def schedule_interview(payload: dict):
#     session = SessionLocal()
#     try:
#         print("\n\n================ SCHEDULER START ================")
#         print("📩 Incoming Payload:", payload)

#         # ---------------- BASIC FIELDS ----------------
#         candidate_id = payload.get("candidate_id")
#         candidate_name = payload.get("candidate_name") or candidate_id
#         candidate_email = payload.get("candidate_email")

#         # Normalize JD ID
#         try:
#             jd_id = int(payload.get("jd_id"))
#         except Exception:
#             jd_id = None

#         start_iso = payload.get("start_iso")
#         end_iso = payload.get("end_iso")

#         if not candidate_id or not start_iso or not end_iso:
#             raise HTTPException(status_code=400, detail="Missing required fields")

#         # ---------------- TIME PARSING ----------------
#         slot_start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
#         slot_end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))

#         # ---------------- 🔎 PRE-CHECK (NEW) ----------------
#         existing = (
#             session.query(InterviewAttempts)
#             .filter(
#                 InterviewAttempts.candidate_id == candidate_id,
#                 InterviewAttempts.jd_id == jd_id,
#                 InterviewAttempts.status.in_(["SCHEDULED"]),
#             )
#             .first()
#         )

#         if existing:
#             raise HTTPException(
#                 status_code=409,
#                 detail={
#                     "message": "Interview already scheduled",
#                     "attempt_id": existing.attempt_id,
#                     "slot_start": existing.slot_start.isoformat(),
#                     "slot_end": existing.slot_end.isoformat(),
#                     "status": existing.status,
#                 },
#             )

#         # ---------------- CREATE TOKEN ----------------
#         token = str(uuid.uuid4())

#         # ---------------- CREATE ATTEMPT ----------------
#         attempt = InterviewAttempts(
#             candidate_id=candidate_id,
#             jd_id=jd_id,
#             slot_start=slot_start,
#             slot_end=slot_end,
#             interview_token=token,
#             status="SCHEDULED",
#         )

#         session.add(attempt)
#         session.commit()
#         session.refresh(attempt)

#         print("✅ Created attempt:", attempt.attempt_id)

#         # ---------------- EMAIL RESOLUTION ----------------
#         if candidate_email and "@" in candidate_email:
#             pass
#         elif "@" in candidate_id:
#             candidate_email = candidate_id
#         else:
#             row = session.execute(
#                 text("SELECT full_name, email FROM candidates WHERE candidate_id=:cid"),
#                 {"cid": candidate_id},
#             ).fetchone()

#             if row:
#                 candidate_name = row[0] or candidate_name
#                 candidate_email = row[1]

#         # ---------------- TIMEZONE CONVERSION ----------------
#         ist = timezone(timedelta(hours=5, minutes=30))
#         slot_start_ist = slot_start.astimezone(ist)
#         slot_end_ist = slot_end.astimezone(ist)

#         # ---------------- INTERVIEW LINK ----------------
#         # interview_link = (
#         #     f"https://agentic.primehire.ai/validation_panel?"
#         #     f"candidateId={candidate_id}&jd_id={jd_id}&token={token}"
#         # )
#         interview_link = (
#     f"{FRONTEND_BASE}/validation_panel?"
#     f"candidateId={candidate_id}&jd_id={jd_id}&token={token}"
# )

#         message_text = (
#             f"Hi {candidate_name},\n\n"
#             f"Your interview is scheduled from {slot_start_ist} to {slot_end_ist} (IST).\n\n"
#             f"Start interview here:\n{interview_link}\n\n"
#             "⚠ DO NOT share this link.\n\n"
#             "Thanks,\nPrimeHire Team"
#         )

#         # ---------------- SEND EMAIL ----------------
#         if candidate_email:
#             async with httpx.AsyncClient(timeout=10.0) as client:
#                 await client.post(
#                     f"{API_BASE_URL}/mcp/tools/match/send_mail",
#                     json={
#                         "email": candidate_email,
#                         "candidate_name": candidate_name,
#                         "message": message_text,
#                     },
#                 )

#         # ---------------- RESPONSE ----------------
#         return {
#             "ok": True,
#             "attempt_id": attempt.attempt_id,
#             "interview_token": token,
#             "start_iso": start_iso,
#             "end_iso": end_iso,
#         }

#     except HTTPException:
#         # 🔥 Important: rethrow clean API errors (409, 400)
#         session.rollback()
#         raise

#     except Exception as e:
#         session.rollback()
#         print("❌ Scheduler exception:", e)
#         raise HTTPException(status_code=500, detail="Failed to schedule interview")

#     finally:
#         session.close()

@router.post("/scheduler/reschedule")
async def reschedule_interview(payload: dict):
    session = SessionLocal()
    try:
        print("\n\n================ RESCHEDULE START ================")
        print("📩 Incoming Payload:", payload)

        candidate_id = payload.get("candidate_id")
        candidate_name = payload.get("candidate_name") or candidate_id
        candidate_email = payload.get("candidate_email")

        try:
            jd_id = int(payload.get("jd_id"))
        except Exception:
            jd_id = None

        start_iso = payload.get("start_iso")
        end_iso = payload.get("end_iso")

        if not candidate_id or not start_iso or not end_iso:
            raise HTTPException(400, "Missing required fields")

        slot_start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        slot_end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))

        # 🔎 FIND EXISTING ACTIVE ATTEMPT
        existing = (
            session.query(InterviewAttempts)
            .filter(
                InterviewAttempts.candidate_id == candidate_id,
                InterviewAttempts.jd_id == jd_id,
                InterviewAttempts.status.in_(["SCHEDULED", "IN_PROGRESS"]),
            )
            .first()
        )

        if not existing:
            raise HTTPException(404, "No active interview to reschedule")

        # 🔐 CANCEL OLD ATTEMPT
        existing.status = "CANCELLED"
        existing.updated_at = datetime.utcnow()
        session.commit()

        # 🔐 CREATE NEW ATTEMPT
        token = str(uuid.uuid4())

        new_attempt = InterviewAttempts(
            candidate_id=candidate_id,
            jd_id=jd_id,
            slot_start=slot_start,
            slot_end=slot_end,
            interview_token=token,
            status="SCHEDULED",
        )

        session.add(new_attempt)
        session.commit()
        session.refresh(new_attempt)

        print("✅ Rescheduled attempt:", new_attempt.attempt_id)

        # ---------------- EMAIL RESOLUTION ----------------
        if not candidate_email and "@" in candidate_id:
            candidate_email = candidate_id

        ist = timezone(timedelta(hours=5, minutes=30))
        slot_start_ist = slot_start.astimezone(ist)
        slot_end_ist = slot_end.astimezone(ist)

        # interview_link = (
        #     f"https://agentic.primehire.ai/validation_panel?"
        #     f"candidateId={candidate_id}&jd_id={jd_id}&token={token}"
        # )
        interview_link = (
    f"{FRONTEND_BASE}/validation_panel?"
    f"candidateId={candidate_id}&jd_id={jd_id}&token={token}"
)


        message_text = (
            f"Hi {candidate_name},\n\n"
            f"Your interview has been rescheduled.\n\n"
            f"New time:\n"
            f"{slot_start_ist} → {slot_end_ist} (IST)\n\n"
            f"Start interview:\n{interview_link}\n\n"
            "⚠ DO NOT share this link.\n\n"
            "Thanks,\nPrimeHire Team"
        )

        if candidate_email:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{API_BASE_URL}/mcp/tools/match/send_mail",
                    json={
                        "email": candidate_email,
                        "candidate_name": candidate_name,
                        "message": message_text,
                    },
                )

        return {
            "ok": True,
            "attempt_id": new_attempt.attempt_id,
            "interview_token": token,
            "start_iso": start_iso,
            "end_iso": end_iso,
        }

    except HTTPException:
        session.rollback()
        raise

    except Exception as e:
        session.rollback()
        print("❌ Reschedule error:", e)
        raise HTTPException(500, "Failed to reschedule interview")

    finally:
        session.close()
        
#scheduler_guard.py


@router.get("/scheduler/validate_access")
async def validate_interview_access(
    candidate_id: str,
    jd_id: int,
    token: str
):
    session = SessionLocal()
    try:
        attempt = (
            session.query(InterviewAttempts)
            .filter(
                InterviewAttempts.candidate_id == candidate_id,
                InterviewAttempts.jd_id == jd_id,
                InterviewAttempts.interview_token == token,
            )
            .order_by(InterviewAttempts.created_at.desc())
            .first()
        )

        if not attempt:
            print("❌ Invalid interview link")
            print("candidate_id:", candidate_id)
            print("jd_id:", jd_id)
            print("token:", token)
            raise HTTPException(403, "Invalid interview link")


        # ✅ ALWAYS USE UTC AWARE TIME (same as stage 3)
        now = datetime.now(timezone.utc)

        slot_start = attempt.slot_start
        slot_end = attempt.slot_end

        # 🛡️ Safety check (DB should already store UTC)
        if slot_start.tzinfo is None:
            slot_start = slot_start.replace(tzinfo=timezone.utc)
        if slot_end.tzinfo is None:
            slot_end = slot_end.replace(tzinfo=timezone.utc)

        # ⏰ Optional grace window (keep small)
        GRACE_MINUTES = 0
        start_allowed = slot_start - timedelta(minutes=GRACE_MINUTES)
        end_allowed = slot_end + timedelta(minutes=GRACE_MINUTES)

        # ❌ Too early
        if now < start_allowed:
            return {
                "ok": False,
                "reason": "TOO_EARLY",
                "slot_start": slot_start.isoformat(),
                "slot_end": slot_end.isoformat(),
                "status": attempt.status,
            }

        # ❌ Expired
        if now > end_allowed:
            if attempt.status not in ["COMPLETED", "LOCKED"]:
                attempt.status = "EXPIRED"
                attempt.updated_at = now
                session.commit()

            return {
                "ok": False,
                "reason": "EXPIRED",
                "slot_start": slot_start.isoformat(),
                "slot_end": slot_end.isoformat(),
                "status": "EXPIRED",
            }

        # 🔒 Status protection
        if attempt.status in ["LOCKED", "COMPLETED"]:
            return {
                "ok": False,
                "reason": "INVALID_STATUS",
                "status": attempt.status,
            }

        

        return {
            "ok": True,
            "attempt_id": attempt.attempt_id,
            "slot_start": slot_start.isoformat(),
            "slot_end": slot_end.isoformat(),
            "status": attempt.status,
        }

    finally:
        session.close()


@router.get("/scheduler/existing")
async def get_existing_schedule(candidate_id: str, jd_id: int | None = None):
    session = SessionLocal()
    try:
        q = session.query(InterviewAttempts).filter(
            InterviewAttempts.candidate_id == candidate_id,
            InterviewAttempts.status.in_(["SCHEDULED", "IN_PROGRESS"]),
        )

        if jd_id is not None:
            q = q.filter(InterviewAttempts.jd_id == jd_id)

        attempt = q.first()

        if not attempt:
            return {"exists": False}

        return {
            "exists": True,
            "attempt_id": attempt.attempt_id,
            "slot_start": attempt.slot_start.isoformat(),
            "slot_end": attempt.slot_end.isoformat(),
            "status": attempt.status,
        }

    finally:
        session.close()

# ============================================================
# 📘 MCQ GENERATOR — Generate 4 MCQ QUESTIONS using GPT
# ============================================================

from openai import AsyncOpenAI
mcq_client = AsyncOpenAI()

MCQ_DIR = "logs/mcq"
os.makedirs(MCQ_DIR, exist_ok=True)


def mcq_file(candidate_id: str):
    return os.path.join(MCQ_DIR, f"{candidate_id}.json")


# FILE: app/mcp/tools/interview_bot_beta.py

@router.post("/generate-mcq")
async def generate_mcq(
    job_description: str = Form(...),
    attempt_id: int = Form(...),        # ✅ NEW
    jd_id: int | None = Form(None)
):
    """
    Generates 4 MCQ questions from JD.
    Saved per ATTEMPT (not candidate).
    """

    print("\n========== MCQ GENERATION ==========")
    print("Attempt ID:", attempt_id)
    print("JD ID:", jd_id)

    prompt = f"""
You are a SENIOR TECHNICAL INTERVIEWER hiring for this role.

Your task is to generate EXACTLY 4 HARD, NON-GENERIC, TECHNICAL MCQ questions
directly derived from the Job Description below.

STRICT RULES (DO NOT VIOLATE):
1. Questions MUST test real-world implementation knowledge, not definitions.
2. NO generic theory, buzzwords, or textbook questions.
3. At least:
   - 1 question on system design / architecture decisions
   - 1 question on debugging, failure modes, or edge cases
   - 1 question on performance, scalability, or optimization
   - 1 question on code-level behavior or API misuse
4. Options must be plausible — only ONE correct answer.
5. Incorrect options must represent COMMON MISTAKES made by engineers.
6. Use concrete scenarios, constraints, or code behavior.
7. Assume the candidate has 5+ years of experience.

JOB DESCRIPTION:
----------------
{job_description}
----------------

OUTPUT FORMAT (STRICT JSON ONLY, NO MARKDOWN, NO EXPLANATION):

{{
  "mcq": [
    {{
      "question": "A scenario-based, deeply technical question",
      "options": [
        "Option A",
        "Option B",
        "Option C",
        "Option D"
      ],
      "correct": "A"
    }}
  ]
}}
"""


    try:
        res = await mcq_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=900
        )

        raw = res.choices[0].message.content
        print("GPT RAW MCQ:", raw)

        try:
            mcq_json = json.loads(raw)
        except:
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            mcq_json = json.loads(cleaned)

        mcq_list = mcq_json.get("mcq", [])

        # -----------------------------
        # SAVE PER ATTEMPT
        # -----------------------------
        path = os.path.join("logs/mcq", f"attempt_{attempt_id}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_blob = {
            "attempt_id": attempt_id,
            "jd_id": jd_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "mcq": mcq_list
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_blob, f, indent=2)

        return {
            "ok": True,
            "attempt_id": attempt_id,
            "jd_id": jd_id,
            "mcq": mcq_list
        }

    except Exception as e:
        print("❌ MCQ generator error:", e)
        return {"ok": False, "error": str(e)}
