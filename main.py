import os
import json
import uuid
import logging
from pathlib import Path
import requests

from fastapi import (
    FastAPI,
    Request,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool

# =========================================================
# Logging Naga
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("primehire")

# =========================================================
# App
# =========================================================
app = FastAPI(title="PrimeHire Backend - MCP Adapter")

# =========================================================
# Middleware (ALB-safe)
# =========================================================
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],  # Required for ALB + WS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Token DB
# =========================================================
TOKEN_DB = Path(__file__).resolve().parent / "tokens_db.json"
TOKEN_DB.touch(exist_ok=True)

# =========================================================
# Database (register models ONCE per worker)
# =========================================================
# =========================================================
# Database Init
# =========================================================
from app.db import Base, engine

# Import all models so SQLAlchemy registers them
from app.mcp.tools.interview_attempts_model import InterviewAttempts
from app.mcp.tools.jd_history import GeneratedJDHistory
from app.mcp.tools.resume_tool import Candidate
from app.mcp.tools.jd_cache import TempJDCache  # ← this is where jd_cache table is registered
from app.mcp.tools import company_auth



print("🟡 Initializing database...")
Base.metadata.create_all(bind=engine)
print("🟢 Database ready.")

# =========================================================
# MCP / Tools (loaded once per worker)
# =========================================================
from app.mcp.server_core import run_tool
from app.mcp.tools import (
    jd_tool,
    zoho_tool,
    matcher_tool,
    mailmind_tool,
    interview_bot,
    candidate_validation,
    resume_tool,
    primehire_brain,
    linkedin_poster,
    matcher_history,
    jd_history,
    interview_bot_beta,
    candidate_api,
    jd_cache,
)

# =========================================================
# Routers Naga
# =========================================================
app.include_router(candidate_validation.router, prefix="/mcp/tools/candidate_validation")
app.include_router(resume_tool.router, prefix="/mcp/tools/resume")
app.include_router(mailmind_tool.router, prefix="/mcp/tools")
app.include_router(jd_tool.router, prefix="/mcp/tools/jd")
app.include_router(matcher_tool.router, prefix="/mcp/tools/match")
app.include_router(interview_bot.router, prefix="/mcp/interview")
app.include_router(primehire_brain.router, prefix="/mcp/tools/resume")
app.include_router(linkedin_poster.router, prefix="/mcp/tools/linkedin")
app.include_router(matcher_history.router, prefix="/mcp/tools/match_history")
app.include_router(jd_history.router, prefix="/mcp/tools/jd_history")
app.include_router(interview_bot_beta.router, prefix="/mcp/interview_bot_beta")
app.include_router(candidate_api.router, prefix="/mcp/tools")
app.include_router(jd_cache.router, prefix="/mcp/tools/jd_cache")
app.include_router(company_auth.router, prefix="/mcp/tools/company_auth")

 
@app.get("/health")
async def health():
    return {"status": "ok"}

# =========================================================
# MCP Tool Runner (THREADPOOL SAFE)
# =========================================================
@app.post("/mcp/run_tool")
async def http_run_tool(req: Request):
    body = await req.json()
    tool = body.get("tool")
    params = body.get("params", {})

    if not tool:
        raise HTTPException(status_code=400, detail="Missing tool")

    try:
        result = await run_in_threadpool(run_tool, tool, params)
        return {"ok": True, "result": result}
    except Exception as e:
        logger.exception("Tool run failed")
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================
# Chat
# =========================================================
@app.post("/chat")
async def chat(payload: dict):
    msg = payload.get("message")
    if not msg:
        raise HTTPException(status_code=400, detail="Missing message")

    result = await run_in_threadpool(
        run_tool,
        "chatbot.handle_message",
        {"message": msg},
    )
    return {"response": result}
 
async def decide_feature_from_message(message: str) -> str:
    m = message.strip().lower()

    if m.startswith("start jd"):
        return "JD Creator"
    if m.startswith("start profile"):
        return "Profile Matcher"
    if m.startswith("start upload"):
        return "Upload Resumes"
    if "resume" in m:
        return "Upload Resumes"
    if "interview" in m:
        return "InterviewBot"

    return "PrimeHireBrain"

# =========================================================
# WebSocket — MULTI USER SAFE
# =========================================================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    session_id = uuid.uuid4().hex
    logger.info(f"🌐 WS CONNECTED [{session_id}]")

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
                message = data.get("message", raw)
            except Exception:
                message = raw

            feature = await decide_feature_from_message(message)

            await ws.send_json({
                "type": "feature_detected",
                "session_id": session_id,
                "data": feature,
            })

    except WebSocketDisconnect:
        logger.info(f"🔌 WS DISCONNECTED [{session_id}]")
    except Exception as e:
        logger.exception(f"🔥 WS ERROR [{session_id}]")
        await ws.close()
