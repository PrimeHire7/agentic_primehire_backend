
# # main.py - FastAPI HTTP & WebSocket wrapper for MCP tool dispatcher.

# # Endpoints:
# # - POST /mcp/run_tool        -> {"tool": "tool.name", "params": {...}}  returns tool result
# # - POST /chat                -> {"message": "..."} conversational endpoint
# # - POST /upload_resumes       -> multipart upload (files) -> proxies to resume upload tool
# # - GET  /zoho/candidates     -> proxies to Zoho fetch (for quick testing)
# # - WebSocket /ws             -> conversational WebSocket for frontend


# import os
# import subprocess
# import json
# import logging
# from pathlib import Path
# import tempfile
# import requests

# import uvicorn
# from fastapi import FastAPI, Request, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse
# from fastapi.middleware.trustedhost import TrustedHostMiddleware

# # MCP imports - ensures all tools are loaded
# from app.mcp.server_core import run_tool
# from app.mcp import chatbot_agent, tools
# from app.mcp.tools import (
#     jd_tool,
#     zoho_tool,
#     matcher_tool,
#     mailmind_tool,
#     interview_bot,
#     candidate_validation,
#     resume_tool,
#     primehire_brain,
#     linkedin_poster,
#     matcher_history,
#     jd_history,
#     interview_bot_beta,
#     candidate_api,
# )

# # -------------------------
# # Logging setup
# # -------------------------
# logger = logging.getLogger("primehire")
# logging.basicConfig(level=logging.INFO)

# # -------------------------
# # Token DB setup
# # -------------------------
# TOKEN_DB = Path(__file__).resolve().parent / "tokens_db.json"
# TOKEN_DB.touch(exist_ok=True)
# logger.info(f"Using token DB path: {TOKEN_DB.resolve()}")



# # ============================================================
# # 🔥 DATABASE: LOAD ALL MODELS BEFORE create_all()
# # ============================================================
# from app.db import Base, engine

# # Import models so SQLAlchemy registers them under Base.metadata
# from app.mcp.tools.resume_tool import Candidate
# from app.mcp.tools.jd_history import GeneratedJDHistory
# from app.mcp.tools.interview_attempts_model import InterviewAttempts
# from app.mcp.tools import jd_cache
# # Create all tables (ONLY ONCE)
# Base.metadata.create_all(bind=engine)

# # -------------------------
# # Zoho credentials
# # -------------------------
# ZOHO_CLIENT_ID = "1000.7EDK5QI3TSUU214UOL80N0VMWKMKYO"
# ZOHO_CLIENT_SECRET = "c73daf5909aca154c655fcd80d2b363b483549846e"
# REDIRECT_URI = "https://primehire.nirmataneurotech.com/callback"
# ACCOUNTS_BASE = "https://accounts.zoho.com"
# DEFAULT_EMAIL = "director@nirmataneurotech.com"
# ORG_ID = "901310447"
# API_BASE = "https://recruit.zoho.com/recruit/v2"

# app = FastAPI(title="PrimeHire Backend - MCP adapter")
# # -------------------------
# # CORS configuration
# # -------------------------
# origins = [
#     "http://localhost:3000",
#     "http://127.0.0.1:3000",
#     "http://127.0.0.1:8080",
#     "http://localhost:5173",
#     "http://127.0.0.1:5173",
#     "http://localhost:8080",
#     "https://primehire.nirmataneurotech.com", 
#     "https://prime-hire-demo.vercel.app",
#     "https://primehire-beta-ui.vercel.app",
#     "https://www.primehire-beta-ui.vercel.app"
    
#      # ✅ only HTTPS for production
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------
# # Register MCP routers
# # -------------------------
# app.include_router(candidate_validation.router, prefix="/mcp/tools/candidate_validation")
# app.include_router(resume_tool.router, prefix="/mcp/tools/resume")
# app.include_router(mailmind_tool.router, prefix="/mcp/tools")
# app.include_router(jd_tool.router, prefix="/mcp/tools/jd")
# app.include_router(matcher_tool.router, prefix="/mcp/tools/match")
# app.include_router(interview_bot.router, prefix="/mcp/interview")
# app.include_router(primehire_brain.router, prefix="/mcp/tools/resume")
# app.include_router(linkedin_poster.router, prefix="/mcp/tools/linkedin")
# app.include_router(matcher_history.router, prefix="/mcp/tools/match_history")
# app.include_router(jd_history.router, prefix="/mcp/tools/jd_history")
# app.include_router(interview_bot_beta.router, prefix="/mcp/interview_bot_beta")
# app.include_router(candidate_api.router, prefix="/mcp/tools")
# app.include_router(jd_cache.router, prefix="/mcp/tools/jd_cache")
# # -------------------------
# # Health check
# # -------------------------
# @app.get("/health")
# async def health():
#     return {"status": "ok"}

# # -------------------------
# # Generic MCP tool runner
# # -------------------------
# @app.post("/mcp/run_tool")
# async def http_run_tool(body: Request):
#     payload = await body.json()
#     tool_name = payload.get("tool")
#     params = payload.get("params", {})

#     if not tool_name:
#         raise HTTPException(status_code=400, detail="Missing 'tool' in body")

#     try:
#         result = await run_tool(tool_name, params)
#         return {"ok": True, "result": result}
#     except Exception as e:
#         logger.exception("Tool run failed")
#         raise HTTPException(status_code=500, detail=str(e))

# # -------------------------
# # Conversational chat endpoint
# # -------------------------
# @app.post("/chat")
# async def chat_endpoint(payload: dict):
#     message = payload.get("message")
#     if not message:
#         raise HTTPException(status_code=400, detail="Missing 'message'")

#     res = await run_tool("chatbot.handle_message", {"message": message})
#     return {"response": res}

# # -------------------------
# # Quick proxy for Zoho candidates
# # -------------------------
# @app.get("/zoho/candidates")
# async def zoho_candidates(email: str = None, page: int = 1, per_page: int = 50):
#     if not email:
#         email = DEFAULT_EMAIL
#     result = await run_tool(
#         "zoho.fetch_candidates", {"email": email, "page": page, "per_page": per_page}
#     )
#     return result

# # -------------------------
# # Token helpers
# # -------------------------
# def save_tokens(email: str, tokens: dict):
#     all_tokens = json.loads(TOKEN_DB.read_text() or "{}")
#     all_tokens[email] = tokens
#     TOKEN_DB.write_text(json.dumps(all_tokens, indent=2))
#     logger.info(f"Tokens saved for {email}")

# def load_tokens(email: str):
#     try:
#         all_tokens = json.loads(TOKEN_DB.read_text() or "{}")
#         return all_tokens.get(email)
#     except Exception as e:
#         logger.error(f"Error loading tokens: {e}")
#         return None

# # -------------------------
# # Zoho OAuth callback
# # -------------------------
# @app.get("/callback", response_class=HTMLResponse)
# def callback(request: Request):
#     code = request.query_params.get("code")
#     if not code:
#         raise HTTPException(status_code=400, detail="Missing authorization code")

#     data = {
#         "code": code,
#         "client_id": ZOHO_CLIENT_ID,
#         "client_secret": ZOHO_CLIENT_SECRET,
#         "redirect_uri": REDIRECT_URI,
#         "grant_type": "authorization_code",
#     }
#     resp = requests.post(f"{ACCOUNTS_BASE}/oauth/v2/token", data=data)
#     if resp.status_code != 200:
#         raise HTTPException(status_code=500, detail=f"Token exchange failed: {resp.text}")

#     tokens = resp.json()
#     if "access_token" not in tokens:
#         raise HTTPException(status_code=500, detail=f"Invalid token response: {tokens}")

#     save_tokens(DEFAULT_EMAIL, tokens)

#     html_content = f"""
#     <html>
#         <head>
#             <title>Zoho Connected</title>
#             <script>
#                 window.opener.postMessage('ZOHO_AUTH_SUCCESS', window.location.origin);
#                 setTimeout(() => window.close(), 3000);
#             </script>
#         </head>
#         <body>
#             <h2>✅ Zoho connected successfully!</h2>
#             <p>You can close this window if it doesn't close automatically.</p>
#         </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content, status_code=200)

# # -------------------------
# # Candidate fetch + preprocess
# # -------------------------
# def preprocess_candidate(c):
#     return {
#         "id": c.get("id"),
#         "Full_Name": c.get("Full_Name") or "No Name",
#         "Email": c.get("Email"),
#         "Current_Employer": c.get("Current_Employer"),
#         "Current_Job_Title": c.get("Current_Job_Title"),
#         "Experience_in_Years": c.get("Experience_in_Years"),
#         "Candidate_Status": c.get("Candidate_Status"),
#     }

# @app.get("/fetch_candidates")
# def fetch_candidates(email: str, page: int = 1, per_page: int = 50):
#     tokens = load_tokens(email)
#     if not tokens or "access_token" not in tokens:
#         raise HTTPException(status_code=400, detail="No access token found. Authorize first.")

#     headers = {"Authorization": f"Zoho-oauthtoken {tokens['access_token']}", "orgId": ORG_ID}
#     resp = requests.get(f"{API_BASE}/Candidates?page={page}&per_page={per_page}", headers=headers)

#     if resp.status_code == 401:
#         logger.info("Access token expired, refreshing...")
#         headers["Authorization"] = f"Zoho-oauthtoken {refresh_access_token(email)}"
#         resp = requests.get(f"{API_BASE}/Candidates?page={page}&per_page={per_page}", headers=headers)

#     if resp.status_code != 200:
#         raise HTTPException(status_code=500, detail=resp.text)

#     data = resp.json().get("data", [])
#     preprocessed = [preprocess_candidate(c) for c in data]
#     return {"count": len(preprocessed), "candidates": preprocessed}

# # -------------------------
# # WebSocket endpoint (Intent-Only)
# # -------------------------
# async def decide_feature_from_message(message: str) -> str:
#     msg = message.strip().lower()

#     # ============================================================
#     # 1️⃣ STRICT PREFIX RULES (these override everything else)
#     # ============================================================

#     if msg.startswith("start jd creator"):
#         return "JD Creator"

#     if msg.startswith("start profile matcher"):
#         return "Profile Matcher"

#     if msg.startswith("start upload resumes"):
#         return "Upload Resumes"

#     if msg.startswith("start interviewbot") or msg.startswith("start interview bot"):
#         return "InterviewBot"

#     # ============================================================
#     # 2️⃣ EXACT TASK NAMES (high priority)
#     # ============================================================

#     if msg in ["jd creator", "create jd", "generate jd"]:
#         return "JD Creator"

#     if msg in ["profile matcher", "match profiles"]:
#         return "Profile Matcher"

#     # ============================================================
#     # 3️⃣ MATCH HISTORY — must NOT trigger matcher
#     # ============================================================
#     if "match history" in msg or "profile match history" in msg:
#         return "ProfileMatchHistory"

#     # ============================================================
#     # 4️⃣ KEYWORD OVERRIDES (CONTROLLED)
#     # ============================================================

#     # JD always wins over matcher
#     if "job description" in msg or "jd " in msg:
#         return "JD Creator"

#     # ❗️ Matcher must NOT run before JD creator
#     if "candidate" in msg or "match" in msg:
#         return "Profile Matcher"

#     # ============================================================
#     # 5️⃣ Upload resumes
#     # ============================================================
#     if "resume" in msg or "upload" in msg or "pdf" in msg:
#         return "Upload Resumes"

#     # ============================================================
#     # 6️⃣ Interview
#     # ============================================================
#     if "interview" in msg or "question" in msg:
#         return "InterviewBot"

#     # ============================================================
#     # 7️⃣ Default → Brain
#     # ============================================================
#     return "PrimeHireBrain"



# @app.websocket("/ws")
# async def websocket_endpoint(ws: WebSocket):
#     await ws.accept()
#     logger.info("🌐 WebSocket connected for intent routing")

#     try:
#         while True:
#             raw_data = await ws.receive_text()
#             try:
#                 obj = json.loads(raw_data)
#                 message = obj.get("message") or obj
#             except Exception:
#                 message = raw_data

#             logger.info(f"💬 Incoming message: {message}")

#             # Step 1: detect intent
#             feature = await decide_feature_from_message(message)

#             # Step 2: send intent
#             await ws.send_json({
#                 "type": "feature_detected",
#                 "data": feature,
#                 "user_message": message
#             })

           

#     except WebSocketDisconnect:
#         logger.info("🔌 WebSocket disconnected")
#     except Exception as e:
#         logger.error(f"🔥 WebSocket error: {e}")
#         await ws.close()
# # -------------------------
# # Run server
# # -------------------------
# if __name__ == "__main__":
#     port = int(os.getenv("PORT", "8000"))
#     uvicorn.run(
#     "main:app",
#     host="0.0.0.0",
#     port=port,
#     log_level="info"
# )

# main.py — PrimeHire Backend (Production Safe)
# Supports: ALB + WebSockets + Multiple Users + Gunicorn Workers

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
# Logging
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
# Routers
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

# -------------------------
# Zoho credentials
# -------------------------
# ZOHO_CLIENT_ID = "1000.7EDK5QI3TSUU214UOL80N0VMWKMKYO"
# ZOHO_CLIENT_SECRET = "c73daf5909aca154c655fcd80d2b363b483549846e"
# REDIRECT_URI = "https://primehire.nirmataneurotech.com/callback"
# ACCOUNTS_BASE = "https://accounts.zoho.com"
# DEFAULT_EMAIL = "director@nirmataneurotech.com"
# ORG_ID = "901310447"
# API_BASE = "https://recruit.zoho.com/recruit/v2"

# =========================================================
# Health
# =========================================================
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

# =========================================================
# Zoho OAuth
# =========================================================
# ZOHO_CLIENT_ID = "ZOHO_CLIENT_ID"
# ZOHO_CLIENT_SECRET = "ZOHO_CLIENT_SECRET"
# REDIRECT_URI = "https://primehire.nirmataneurotech.com/callback"
# ACCOUNTS_BASE = "https://accounts.zoho.com"
# DEFAULT_EMAIL = "director@nirmataneurotech.com"

# def save_tokens(email: str, tokens: dict):
#     data = json.loads(TOKEN_DB.read_text() or "{}")
#     data[email] = tokens
#     TOKEN_DB.write_text(json.dumps(data, indent=2))

# @app.get("/callback", response_class=HTMLResponse)
# def callback(request: Request):
#     code = request.query_params.get("code")
#     if not code:
#         raise HTTPException(status_code=400, detail="Missing authorization code")

#     resp = requests.post(
#         f"{ACCOUNTS_BASE}/oauth/v2/token",
#         data={
#             "code": code,
#             "client_id": ZOHO_CLIENT_ID,
#             "client_secret": ZOHO_CLIENT_SECRET,
#             "redirect_uri": REDIRECT_URI,
#             "grant_type": "authorization_code",
#         },
#     )

#     save_tokens(DEFAULT_EMAIL, resp.json())
#     return HTMLResponse("<h2>Zoho connected</h2>")

# =========================================================
# Intent Detection (cheap + async-safe)
# =========================================================
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
