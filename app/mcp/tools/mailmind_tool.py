
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import imaplib
import email
import os
import traceback
from datetime import datetime, timedelta
from fastapi.responses import FileResponse, JSONResponse
from concurrent.futures import ThreadPoolExecutor
import json
import asyncio
import hashlib

from typing import List, Optional
from pydantic import BaseModel
import uuid
from sqlalchemy.orm import Session
from app.db import engine
from app.mcp.tools.resume_tool import (
    extract_and_store_resume,
    update_progress,
    PROGRESS_FILE,
    quick_extract_email,
    Candidate,
)
import logging
logger = logging.getLogger("mailmind")

# Control concurrency (adjust based on server CPU)
executor = ThreadPoolExecutor(max_workers=6)
router = APIRouter()

RESUMES_DIR = "/home/ubuntu/agentic_primehire_dev/resumes"
os.makedirs(RESUMES_DIR, exist_ok=True)

PLATFORM_IMAP = {
    "gmail": "imap.gmail.com",
    "outlook": "imap.secureserver.net",
    "godaddy": "imap.secureserver.net",
}




class MailCreds(BaseModel):
    email: str
    password: str
    platform: str = "gmail"
    imap_server: str | None = None
    imap_port: int = 993


# ============================================
#  CORS PREFLIGHT FIX  (CRITICAL)
# ============================================
@router.options("/mailmind/fetch-resumes")
async def cors_preflight(request: Request):
    origin = request.headers.get("origin", "*")

    return JSONResponse(
        {"ok": True},
        headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
        },
    )


# ============================================
#  LOGIN / CONNECTION CHECK
# ============================================
@router.post("/mailmind/connect")
def mailmind_connect(creds: MailCreds):
    if not creds.imap_server:
        creds.imap_server = PLATFORM_IMAP.get(creds.platform.lower())
        if not creds.imap_server:
            raise HTTPException(status_code=400, detail="Invalid platform")

    try:
        mail = imaplib.IMAP4_SSL(creds.imap_server, creds.imap_port)
        mail.login(creds.email, creds.password)
        mail.logout()
        return {"message": f"Connected successfully to {creds.platform}!"}
    except imaplib.IMAP4.error:
        raise HTTPException(status_code=400, detail="Invalid email/password or IMAP settings")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================================
#  ATTACHMENT EXTRACTION HELPER
# ================================
def extract_attachments(msg):
    attachments = []

    def walk_message(message):
        for part in message.walk():
            content_type = part.get_content_type()

            # If this is a forwarded message, dive inside it
            if content_type == "message/rfc822":
                payload = part.get_payload()
                if isinstance(payload, list):
                    for sub_msg in payload:
                        walk_message(sub_msg)
                continue

            if part.get_content_maintype() == "multipart":
                continue

            filename = part.get_filename()
            disposition = (part.get("Content-Disposition") or "").lower()

            if not filename and "attachment" not in disposition:
                continue

            try:
                if filename:
                    decoded = email.header.decode_header(filename)[0][0]
                    if isinstance(decoded, bytes):
                        filename = decoded.decode(errors="ignore")
            except:
                pass

            if filename and filename.lower().endswith((".pdf", ".doc", ".docx")):
                data = part.get_payload(decode=True)
                if data:
                    attachments.append((filename, data))

    walk_message(msg)
    return attachments



# ============================================
#  FETCH RESUMES (POST ONLY)
# ============================================
@router.post("/mailmind/fetch-resumes")
async def fetch_resumes(creds: MailCreds, request: Request):

    # Should never happen because OPTIONS has its own handler,
    # but we keep it as a safety guard.
    if request.method == "OPTIONS":
        return JSONResponse({"ok": True})

    try:
        if not creds.imap_server:
            creds.imap_server = PLATFORM_IMAP.get(creds.platform.lower())
            if not creds.imap_server:
                raise HTTPException(status_code=400, detail="Invalid platform")

        imap = imaplib.IMAP4_SSL(creds.imap_server, creds.imap_port)
        imap.login(creds.email, creds.password)

        status, _ = imap.select("INBOX")
        print("INBOX SELECT:", status)

        since_date = (datetime.now() - timedelta(days=7)).strftime("%d-%b-%Y")
        FOLDERS = [
            "INBOX",
            '"[Gmail]/All Mail"',
            '"All Mail"',
            '"Inbox"',
        ]
        for folder in FOLDERS:
            status, _ = imap.select(folder)
            if status == "OK":
                print(f"Using folder: {folder}")
                break
        # status, data = imap.search(None, f'(SINCE "{since_date}")')
        # Gmail / Outlook compatible attachment search
        # status, data = imap.search(
        #     None,
        #     f'(SINCE "{since_date}" OR HEADER Content-Type "multipart/mixed" OR HEADER Content-Type "multipart/related")'
        # )
        status, data = imap.search(None, '(HEADER Content-Disposition "attachment")')
        if status != "OK" or not data[0]:
            status, data = imap.search(None, f'(SINCE "{since_date}")')

        email_ids = data[0].split() if status == "OK" else []

        if not email_ids:
            print("Primary search empty — falling back to ALL")
            status, data = imap.search(None, "ALL")
            email_ids = data[0].split()[-100:]

        print(f"DEBUG: Found {len(email_ids)} emails to scan")




        if status != "OK":
            raise HTTPException(status_code=400, detail="Failed to search emails")

        
        print(f"DEBUG: Found {len(email_ids)} emails since {since_date}")

        saved_files = []

        # ----------------------------------------
        #  Extract attachments
        # ----------------------------------------
        for eid in email_ids:
            status, msg_data = imap.fetch(eid, "(RFC822)")
            if status != "OK":
                continue

            msg = email.message_from_bytes(msg_data[0][1])
            subject = msg.get("Subject")

            print(f"Checking mail: {subject}")

            attachments = extract_attachments(msg)
            for filename, content in attachments:
                print(f"Found attachment: {filename} ({len(content)} bytes)")

            if not attachments:
                print("No attachments in mail:", subject)
                continue

            for filename, content in attachments:
                # safe_name = filename.replace("/", "_").replace("\\", "_")
                base, ext = os.path.splitext(filename)
                # safe_name = f"{base}_{eid.decode()}{ext}".replace("/", "_").replace("\\", "_")
                eid_str = eid.decode() if isinstance(eid, bytes) else str(eid)
                safe_name = f"{base}_{eid_str}{ext}".replace("/", "_").replace("\\", "_")

                path = os.path.join(RESUMES_DIR, safe_name)

                with open(path, "wb") as f:
                    f.write(content)

                saved_files.append(safe_name)


        if not saved_files:
            return JSONResponse({"message": "No new resumes found", "result": []})

        # Generate URLs
        urls = [f"/mcp/tools/mailmind/resume/{f}" for f in saved_files]
        imap.logout()

        return {
            "message": f"Fetched {len(saved_files)} resumes",
            "count": len(saved_files),
            "files": saved_files,
            "result": urls,
        }

    except imaplib.IMAP4.error:
        raise HTTPException(status_code=400, detail="Invalid credentials or IMAP error")
    except Exception as e:
        print("ERROR in fetch_resumes:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Server error: {e}")


# ============================================
#  SERVE RESUME FILES
# ============================================
@router.get("/mailmind/resume/{filename}")
def serve_resume(filename: str):
    file_path = os.path.join(RESUMES_DIR, filename)
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        media = "application/pdf"
        disp = f'inline; filename="{filename}"'
    else:
        media = "application/octet-stream"
        disp = f'attachment; filename="{filename}"'

    return FileResponse(file_path, media_type=media, headers={"Content-Disposition": disp})



# @router.post("/mailmind/upload-extracted")
# async def upload_extracted(
#     files: list[str],
#     overwrite: bool = False,
#     skip_duplicates: bool = False
# ):
#     session = Session(bind=engine)

#     to_process = []
#     skipped = []

#     for filename in files:
#         path = os.path.join(RESUME_DIR, filename)
#         if not os.path.exists(path):
#             skipped.append({"filename": filename, "reason": "file_not_found"})
#             continue

#         email = quick_extract_email(path)
#         existing = None
#         if email:
#             existing = session.query(Candidate).filter_by(email=email).first()

#         if existing and not overwrite:
#             skipped.append({
#                 "filename": filename,
#                 "email": email,
#                 "existing_name": existing.full_name
#             })
#             continue

#         if existing and overwrite:
#             session.delete(existing)
#             session.commit()

#         to_process.append({"filename": filename, "path": path})

#     session.close()

#     if not to_process:
#         return {"status": "skipped_all", "skipped": skipped}

#     # 🔥 INIT PROGRESS (ONCE)
#     with open(PROGRESS_FILE, "w") as f:
#         json.dump({
#             "total": len(to_process),
#             "processed": 0,
#             "completed": [],
#             "errors": [],
#             "status": "processing"
#         }, f, indent=2)

#     # 🔥 BACKGROUND WORK
#     def bg_job(entry):
#         filename = entry["filename"]
#         path = entry["path"]

#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)

#         try:
#             loop.run_until_complete(extract_and_store_resume(path))
#             update_progress(filename, "done")
#         except Exception:
#             traceback.print_exc()
#             update_progress(filename, "error")
#         finally:
#             loop.close()

#     for entry in to_process:
#         executor.submit(bg_job, entry)

#     return {
#         "status": "processing",
#         "processed": [e["filename"] for e in to_process],
#         "skipped": skipped
#     }


from typing import Optional
import re
from sqlalchemy.orm import Session

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}")

def quick_extract_email_fast(file_path: str) -> Optional[str]:
    """
    Lightweight email extraction (no LLM, no heavy PDF parsing).
    Reads first ~100KB and regex scans.
    """
    try:
        with open(file_path, "rb") as f:
            raw = f.read(100_000)

        text = raw.decode(errors="ignore")
        match = EMAIL_REGEX.search(text)
        return match.group(0) if match else None
    except Exception:
        return None


def email_exists(email: str) -> bool:
    try:
        session = Session(bind=engine)
        exists = session.query(Candidate).filter(Candidate.email == email).first() is not None
        session.close()
        return exists
    except Exception:
        return False

from pydantic import BaseModel

class UploadExtractedRequest(BaseModel):
    files: List[str]
    overwrite: bool = False


from sqlalchemy.orm import Session
from app.db import engine
from app.mcp.tools.resume_tool import quick_extract_email, Candidate

@router.post("/mailmind/upload-extracted")
async def upload_extracted(payload: dict):
    files = payload.get("files", [])
    overwrite = payload.get("overwrite", False)

    session = Session(bind=engine)

    to_upload = []
    duplicates = []
    skipped = []

    for fname in files:
        path = os.path.join(RESUMES_DIR, fname)
        if not os.path.exists(path):
            skipped.append({"filename": fname, "reason": "missing"})
            continue

        email = quick_extract_email(path)
        if email:
            exists = session.query(Candidate).filter(Candidate.email == email).first()
            if exists and not overwrite:
                duplicates.append({
                    "filename": fname,
                    "email": email,
                    "existing_name": exists.full_name
                })
                continue

        to_upload.append(fname)

    session.close()

    # Stop and report duplicates
    if duplicates and not overwrite:
        return {
            "status": "duplicate",
            "duplicates": duplicates,
            "skipped": skipped,
            "to_upload": to_upload
        }

    if not to_upload:
        return {
            "status": "nothing_to_upload",
            "duplicates": duplicates,
            "skipped": skipped
        }

    # Forward only valid files
    import httpx

    async with httpx.AsyncClient(timeout=300) as client:
        files_payload = []
        for fname in to_upload:
            path = os.path.join(RESUMES_DIR, fname)
            files_payload.append(("files", (fname, open(path, "rb"))))

        res = await client.post(
            "http://localhost:8000/mcp/tools/resume/upload",
            params={"overwrite": overwrite},
            files=files_payload
        )

    return {
        "status": "processing",
        "job_id": res.json().get("job_id"),
        "uploaded": to_upload,
        "duplicates": duplicates,
        "skipped": skipped
    }
