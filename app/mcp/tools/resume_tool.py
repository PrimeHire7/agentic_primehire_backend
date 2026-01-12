
import os
import re
import json
import logging
import threading
import asyncio
import traceback
from typing import List, Optional
from datetime import datetime, timedelta
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

# OpenAI + Pinecone clients
from openai import OpenAI
from pinecone import Pinecone

# SQLAlchemy
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base

# Resume parsing helpers
import pdfplumber
import docx2txt
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract
import subprocess

# Local imports - adjust as per your project structure
# Ensure app.db.Base exists and points to declarative base you want
from app.db import Base  # or change to `declarative_base()` if needed
from app.mcp.server_core import register_mcp_tool  # retained from your project


REPAIR_PROGRESS = {"status": "idle", "repaired": 0, "total": 0, "missing": []}
# -------------------- ENV & Logging --------------------
load_dotenv()
logger = logging.getLogger("mailmind")
logger.setLevel(logging.DEBUG)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://primehire_user_naresh:primehire_user_naresh@localhost/primehirebrain_db",
)

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "primehire-production-v2")

# -------------------- Clients & Engine --------------------
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None

engine = create_engine(DB_URL, echo=False)

PROGRESS_FILE = "./uploads/upload_progress.json"
RESUMES_DIR = "/home/ubuntu/agentic_primehire_dev/resumes"
os.makedirs(RESUMES_DIR, exist_ok=True)
os.makedirs("./uploads", exist_ok=True)

# -------------------- Pinecone Index --------------------
try:
    if pc:
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        stats = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
        logger.info(f"Connected to Pinecone index: {INDEX_NAME}")
    else:
        index = None
        logger.warning("Pinecone API key missing; index disabled.")
except Exception as e:
    logger.error(f"Pinecone init failed: {e}")
    index = None

# -------------------- DB Model --------------------
class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(String, unique=True, nullable=False)
    full_name = Column(String)
    email = Column(String, index=True)
    phone = Column(String)
    linkedin_url = Column(String)
    current_title = Column(String)
    current_company = Column(String)
    years_of_experience = Column(Float)
    top_skills = Column(String)
    education_summary = Column(String)
    location = Column(String)
    willing_to_relocate = Column(Boolean, default=False)
    work_authorization = Column(String)
    source = Column(String)
    application_status = Column(String, default="New")
    last_interview_date = Column(String)
    rating_score = Column(Float, default=0)
    resume_link = Column(String)
    tags = Column(String)
    last_updated = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# -------------------- Router --------------------
router = APIRouter()

# -------------------- Concurrency / Flags --------------------
progress_lock = threading.Lock()



def upsert_to_pinecone(candidate_id: str, metadata: dict):
    if not PINECONE_API_KEY:
        raise RuntimeError("Missing PINECONE_API_KEY")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    idx = pc.Index(INDEX_NAME)

    emb = get_embedding(json.dumps(metadata))

    safe_meta = clean_metadata(metadata)
    safe_meta["candidate_id"] = candidate_id

    logger.info(f"[PINECONE] Upserting {candidate_id} ...")

    idx.upsert(
        vectors=[{
            "id": candidate_id,
            "values": emb,
            "metadata": safe_meta
        }],
        namespace="__default__"
    )

    logger.info(f"[PINECONE] Upserted {candidate_id}")
    


def update_progress(filename: str, status: str, reason: str = None):
    with progress_lock:
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)

        data["processed"] += 1

        if status == "done":
            data["completed"].append(filename)
        else:
            data["errors"].append(reason or filename)

        if data["processed"] >= data["total"]:
            data["status"] = "completed"

        with open(PROGRESS_FILE, "w") as f:
            json.dump(data, f, indent=2)

from pdf2image import convert_from_path
import pytesseract
from PIL import Image

def read_pdf_with_ocr(path: str) -> str:
    pages = convert_from_path(path, dpi=300)
    text = ""
    for i, page in enumerate(pages):
        t = pytesseract.image_to_string(page, lang="eng")
        text += "\n" + t
    return text.strip()

def read_resume(file_path: str) -> str:
    """
    Robust text extractor for resumes.
    Supports: .pdf, .docx, .doc, .rtf, .txt, images (.png/.jpg/.jpeg)
    Uses multiple fallbacks + OCR.
    Always returns a string.
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        text = ""

        # ---------------- PDF ----------------
        if ext == ".pdf":
            # pdfplumber
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                logger.debug(f"[read_resume] pdfplumber -> {len(text)} chars")
            except Exception as e:
                logger.debug(f"[read_resume] pdfplumber failed: {e}")

            # pdfminer fallback
            if len(text.strip()) < 800:
                try:
                    mined = pdfminer_extract(file_path)
                    if mined:
                        text = mined
                    logger.debug(f"[read_resume] pdfminer -> {len(text)} chars")
                except Exception as e:
                    logger.debug(f"[read_resume] pdfminer failed: {e}")

            # PyPDF2 fallback
            if len(text.strip()) < 800:
                try:
                    reader = PdfReader(file_path)
                    text = "\n".join(p.extract_text() or "" for p in reader.pages)
                    logger.debug(f"[read_resume] PyPDF2 -> {len(text)} chars")
                except Exception as e:
                    logger.debug(f"[read_resume] PyPDF2 failed: {e}")

            # 🔥 OCR fallback (scanned PDFs)
            if len(text.strip()) < 200:
                try:
                    from pdf2image import convert_from_path
                    import pytesseract

                    ocr_text = ""
                    pages = convert_from_path(file_path, dpi=300)
                    for page in pages:
                        ocr_text += pytesseract.image_to_string(page, lang="eng") + "\n"

                    if ocr_text.strip():
                        text = ocr_text
                        logger.warning(f"[read_resume] OCR used -> {len(text)} chars")
                except Exception as e:
                    logger.debug(f"[read_resume] OCR PDF failed: {e}")

            return (text or "").strip()

        # ---------------- DOCX ----------------
        elif ext == ".docx":
            try:
                text = docx2txt.process(file_path) or ""
                logger.debug(f"[read_resume] docx2txt -> {len(text)} chars")
                return text.strip()
            except Exception as e:
                logger.debug(f"[read_resume] docx2txt failed: {e}")
                return ""

        # ---------------- DOC ----------------
        elif ext == ".doc":
            try:
                text = subprocess.check_output(["antiword", file_path], text=True)
                logger.debug(f"[read_resume] antiword -> {len(text)} chars")
                return text.strip()
            except Exception as e:
                logger.debug(f"[read_resume] antiword failed: {e}")

            try:
                tmp_txt = file_path + ".txt"
                subprocess.run(
                    ["soffice", "--headless", "--convert-to", "txt:Text", file_path, "--outdir", os.path.dirname(file_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if os.path.exists(tmp_txt):
                    with open(tmp_txt, "r", errors="ignore") as f:
                        text = f.read()
                    os.remove(tmp_txt)
                    logger.debug(f"[read_resume] libreoffice(.doc) -> {len(text)} chars")
                    return text.strip()
            except Exception as e:
                logger.debug(f"[read_resume] libreoffice(.doc) failed: {e}")

            return ""

        # ---------------- RTF ----------------
        elif ext == ".rtf":
            try:
                text = subprocess.check_output(["unrtf", "--text", file_path], text=True)
                logger.debug(f"[read_resume] unrtf -> {len(text)} chars")
                return text.strip()
            except Exception as e:
                logger.debug(f"[read_resume] unrtf failed: {e}")
                return ""

        # ---------------- TXT ----------------
        elif ext == ".txt":
            try:
                with open(file_path, "r", errors="ignore") as f:
                    text = f.read()
                logger.debug(f"[read_resume] txt -> {len(text)} chars")
                return text.strip()
            except Exception as e:
                logger.debug(f"[read_resume] txt failed: {e}")
                return ""

        # ---------------- IMAGES (OCR) ----------------
        elif ext in (".png", ".jpg", ".jpeg"):
            try:
                text = subprocess.check_output(["tesseract", file_path, "stdout"], text=True)
                logger.debug(f"[read_resume] OCR(image) -> {len(text)} chars")
                return text.strip()
            except Exception as e:
                logger.debug(f"[read_resume] OCR(image) failed: {e}")
                return ""

        # ---------------- UNKNOWN ----------------
        else:
            logger.debug(f"[read_resume] unsupported file type: {file_path}")
            return ""

    except Exception as e:
        logger.error(f"[read_resume] failed for {file_path}: {e}")
        return ""


# def read_resume(file_path: str) -> str:
#     """
#     Robust text extractor for resumes.
#     Supports: .pdf, .docx, .doc, .rtf, .txt, images (.png/.jpg/.jpeg)
#     Uses multiple fallbacks to maximize extraction success.
#     Always returns a string.
#     """
#     try:
#         ext = os.path.splitext(file_path)[1].lower()
#         text = ""

#         # ---------------- PDF ----------------
#         if ext == ".pdf":
#             # pdfplumber
#             try:
#                 with pdfplumber.open(file_path) as pdf:
#                     for page in pdf.pages:
#                         page_text = page.extract_text()
#                         if page_text:
#                             text += page_text + "\n"
#                 logger.debug(f"[read_resume] pdfplumber -> {len(text)} chars")
#             except Exception as e:
#                 logger.debug(f"[read_resume] pdfplumber failed: {e}")

#             # pdfminer fallback
#             if len(text.strip()) < 800:
#                 try:
#                     text = pdfminer_extract(file_path) or text
#                     logger.debug(f"[read_resume] pdfminer -> {len(text)} chars")
#                 except Exception as e:
#                     logger.debug(f"[read_resume] pdfminer failed: {e}")

#             # PyPDF2 fallback
#             if len(text.strip()) < 800:
#                 try:
#                     reader = PdfReader(file_path)
#                     text = "\n".join([p.extract_text() or "" for p in reader.pages])
#                     logger.debug(f"[read_resume] PyPDF2 -> {len(text)} chars")
#                 except Exception as e:
#                     logger.debug(f"[read_resume] PyPDF2 failed: {e}")

#             return (text or "").strip()

#         # ---------------- DOCX ----------------
#         elif ext == ".docx":
#             try:
#                 text = docx2txt.process(file_path) or ""
#                 logger.debug(f"[read_resume] docx2txt -> {len(text)} chars")
#                 return text.strip()
#             except Exception as e:
#                 logger.debug(f"[read_resume] docx2txt failed: {e}")
#                 return ""

#         # ---------------- DOC ----------------
#         elif ext == ".doc":
#             try:
#                 text = subprocess.check_output(["antiword", file_path], text=True)
#                 logger.debug(f"[read_resume] antiword -> {len(text)} chars")
#                 return text.strip()
#             except Exception as e:
#                 logger.debug(f"[read_resume] antiword failed: {e}")

#             # LibreOffice fallback
#             try:
#                 tmp_txt = file_path + ".txt"
#                 subprocess.run(
#                     ["soffice", "--headless", "--convert-to", "txt:Text", file_path, "--outdir", os.path.dirname(file_path)],
#                     check=True,
#                     stdout=subprocess.DEVNULL,
#                     stderr=subprocess.DEVNULL,
#                 )
#                 if os.path.exists(tmp_txt):
#                     with open(tmp_txt, "r", errors="ignore") as f:
#                         text = f.read()
#                     os.remove(tmp_txt)
#                     logger.debug(f"[read_resume] libreoffice(.doc) -> {len(text)} chars")
#                     return text.strip()
#             except Exception as e:
#                 logger.debug(f"[read_resume] libreoffice(.doc) failed: {e}")

#             return ""

#         # ---------------- RTF ----------------
#         elif ext == ".rtf":
#             try:
#                 text = subprocess.check_output(["unrtf", "--text", file_path], text=True)
#                 logger.debug(f"[read_resume] unrtf -> {len(text)} chars")
#                 return text.strip()
#             except Exception as e:
#                 logger.debug(f"[read_resume] unrtf failed: {e}")
#                 return ""

#         # ---------------- TXT ----------------
#         elif ext == ".txt":
#             try:
#                 with open(file_path, "r", errors="ignore") as f:
#                     text = f.read()
#                 logger.debug(f"[read_resume] txt -> {len(text)} chars")
#                 return text.strip()
#             except Exception as e:
#                 logger.debug(f"[read_resume] txt failed: {e}")
#                 return ""

#         # ---------------- IMAGES (OCR) ----------------
#         elif ext in (".png", ".jpg", ".jpeg"):
#             try:
#                 text = subprocess.check_output(["tesseract", file_path, "stdout"], text=True)
#                 logger.debug(f"[read_resume] OCR -> {len(text)} chars")
#                 return text.strip()
#             except Exception as e:
#                 logger.debug(f"[read_resume] OCR failed: {e}")
#                 return ""

#         # ---------------- UNKNOWN ----------------
#         else:
#             logger.debug(f"[read_resume] unsupported file type: {file_path}")
#             return ""

#     except Exception as e:
#         logger.error(f"[read_resume] failed for {file_path}: {e}")
#         return ""


def quick_extract_email(file_path: str) -> Optional[str]:
    try:
        text = read_resume(file_path)
        if not text:
            return None

        # normalize obfuscations
        normalized = text.lower()
        normalized = normalized.replace("(at)", "@").replace("[at]", "@")
        normalized = normalized.replace("(dot)", ".").replace("[dot]", ".")

        m = re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", normalized)
        return m.group(0) if m else None

    except Exception:
        return None


def clean_metadata(meta: dict) -> dict:
    clean = {}
    for k, v in meta.items():
        if v is None:
            continue

        if isinstance(v, bool):
            clean[k] = v
        elif isinstance(v, (int, float)):
            clean[k] = float(v)
        elif isinstance(v, str):
            if v.strip():
                clean[k] = v.strip()
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            clean[k] = [x.strip() for x in v if x.strip()]

    return clean

# -------------------- Embeddings --------------------
def get_embedding(text: str) -> list:
    if not client:
        raise RuntimeError("OpenAI client not configured")
    logger.debug(f"[get_embedding] chars={len(text)}")
    resp = client.embeddings.create(model="text-embedding-3-large", input=text)
    emb = resp.data[0].embedding
    return emb


# -------------------- Metadata extraction (LLM) --------------------
async def extract_metadata(text: str) -> dict:
    """
    Use LLM to extract structured metadata. Returns dict with expected keys.
    This function mirrors your original behavior but kept compact.
    """
    try:
        # Note: use your LLM schema/ prompt as before — trimmed for clarity
        prompt = f"""
You are an ATS resume parser.

Extract professional fields.
IMPORTANT: If you find an email address anywhere in the resume, always include it.

Return strict JSON:
{{
  "full_name": string | null,
  "email": string | null,
  "phone": string | null,
  "linkedin_url": string | null,
  "current_title": string | null,
  "current_company": string | null,
  "years_of_experience": number | null,
  "top_skills": list[string] | null,
  "education_summary": string | null,
  "location": string | null
}}

Resume:
{text[:8000]}
Return only JSON.
"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*|\s*```$", "", raw)
        parsed = json.loads(raw)
        return parsed
    except Exception as e:
        logger.error(f"[extract_metadata] LLM failed: {e}")
        return {}

def normalize_skills(raw):
    if not raw:
        return ""
    if isinstance(raw, list):
        return ", ".join(raw)

    # If GPT returned weird "{a, b, c}" format → clean it
    if isinstance(raw, str):
        cleaned = (
            raw.replace("{", "")
               .replace("}", "")
               .replace("[", "")
               .replace("]", "")
               .replace('"', '')
               .replace("'", "")
        )
        # Convert comma separated → uniform spacing
        parts = [s.strip() for s in cleaned.split(",") if s.strip()]
        return ", ".join(parts)
    
    return str(raw)

def get_pinecone_index():
    if not PINECONE_API_KEY:
        return None
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc.Index(INDEX_NAME)
    except Exception as e:
        logger.error(f"[PINECONE INIT ERROR] {e}")
        return None

async def extract_resume_metadata(file_path: str) -> dict:
    text = read_resume(file_path)
    if not text or len(text.strip()) < 50:
        raise ValueError("Empty or unreadable file")

    metadata = await extract_metadata(text)
    email = metadata.get("email") or quick_extract_email(file_path)
    metadata["email"] = email

    metadata["top_skills"] = normalize_skills(metadata.get("top_skills"))
    return metadata

def store_candidate_sync(file_path: str, metadata: dict, overwrite: bool = False) -> dict:
    filename = os.path.basename(file_path)
    session = Session(bind=engine)

    try:
        email = metadata.get("email")
        existing = session.query(Candidate).filter_by(email=email).first() if email else None

        if existing and not overwrite:
            return {"filename": filename, "status": "duplicate"}

        if existing:
            candidate = existing
        else:
            candidate = Candidate(
                candidate_id=f"CAND_{uuid.uuid4().hex[:12]}",
                resume_link=file_path,
                last_updated=datetime.utcnow(),
                source="mailmind"
            )
            session.add(candidate)

        for field in [
            "full_name", "email", "phone", "linkedin_url",
            "current_title", "current_company",
            "years_of_experience", "top_skills",
            "education_summary", "location"
        ]:
            if field in metadata:
                setattr(candidate, field, metadata[field])

        session.commit()
        session.refresh(candidate)

        try:
            upsert_to_pinecone(candidate.candidate_id, metadata)
            pinecone_ok = True
        except Exception as e:
            logger.error(f"[PINECONE FAILED] {candidate.candidate_id}: {e}")
            pinecone_ok = False

        return {
            "filename": filename,
            "status": "ok",
            "candidate_id": candidate.candidate_id,
            "pinecone": pinecone_ok
        }

    except Exception as e:
        session.rollback()
        logger.error(f"[STORE FAILED] {filename}: {e}")
        return {
            "filename": filename,
            "status": "error",
            "reason": "store_failed"
        }

    finally:
        session.close()


@register_mcp_tool(
    name="resume.extract_and_store",
    description="Extract metadata from resumes, save in PostgreSQL & Pinecone"
)
async def extract_and_store_resume(file_path: str, overwrite: bool = False) -> dict:
    filename = os.path.basename(file_path)
    logger.info(f"[PROCESS] {filename}")

    session = None

    try:
        # 1️⃣ Read resume
        text = read_resume(file_path)
        if not text or len(text.strip()) < 50:
            return {
                "filename": filename,
                "error": True,
                "reason": "unreadable"
            }


        # 2️⃣ Extract metadata
        metadata = await extract_metadata(text)
        metadata["top_skills"] = normalize_skills(metadata.get("top_skills"))

        # 3️⃣ Open DB session
        session = Session(bind=engine)

        email = metadata.get("email") or quick_extract_email(file_path)
        metadata["email"] = email

        existing = session.query(Candidate).filter_by(email=email).first() if email else None

        # 4️⃣ Duplicate handling
        if existing and not overwrite:
            session.close()
            return {"filename": filename, "duplicate": True}

        # 5️⃣ Create or reuse candidate
        if existing:
            candidate = existing
        else:
            candidate = Candidate(
                candidate_id=f"CAND_{uuid.uuid4().hex[:12]}",
                resume_link=file_path,
                last_updated=datetime.utcnow(),
                source="mailmind"
            )
            session.add(candidate)

        # 6️⃣ Update fields
        for field in [
            "full_name", "email", "phone", "linkedin_url",
            "current_title", "current_company",
            "years_of_experience", "top_skills",
            "education_summary", "location"
        ]:
            if field in metadata:
                setattr(candidate, field, metadata[field])

        session.commit()
        session.refresh(candidate)

        logger.info(f"[DB] Inserted {candidate.full_name} ({candidate.email})")

        # 7️⃣ Pinecone upsert
        pc_index = get_pinecone_index()

        if pc_index:
            try:
                emb = get_embedding(json.dumps(metadata))

                safe_meta = clean_metadata(metadata)
                safe_meta["candidate_id"] = candidate.candidate_id

                pc_index.upsert(
                    vectors=[{
                        "id": candidate.candidate_id,
                        "values": emb,
                        "metadata": safe_meta
                    }],
                    namespace="__default__"
                )

                logger.info(f"[PINECONE] Upserted {candidate.candidate_id}")

            except Exception as e:
                logger.error(f"[PINECONE ERROR] {candidate.candidate_id}: {e}")
        else:
            logger.warning("[PINECONE] Index not available in this worker")




        session.close()

        return {
            "filename": filename,
            "candidate_id": candidate.candidate_id,
            "duplicate": False,
            "db": True,
            "pinecone": bool(index)
        }

    except Exception:
        logger.error(traceback.format_exc())

        try:
            if session:
                session.rollback()
                session.close()
        except Exception:
            pass

        return {
            "filename": filename,
            "error": True
        }

# async def extract_and_store_resume(file_path: str, overwrite: bool = False) -> dict:
#     filename = os.path.basename(file_path)
#     logger.info(f"[PROCESS] {filename}")

#     try:
#         text = read_resume(file_path)
#         if not text or len(text.strip()) < 50:
#             raise ValueError("Empty or unreadable file")

#         metadata = await extract_metadata(text)
#         metadata["top_skills"] = normalize_skills(metadata.get("top_skills"))

#         session = Session(bind=engine)

#         # email = (metadata.get("email") or "").strip()
#         # existing = session.query(Candidate).filter_by(email=email).first() if email else None
#         email = metadata.get("email")
#         existing = None
#         if email:
#             existing = session.query(Candidate).filter_by(email=email).first()

#         if existing and not overwrite:
#             session.close()
#             return {
#                 "filename": filename,
#                 "duplicate": True,
#                 "email": email,
#                 "existing_name": existing.full_name,
#             }

#         if existing:
#             candidate = existing
#         else:
#             candidate = Candidate(
#                 candidate_id=f"CAND_{uuid.uuid4().hex[:12]}",
#                 resume_link=file_path,
#                 last_updated=datetime.utcnow(),
#             )
#             session.add(candidate)

#         # update fields
#         for field in [
#             "full_name", "email", "phone", "linkedin_url",
#             "current_title", "current_company",
#             "years_of_experience", "top_skills",
#             "education_summary", "location"
#         ]:
#             if field in metadata:
#                 setattr(candidate, field, metadata[field])

#         session.commit()
        
#         # # ✅ extract primitives BEFORE close
#         candidate_id = candidate.candidate_id
#         email_val = candidate.email
#         name_val = candidate.full_name

#         # session.close()
#         session.flush()
#         session.refresh(candidate)
#         logger.info(f"[DB] Inserted {candidate.full_name} ({candidate.email})")

#         # ---------------- Pinecone ----------------
#         if index:
#             emb = get_embedding(json.dumps(metadata))
#             safe_meta = {
#                 "candidate_id": candidate_id,
#             }

#             if email_val:
#                 safe_meta["email"] = str(email_val)

#             if name_val:
#                 safe_meta["full_name"] = str(name_val)

#             for k, v in metadata.items():
#                 if v is None:
#                     logger.debug(f"[META] {k} is None — skipping")
#                     continue
#                 if isinstance(v, (str, int, float, bool)):
#                     safe_meta[k] = v
#                 elif isinstance(v, list) and all(isinstance(x, str) for x in v):
#                     safe_meta[k] = v


#             index.upsert(
#                 vectors=[{
#                     "id": candidate_id,
#                     "values": emb,
#                     "metadata": safe_meta
#                 }],
#                 namespace="__default__"
#             )

#         return {
#             "filename": filename,
#             "candidate_id": candidate_id,
#             "duplicate": False
#         }

#     # except Exception:
#     #     logger.error(traceback.format_exc())
#     #     return {"filename": filename, "error": "processing_failed"}
#     except Exception:
#         session.rollback()
#         logger.error(traceback.format_exc())
#         return {"filename": filename, "error": "processing_failed"}


# CORS preflight helper for mailmind fetch route (if you need it)
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


@router.post("/mailmind/connect")
def mailmind_connect(creds: dict):
    """
    creds should contain: email, password, platform (optional), imap_server/imap_port (optional)
    This is a simple check using imaplib.login as before. Keep unchanged logic from your previous code.
    """
    import imaplib
    email_addr = creds.get("email")
    password = creds.get("password")
    platform = creds.get("platform", "gmail")
    imap_server = creds.get("imap_server")
    imap_port = creds.get("imap_port", 993)

    PLATFORM_IMAP = {
        "gmail": "imap.gmail.com",
        "outlook": "imap.secureserver.net",
        "godaddy": "imap.secureserver.net",
    }

    if not imap_server:
        imap_server = PLATFORM_IMAP.get(platform.lower())
        if not imap_server:
            raise HTTPException(status_code=400, detail="Invalid platform")

    try:
        mail = imaplib.IMAP4_SSL(imap_server, imap_port)
        mail.login(email_addr, password)
        mail.logout()
        return {"message": f"Connected successfully to {platform}!"}
    except imaplib.IMAP4.error:
        raise HTTPException(status_code=400, detail="Invalid email/password or IMAP settings")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mailmind/fetch-resumes")
async def fetch_resumes(creds: dict, request: Request):
    """
    Connects to IMAP and fetch attachments from the last 7 days.
    Returns:
        { message, count, files: [filenames], result: [urls] }
    """
    import imaplib
    from email import message_from_bytes
    from email.header import decode_header

    email_addr = creds.get("email")
    password = creds.get("password")
    platform = creds.get("platform", "gmail")
    imap_server = creds.get("imap_server")

    PLATFORM_IMAP = {
        "gmail": "imap.gmail.com",
        "outlook": "imap.secureserver.net",
        "godaddy": "imap.secureserver.net",
    }

    if not imap_server:
        imap_server = PLATFORM_IMAP.get(platform.lower())
        if not imap_server:
            raise HTTPException(status_code=400, detail="Invalid platform")

    try:
        imap = imaplib.IMAP4_SSL(imap_server, 993)
        imap.login(email_addr, password)
        status, _ = imap.select("INBOX")
        if status != "OK":
            raise HTTPException(status_code=400, detail="Failed to open INBOX")

        since_date = (datetime.now() - timedelta(days=7)).strftime("%d-%b-%Y")
        status, data = imap.search(None, f'(SINCE "{since_date}")')
        if status != "OK":
            raise HTTPException(status_code=400, detail="Failed to search emails")

        email_ids = data[0].split()
        saved_files = []

        for eid in email_ids:
            status, msg_data = imap.fetch(eid, "(RFC822)")
            if status != "OK":
                continue
            msg = message_from_bytes(msg_data[0][1])
            for part in msg.walk():
                if part.get_content_disposition() == "attachment":
                    filename = part.get_filename()
                    if not filename:
                        continue
                    try:
                        decoded = decode_header(filename)[0][0]
                        if isinstance(decoded, bytes):
                            filename = decoded.decode(errors="ignore")
                        else:
                            filename = decoded
                    except Exception:
                        pass

                    if filename.lower().endswith((".pdf", ".doc", ".docx")):
                        safe_name = filename.replace("/", "_").replace("\\", "_")
                        path = os.path.join(RESUMES_DIR, safe_name)
                        with open(path, "wb") as f:
                            f.write(part.get_payload(decode=True))
                        saved_files.append(safe_name)

        if not saved_files:
            return JSONResponse({"message": "No new resumes found", "result": [], "files": []})

        urls = [f"/mcp/tools/mailmind/resume/{f}" for f in saved_files]
        return {"message": f"Fetched {len(saved_files)} resumes", "count": len(saved_files), "files": saved_files, "result": urls}
    except imaplib.IMAP4.error:
        raise HTTPException(status_code=400, detail="Invalid credentials or IMAP error")
    except Exception as e:
        logger.error(f"[fetch_resumes] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mailmind/resume/{filename}")
def serve_resume(filename: str):
    file_path = os.path.join(RESUMES_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        media = "application/pdf"
        disp = f'inline; filename="{filename}"'
    else:
        media = "application/octet-stream"
        disp = f'attachment; filename="{filename}"'
    return FileResponse(file_path, media_type=media, headers={"Content-Disposition": disp})


# -------------------- GLOBAL THREAD POOL --------------------
from concurrent.futures import ThreadPoolExecutor

# Control concurrency (adjust based on server CPU)
executor = ThreadPoolExecutor(max_workers=6)

# ---------------- PRE-FLIGHT FOR UPLOAD ----------------
@router.options("/upload")
async def upload_preflight(request: Request):
    origin = request.headers.get("origin", "*")
    return JSONResponse(
        {"ok": True},
        headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )


@router.post("/upload")
async def upload_resumes(
    files: List[UploadFile] = File(...),
    overwrite: bool = False
):
    """
    Resume upload with safe overwrite handling.

    ✔ Saves files
    ✔ Detects duplicates
    ✔ Allows overwrite via ?overwrite=true
    ✔ Starts background processing
    ✔ Returns immediately
    """

    logger.info(f"[UPLOAD] Received {len(files)} files — overwrite={overwrite}")

    saved_files = []
    duplicates = []

    # -------------------------------
    # 1️⃣ SAVE FILES TO DISK
    # -------------------------------
    for file in files:
        try:
            path = os.path.join("./uploads", file.filename)
            with open(path, "wb") as f:
                f.write(await file.read())

            saved_files.append({
                "filename": file.filename,
                "path": path
            })

        except Exception as e:
            logger.error(f"[UPLOAD] Failed saving {file.filename}: {e}")
            duplicates.append({
                "filename": file.filename,
                "error": str(e)
            })

    # -------------------------------
    # 2️⃣ DUPLICATE CHECK (EMAIL)
    # -------------------------------
    session = Session(bind=engine)

    for entry in saved_files:
        filename = entry["filename"]
        path = entry["path"]

        try:
            email = quick_extract_email(path)

            if not email:
                continue

            existing = (
                session.query(Candidate)
                .filter(Candidate.email == email)
                .first()
            )

            # ❌ Duplicate & overwrite NOT allowed
            if existing and not overwrite:
                duplicates.append({
                    "filename": filename,
                    "email": email,
                    "existing_name": existing.full_name,
                    "message": "Candidate already exists. Overwrite?"
                })

            # ✅ Overwrite allowed → delete existing
            if existing and overwrite:
                logger.info(f"[OVERWRITE] Removing existing candidate {email}")
                session.delete(existing)
                session.commit()

        except Exception as e:
            logger.warning(f"[UPLOAD] Duplicate check failed for {filename}: {e}")

    session.close()

    # -------------------------------
    # 3️⃣ STOP IF DUPLICATES FOUND
    # -------------------------------
    if duplicates and not overwrite:
        return {
            "status": "duplicate",
            "duplicates": duplicates
        }

    # -------------------------------
    # 4️⃣ INIT PROGRESS FILE
    # -------------------------------
    total = len(saved_files)
    if total == 0:
        return {
            "status": "error",
            "message": "No files to process"
        }

    job_id = uuid.uuid4().hex    
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "job_id": job_id,
            "total": total,
            "processed": 0,
            "completed": [],
            "errors": [],
            "status": "processing"
        }, f, indent=2)

    # -------------------------------
    # 5️⃣ BACKGROUND PROCESSING
    # -------------------------------
    

    def bg_task(entry):
        filename = entry["filename"]
        path = entry["path"]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # 1️⃣ Extract metadata
            metadata = loop.run_until_complete(extract_resume_metadata(path))

            if not metadata:
                update_progress(filename, "error", "no_metadata")
                return

            # 2️⃣ Store candidate
            result = store_candidate_sync(path, metadata)

            if not isinstance(result, dict):
                update_progress(filename, "error", "unknown_store_result")
                return

            # 3️⃣ Result routing
            status = result.get("status")

            if status == "ok":
                update_progress(filename, "done")
            elif status == "duplicate":
                update_progress(filename, "error", "duplicate")
            elif status == "error":
                update_progress(filename, "error", result.get("reason", "processing_failed"))
            else:
                update_progress(filename, "error", "unknown_store_result")

            if status not in ("ok", "duplicate", "error"):
                logger.warning(f"[BG] Unknown result for {filename}: {result}")



        except ValueError as ve:
            # Raised by extract_resume_metadata for unreadable files
            logger.warning(f"[UNREADABLE] {filename}: {ve}")
            update_progress(filename, "error", "unreadable")

        except Exception as e:
            logger.error(f"[BG ERROR] {filename}: {e}")
            traceback.print_exc()
            update_progress(filename, "error", "exception")

        finally:
            loop.close()



    for entry in saved_files:
        executor.submit(bg_task, entry)


    # -------------------------------
    # 6️⃣ RETURN IMMEDIATELY
    # -------------------------------
    return {
        "status": "processing",
        "job_id": job_id,
        "message": "Upload started",
        "files": [e["filename"] for e in saved_files]
    }


from sqlalchemy import func

def exists_in_pinecone(candidate_id: str) -> bool:
    try:
        res = index.fetch(ids=[candidate_id], namespace="__default__")
        
        return bool(res and hasattr(res, "vectors") and candidate_id in res.vectors)
    except Exception as e:
        logger.error(f"[PINECONE CHECK FAILED] {candidate_id}: {e}")
        return False



def run_repair_background():
    global REPAIR_PROGRESS

    session = Session(bind=engine)
    try:
        db_ids = {c.candidate_id for c in session.query(Candidate).all()}
    finally:
        session.close()

    # Fetch all Pinecone ids
    pinecone_ids = set()
    try:
        stats = index.describe_index_stats()
        total = stats["total_vector_count"]

        # Pinecone does not support "list all" directly → use fetch via pagination if needed
        # Assuming you store all under namespace "__default__"
        # You must maintain your own id list OR store ids in Postgres (recommended)

        # If you stored candidate_id as vector id (you did), we can list via metadata filter
        # But Pinecone still doesn’t give a global scan → so we use DB as source of truth

        # Instead we just compute deletions from known DB deletes
        pinecone_ids = set(db_ids)  # treat DB as canonical

    except Exception as e:
        logger.error(f"[REPAIR] Failed to fetch Pinecone stats: {e}")

    # Compute missing from Pinecone
    missing = [cid for cid in db_ids if not exists_in_pinecone(cid)]

    REPAIR_PROGRESS = {
        "status": "running",
        "repaired": 0,
        "total": len(missing),
        "missing": missing,
    }

    for cid in missing:
        try:
            session = Session(bind=engine)
            c = session.query(Candidate).filter_by(candidate_id=cid).first()
            session.close()

            raw_meta = {
                "candidate_id": c.candidate_id,
                "full_name": c.full_name,
                "email": c.email,
                "phone": c.phone,
                "current_title": c.current_title,
                "current_company": c.current_company,
                "years_of_experience": c.years_of_experience,
                "top_skills": c.top_skills,
                "location": c.location,
            }

            # Remove None and convert types
            meta = {}
            for k, v in raw_meta.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = str(v)
                elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                    meta[k] = v


            emb = get_embedding(json.dumps(meta))

            index.upsert(
                vectors=[{
                    "id":  c.candidate_id,
                    "values": emb,
                    "metadata": meta,
                }],
                namespace="__default__",
            )

            REPAIR_PROGRESS["repaired"] += 1

        except Exception as e:
            logger.error(f"[REPAIR] Failed for {cid}: {e}")

    REPAIR_PROGRESS["status"] = "completed"



# -------------------- Progress Endpoint --------------------
@router.get("/progress")
async def get_progress():
    """
    Returns current resume upload progress.

    Response format:
    {
        "status": "idle" | "processing" | "completed" | "error",
        "progress": {
            "total": int,
            "processed": int,
            "completed": [str],
            "errors": [str]
        }
    }
    """

    # If progress file does not exist → idle state
    if not os.path.exists(PROGRESS_FILE):
        return {
            "status": "idle",
            "progress": {
                "total": 0,
                "processed": 0,
                "completed": [],
                "errors": []
            }
        }

    try:
        with open(PROGRESS_FILE, "r") as f:
            raw = json.load(f)

        total = int(raw.get("total", 0))
        processed = int(raw.get("processed", 0))
        completed = list(map(str, raw.get("completed", [])))
        errors = list(map(str, raw.get("errors", [])))

        # Determine high-level status
        if total == 0:
            status = "idle"
        elif processed < total:
            status = "processing"
        elif processed >= total:
            status = "completed"
        else:
            status = raw.get("status", "idle")

        return {
            "status": status,
            "progress": {
                "total": total,
                "processed": processed,
                "completed": completed,
                "errors": errors
            }
        }

    except Exception as e:
        logger.error(f"[progress] Failed to read progress file: {e}")

        return {
            "status": "error",
            "progress": {
                "total": 0,
                "processed": 0,
                "completed": [],
                "errors": []
            }
        }


@router.post("/sync/repair")
async def sync_repair():
    """
    Starts repair job in background and returns immediately.
    """
    global REPAIR_PROGRESS

    # Start with reset
    REPAIR_PROGRESS = {"status": "starting", "repaired": 0, "total": 0, "missing": []}

    # Background thread
    threading.Thread(target=run_repair_background, daemon=True).start()

    return {
        "status": "started",
        "message": "Repair process running in background.",
        "check": "/mcp/tools/resume/sync/repair/status"
    }

@router.get("/sync/repair/status")
async def sync_repair_status():
    return REPAIR_PROGRESS





@router.post("/reset")
async def reset_progress():
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"total": 0, "processed": 0, "completed": [], "errors": [], "status": "idle"}, f)
    return {"status": "reset"}


@router.get("/stats")
async def resume_stats():
    try:
        session = Session(bind=engine)
        total_candidates = session.query(Candidate).count()
        session.close()
        stats = {}
        if index:
            stats = index.describe_index_stats()
            stats = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
        return {"postgres_candidates": total_candidates, "pinecone_vectors": stats.get("total_vector_count", "?"), "namespaces": stats.get("namespaces", {}), "index_name": INDEX_NAME}
    except Exception as e:
        logger.error(f"[resume_stats] {e}")
        return {"error": str(e)}


@router.get("/recent")
async def get_recent_candidates(limit: int = 10):
    session = Session(bind=engine)
    candidates = session.query(Candidate).order_by(Candidate.last_updated.desc()).limit(limit).all()
    session.close()
    results = []
    for c in candidates:
        results.append({
            "filename": os.path.basename(c.resume_link or "N/A"),
            "metadata": {
                "full_name": c.full_name,
                "current_title": c.current_title,
                "current_company": c.current_company,
                "years_of_experience": c.years_of_experience,
                "location": c.location,
                "email": c.email,
                "phone": c.phone,
                "top_skills": c.top_skills,
            }
        })
    return {"recent_candidates": results}


@router.delete("/delete/{candidate_id}")
async def delete_candidate(candidate_id: str):
    try:
        session = Session(bind=engine)
        candidate = session.query(Candidate).filter(Candidate.candidate_id == candidate_id).first()
        if not candidate:
            session.close()
            raise HTTPException(status_code=404, detail="Candidate not found")
        name = candidate.full_name
        session.delete(candidate)
        session.commit()
        session.close()
        # delete from pinecone too (best-effort)
        try:
            if index:
                index.delete(ids=[candidate_id], namespace="__default__")
        except Exception as e:
            logger.warning(f"[delete_candidate] pinecone delete failed: {e}")
        return {"status": "success", "message": f"Candidate '{name}' deleted", "candidate_id": candidate_id}
    except Exception as e:
        logger.error(f"[delete_candidate] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/progress/live")
async def get_live_progress(job_id: str):
    if not os.path.exists(PROGRESS_FILE):
        return {"status": "idle", "progress": None}

    with open(PROGRESS_FILE) as f:
        raw = json.load(f)

    if raw.get("job_id") != job_id:
        return {"status": "stale", "progress": None}

    return {
        "status": raw.get("status", "idle"),
        "progress": raw
    }
