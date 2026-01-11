
import os
import re
import json
import shutil
import logging
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.responses import StreamingResponse
from fastapi import HTTPException, Response
import os
import re
from pathlib import Path
from PIL import Image
import httpx
import pytesseract
import pdfplumber
from pdf2image import convert_from_path
# Optional: PaddleOCR fallback
try:
    from paddleocr import PaddleOCR
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
except Exception:
    paddle_ocr = None

from app.mcp.server_core import register_mcp_tool

load_dotenv()
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set — Candidate validation may fail")

router = APIRouter()

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "candidate_validation.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

async def extract_text_from_file(file_path: str, debug_msgs: list) -> str:
    """Extract text from image or PDF file, with OCR fallback."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        # 🧾 Handle PDFs
        if ext == ".pdf":
            debug_msgs.append("📄 PDF detected — extracting text from PDF pages.")
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        text += page_text
                debug_msgs.append(f"🧠 PDF text extracted via pdfplumber (len={len(text)}).")

                # Fallback to OCR if pdfplumber found no text
                if len(text.strip()) < 50:
                    debug_msgs.append("⚠️ PDF appears scanned — switching to OCR fallback via pdf2image.")
                    pages = convert_from_path(file_path)
                    for page in pages:
                        ocr_text = pytesseract.image_to_string(page)
                        text += ocr_text
                    debug_msgs.append(f"✅ OCR fallback completed (len={len(text)}).")

            except Exception as e:
                debug_msgs.append(f"❌ PDF parsing failed ({e}) — using OCR fallback.")
                pages = convert_from_path(file_path)
                for page in pages:
                    text += pytesseract.image_to_string(page)
                debug_msgs.append(f"✅ OCR from PDF pages completed (len={len(text)}).")

        # 🖼️ Handle images
        else:
            debug_msgs.append("🖼️ Image detected — extracting text via OCR.")
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            debug_msgs.append(f"🧠 OCR text extracted from image (len={len(text)}).")

            if len(text.strip()) < 50 and paddle_ocr:
                debug_msgs.append("⚠️ Low text detected (<50 chars) — switching to PaddleOCR.")
                result = paddle_ocr.ocr(file_path, cls=True)
                text = " ".join([line[1][0] for line in result[0]]) if result else text
                debug_msgs.append("✅ PaddleOCR fallback completed.")

        if not text.strip():
            raise ValueError("No text extracted from document (empty after OCR).")

        return text.strip()

    except Exception as e:
        debug_msgs.append(f"❌ Text extraction failed: {e}")
        logger.exception("Text extraction failed")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {e}")
# ---------------------------------------------------------------------
# OCR Extraction
# ---------------------------------------------------------------------
async def extract_text_from_image(image_path: str, debug_msgs: list) -> str:
    """Extract text using pytesseract or PaddleOCR fallback."""
    try:
        img = Image.open(image_path)
        debug_msgs.append("✅ Image opened successfully for OCR.")

        text = pytesseract.image_to_string(img)
        debug_msgs.append(f"🧠 PyTesseract OCR completed. Text length: {len(text)}")

        # If pytesseract fails or returns too short text, try PaddleOCR
        if len(text.strip()) < 50 and paddle_ocr:
            debug_msgs.append("⚠️ Low text detected (<50 chars) — switching to PaddleOCR.")
            result = paddle_ocr.ocr(image_path, cls=True)
            text = " ".join([line[1][0] for line in result[0]]) if result else text
            debug_msgs.append("✅ PaddleOCR fallback completed.")

        if not text.strip():
            debug_msgs.append("❌ OCR failed to extract any text.")
            raise ValueError("No text extracted from image")

        return text.strip()

    except Exception as e:
        debug_msgs.append(f"❌ OCR extraction failed: {e}")
        logger.exception("OCR extraction failed")
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")


# ---------------------------------------------------------------------
# PAN Card Detection Logic
# ---------------------------------------------------------------------
def detect_pan_card(text: str, debug_msgs: list):
    """Detect if the extracted text belongs to an Indian PAN card."""
    text_upper = text.upper()

    pan_regex = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
    pan_numbers = pan_regex.findall(text_upper)

    pan_keywords = [
        "INCOME TAX DEPARTMENT",
        "GOVT",
        "GOVERNMENT OF INDIA",
        "PERMANENT ACCOUNT NUMBER",
        "INCOME TAX",
    ]
    keyword_hits = [kw for kw in pan_keywords if kw in text_upper]

    debug_msgs.append(f"🔍 PAN keywords found: {keyword_hits}")
    debug_msgs.append(f"🔢 PAN pattern matches: {pan_numbers}")

    # Confidence logic
    if len(keyword_hits) >= 2 and len(pan_numbers) >= 1:
        debug_msgs.append("✅ Confirmed Indian PAN card (high confidence).")
        return True, pan_numbers
    elif len(keyword_hits) >= 1 or len(pan_numbers) >= 1:
        debug_msgs.append("⚠️ Possibly a PAN card (medium confidence).")
        return False, pan_numbers
    else:
        debug_msgs.append("❌ Not a PAN card — missing key phrases and pattern.")
        return False, []


# ---------------------------------------------------------------------
# GPT Validation Helper
# ---------------------------------------------------------------------
async def call_gpt_api(prompt: str, debug_msgs: list) -> str:
    """Call GPT-4 API and return result."""
    try:
        debug_msgs.append("🧩 Sending prompt to GPT-4 API.")
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            debug_msgs.append("✅ GPT-4 API response received.")
            return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        debug_msgs.append(f"❌ GPT API call failed: {e}")
        logger.exception("GPT API call failed")
        raise HTTPException(status_code=500, detail=f"GPT API error: {e}")


# ---------------------------------------------------------------------
# Main Validation Function
# ---------------------------------------------------------------------
@register_mcp_tool(
    name="candidate.validate",
    description="Validate candidate PAN card and name using OCR + GPT-4",
)
async def validate_candidate_async(name: str, pan_file_path: str):
    debug_msgs = ["🚀 Starting candidate validation pipeline."]
    try:
        # extracted_text = await extract_text_from_image(pan_file_path, debug_msgs)
        extracted_text = await extract_text_from_file(pan_file_path, debug_msgs)
        debug_msgs.append(f"📜 OCR text extracted ({len(extracted_text)} chars).")

        # Step 1: Detect if this is a valid PAN card
        is_pan_card, pan_numbers = detect_pan_card(extracted_text, debug_msgs)

        # Default response
        validation = {
            "valid_name": False,
            "extracted_name": "",
            "message": ""
        }

        # Not a PAN card case
        if not is_pan_card:
            validation["message"] = (
                "❌ Please upload a valid PAN card (JPG or PDF) with clear visibility."
            )
            debug_msgs.append("❌ Document is not confirmed as a PAN card.")
            result = {
                "ok": True,
                "document_type": "UNKNOWN",
                "is_valid_pan_card": False,
                "pan_numbers": [],
                "validation": validation,
                "debug": debug_msgs,
            }
            return result

        # Step 2: PAN detected — verify name via GPT
        prompt = f"""
Candidate Name: {name}
Extracted text from PAN card: {extracted_text}

Task: Verify if the candidate name matches the name on the PAN card.
Return **only JSON**, strictly in this format:
{{
    "valid_name": true/false,
    "extracted_name": "<name from PAN>",
    "message": "<explanation>"
}}
"""
        raw_content = await call_gpt_api(prompt, debug_msgs)

        try:
            cleaned = raw_content.strip("` \n").replace("json", "")
            parsed = json.loads(cleaned)
            debug_msgs.append("✅ GPT response parsed successfully.")
        except json.JSONDecodeError:
            parsed = {"valid_name": False, "extracted_name": "", "message": raw_content}
            debug_msgs.append("⚠️ GPT response not valid JSON; fallback used.")

        # Simplify messages for frontend clarity
        if parsed.get("valid_name"):
            validation = {
                "valid_name": True,
                "message": "✅ PAN card successfully verified! Proceed to face capture."
            }
        else:
            validation = {
                "valid_name": False,
                "message": (
                    "❌ PAN card name mismatch or unclear. "
                    "Please upload a valid PAN card with better lighting."
                )
            }

        result = {
            "ok": True,
            "document_type": "PAN_CARD",
            "is_valid_pan_card": is_pan_card,
            "pan_numbers": pan_numbers,
            "validation": validation,
            "debug": debug_msgs,
        }

        logger.info(f"Candidate validation result: {result}")
        return result

    except Exception as e:
        debug_msgs.append(f"❌ Validation failed: {e}")
        logger.exception("Candidate validation failed")
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "debug": debug_msgs}
        )



# ---------------------------------------------------------------------
# Endpoint for Frontend
# ---------------------------------------------------------------------
@router.post("/validate_candidate")
async def validate_candidate_endpoint(name: str = Form(...), pan_file: UploadFile = File(...)):
    tmp_file = None
    debug_msgs = []
    try:
        debug_msgs.append("📥 File upload received from frontend.")
        suffix = os.path.splitext(pan_file.filename)[1] or ".png"
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.write(await pan_file.read())
        tmp_file.close()
        debug_msgs.append(f"💾 Temporary file saved at: {tmp_file.name}")

        result = await validate_candidate_async(name=name, pan_file_path=tmp_file.name)
        debug_msgs.append("✅ Validation completed successfully.")

        result["debug"].extend(debug_msgs)
        return JSONResponse(content=result)

    except Exception as e:
        logger.exception("Endpoint failed")
        debug_msgs.append(f"❌ Endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "debug": debug_msgs},
        )

    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
            debug_msgs.append("🧹 Temporary file cleaned up.")

from app.db import SessionLocal               # ✅ REQUIRED FIX
from app.mcp.tools.interview_attempts_model import InterviewAttempts
# ---------------------------------------------------------------------
# Face Image Save / Retrieve
# ---------------------------------------------------------------------
SAVE_DIR = Path("saved_faces")
SAVE_DIR.mkdir(exist_ok=True)

@router.post("/save_face_image")
async def save_face_image(
    attempt_id: int = Form(...),
    face_image: UploadFile = File(...)
):
    if not face_image:
        raise HTTPException(400, "No face image uploaded")

    db = SessionLocal()
    try:
        attempt = db.query(InterviewAttempts).get(attempt_id)
        if not attempt:
            raise HTTPException(404, "Invalid attempt_id")

        # 🔐 Stable person identity
        candidate_id = attempt.candidate_id

    finally:
        db.close()

    # 🔐 SAFE, ATTEMPT-SCOPED FILENAME
    candidate_safe = re.sub(r"[^A-Za-z0-9_]", "_", candidate_id)
    file_name = f"{candidate_safe}__attempt_{attempt_id}.png"
    file_path = SAVE_DIR / file_name

    with open(file_path, "wb") as f:
        shutil.copyfileobj(face_image.file, f)

    return {
        "ok": True,
        "message": "✅ Face image saved",
        "candidate_id": candidate_id,
        "attempt_id": attempt_id,
        "file_name": file_name,
    }


# @router.get("/get_face_image/{candidate_name}/{candidate_id}")
# async def get_face_image(candidate_name: str, candidate_id: str):
#     candidate_name_safe = re.sub(r"[^A-Za-z0-9_]", "_", candidate_name.strip())

#     for ext in [".png", ".jpg", ".jpeg"]:
#         file_path = SAVE_DIR / f"{candidate_name_safe}_{candidate_id}{ext}"
#         if file_path.exists():
#             return FileResponse(path=file_path, media_type="image/png")

#     raise HTTPException(status_code=404, detail="No face image found for candidate")

# @router.get("/get_face_image/{candidate_name}/{candidate_id}")
# async def get_face_image(candidate_name: str, candidate_id: str):
#     candidate_name_safe = re.sub(r"[^A-Za-z0-9_]", "_", candidate_name.strip())

#     # Search for the stored face file
#     for ext in [".png", ".jpg", ".jpeg"]:
#         file_path = SAVE_DIR / f"{candidate_name_safe}_{candidate_id}{ext}"
#         if file_path.exists():
#             # Read file as bytes
#             with open(file_path, "rb") as f:
#                 img_bytes = f.read()

#             # Return with correct CORS headers (required for html2canvas)
#             return Response(
#                 content=img_bytes,
#                 media_type="image/png",
#                 headers={
#                     "Access-Control-Allow-Origin": "*",             # Required
#                     "Access-Control-Allow-Credentials": "true",
#                     "Cross-Origin-Resource-Policy": "cross-origin",# REQUIRED for html2canvas
#                     "Cross-Origin-Embedder-Policy": "unsafe-none"  # REQUIRED for PDF capture
#                 }
#             )

#     raise HTTPException(status_code=404, detail="No face image found for candidate")
# @router.get("/get_face_image/{candidate_name}/{candidate_id}")
# async def get_face_image(candidate_name: str, candidate_id: str):
#     # Clean candidate name for file safety
#     candidate_name_safe = re.sub(r"[^A-Za-z0-9_]", "_", candidate_name.strip())
#     candidate_id_safe = re.sub(r"[^A-Za-z0-9_]", "_", candidate_id.strip())

#     print("\n================ FACE IMAGE FETCH ================")
#     print("Candidate:", candidate_name_safe)
#     print("Candidate ID:", candidate_id)

#     matched_file = None

#     # Detect which file exists
#     for ext in [".png", ".jpg", ".jpeg"]:
#         file_path = SAVE_DIR / f"{candidate_name_safe}_{candidate_id_safe}{ext}"
#         print("Checking:", file_path)

#         if file_path.exists():
#             matched_file = file_path
#             break

#     if not matched_file:
#         print("❌ No stored face image found!")
#         raise HTTPException(status_code=404, detail="No face image found for candidate")

#     print("✅ Found face image:", matched_file)

#     def file_stream():
#         with open(matched_file, "rb") as f:
#             yield from f

#     # IMPORTANT: correct headers for html2canvas / CORS
#     headers = {
#         "Access-Control-Allow-Origin": "*",
#         "Access-Control-Allow-Credentials": "true",
#         "Cross-Origin-Resource-Policy": "cross-origin",
#         "Cross-Origin-Embedder-Policy": "unsafe-none"
#     }

#     # Detect proper media type
#     media_type = "image/jpeg" if matched_file.suffix.lower() in [".jpg", ".jpeg"] else "image/png"

#     return StreamingResponse(
#         file_stream(),
#         media_type=media_type,
#         headers=headers
#     )

@router.get("/get_face_image/{attempt_id}")
async def get_face_image(attempt_id: int):
    """
    Fetch face image by attempt_id (authoritative).
    """
    print("\n================ FACE IMAGE FETCH ================")
    print("Attempt ID:", attempt_id)

    matched_file = None

    # Match: *_attempt_<attempt_id>.png|jpg|jpeg
    for ext in [".png", ".jpg", ".jpeg"]:
        pattern = f"__attempt_{attempt_id}{ext}"
        for file in SAVE_DIR.iterdir():
            if file.name.endswith(pattern):
                matched_file = file
                break
        if matched_file:
            break

    if not matched_file:
        print("❌ No stored face image found for attempt")
        raise HTTPException(status_code=404, detail="Face image not found")

    print("✅ Found face image:", matched_file.name)

    def file_stream():
        with open(matched_file, "rb") as f:
            yield from f

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Cross-Origin-Resource-Policy": "cross-origin",
    }

    media_type = (
        "image/jpeg"
        if matched_file.suffix.lower() in [".jpg", ".jpeg"]
        else "image/png"
    )

    return StreamingResponse(
        file_stream(),
        media_type=media_type,
        headers=headers,
    )