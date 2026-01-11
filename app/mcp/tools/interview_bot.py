
import os
import io
import time
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
import re

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

# DB + transcript
from app.db import SessionLocal
from app.mcp.tools.interview_attempts_model import InterviewAttempts
from app.mcp.tools.interview_bot_beta import transcript_file

# Mediapipe
import mediapipe as mp

# Face recognition (InsightFace)
import insightface
from insightface.app import FaceAnalysis

router = APIRouter()

# -----------------------
# Configurable constants
# -----------------------
SNAPSHOT_DIR = "snapshots"
LOG_FILE = "logs/anomalies.json"
SAVE_DIR = Path("saved_faces")
SESSION_STORE = {}
DEBUG_FACE_MATCH = True

# Behavior thresholds (tweak as needed)
FPS = 3  # frames per second expected from frontend (set your frontend interval accordingly)
NODDING_REQUIRED_SECONDS = 10 
SCANNING_REQUIRED_SECONDS = 10   
GAZE_AWAY_REQUIRED_SECONDS = 3

# Movement thresholds
DX_SCAN_THRESHOLD = 15      # horizontal movement threshold for scanning
DY_NOD_THRESHOLD = 18       # vertical movement threshold for nodding
MOVEMENT_STILL_THRESHOLD = 0.5
STRESS_STDDEV_THRESHOLD = 15

# Pose thresholds (degrees)
GAZE_AWAY_YAW_THRESHOLD = 28
GAZE_AWAY_PITCH_THRESHOLD = 22

# Blink thresholds
BLINK_THRESHOLD = 0.27
BLINK_RECOVERY = 0.30
# BLINK_TRIGGER_FRAMES = 150  # frames after which no-blink becomes anomaly (if blink_count==0)
BLINK_TRIGGER_FRAMES = 30       
# Cooldown defaults (seconds)
DEFAULT_COOLDOWN = 60

# Derived frame counts
NODDING_REQUIRED_FRAMES = int(NODDING_REQUIRED_SECONDS * FPS)
SCANNING_REQUIRED_FRAMES = int(SCANNING_REQUIRED_SECONDS * FPS)
GAZE_AWAY_REQUIRED_FRAMES = int(GAZE_AWAY_REQUIRED_SECONDS * FPS)

# ---------- Face recognition strengthening ----------
FACE_MATCH_STRONG = 0.80
FACE_MATCH_OK = 0.65
FACE_MATCH_SUSPICIOUS = 0.55
FACE_MATCH_CRITICAL = 0.45

FACE_MISMATCH_CONFIRM_SECONDS = 1
FACE_MISMATCH_CONFIRM_FRAMES = int(FACE_MISMATCH_CONFIRM_SECONDS * FPS)

FACE_SIM_HISTORY_SIZE = 30
# 🔥 NEW (instant mismatch shown in UI)
FACE_MISMATCH_INSTANT_THRESHOLD = 0.40
FACE_MISMATCH_COOLDOWN = 30
# Ensure dirs exist
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
SAVE_DIR.mkdir(exist_ok=True)

# -----------------------
# Initialize models
# -----------------------
face_rec = FaceAnalysis(name="buffalo_l")
face_rec.prepare(ctx_id=-1)

REGISTERED_EMBEDS = {}

mp_face_det = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_det.FaceDetection(model_selection=0, min_detection_confidence=0.60)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Eye landmarks for EAR calculation (Mediapipe indexes)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# -----------------------
# Utility helpers
# -----------------------
def dbg(*args):
    if DEBUG_FACE_MATCH:
        print(*args)

def should_trigger(sess, anomaly_type: str, cooldown: int = DEFAULT_COOLDOWN) -> bool:
    """
    Return True only if the given anomaly_type was not triggered within the cooldown window.
    Updates sess['last_trigger'][anomaly_type] on successful trigger.
    """
    sess.setdefault("last_trigger", {})
    last = sess["last_trigger"].get(anomaly_type, 0)
    now = time.time()
    if (now - last) >= cooldown:
        sess["last_trigger"][anomaly_type] = now
        return True
    return False


# def ensure_session(candidate_id: str) -> dict:
#     """Ensure a session dict exists and set default counters."""
#     if candidate_id not in SESSION_STORE:
#         SESSION_STORE[candidate_id] = {}

#     sess = SESSION_STORE[candidate_id]

#     # basic counters
#     sess.setdefault("frames_processed", 0)
#     sess.setdefault("no_face_count", 0)
#     sess.setdefault("anomalies", [])
#     sess.setdefault("anomaly_counts", {})

#     # liveness / movement
#     sess.setdefault("prev_centroid", None)
#     sess.setdefault("still_frames", 0)
#     sess.setdefault("ear_history", [])
#     sess.setdefault("blink_count", 0)

#     # movement histories
#     sess.setdefault("movement_history", [])
#     sess.setdefault("nod_history", [])
#     sess.setdefault("left_right_history", [])
#     sess.setdefault("last_distances", [])

#     # specialized counters (for continuity windows)
#     sess.setdefault("vertical_nod_count", 0)
#     sess.setdefault("horizontal_scan_count", 0)
#     sess.setdefault("gaze_away_frames", 0)

#     # cooldown tracking is handled via should_trigger's last_trigger

#     return sess
def ensure_session(attempt_id: int) -> dict:
    if attempt_id not in SESSION_STORE:
        SESSION_STORE[attempt_id] = {
            "attempt_id": attempt_id,
            "frames_processed": 0,
            "no_face_count": 0,
            "anomalies": [],
            "anomaly_counts": {},
            "last_trigger": {},
            "prev_centroid": None,
            "still_frames": 0,
            "ear_history": [],
            "blink_count": 0,
            "movement_history": [],
            "last_distances": [],
            "vertical_nod_count": 0,
            "horizontal_scan_count": 0,
            "gaze_away_frames": 0,
            "face_mismatch_streak": 0,
            "face_sim_history": [],
        }
    return SESSION_STORE[attempt_id]


def append_to_log(entry: dict):
    """Append anomaly to local anomalies.json (safe, append-only)."""
    try:
        data = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
        data.append(entry)
        with open(LOG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"⚠ Failed writing log: {e}")



def save_anomaly_to_db(attempt_id: int, anomaly: dict):
    db = SessionLocal()
    try:
        attempt = db.query(InterviewAttempts).get(attempt_id)
        if not attempt:
            return

        attempt.anomalies = (attempt.anomalies or []) + [anomaly]
        attempt.updated_at = datetime.utcnow()
        db.commit()

    finally:
        db.close()

def save_anomaly_to_transcript(attempt_id: int, message: str):
    path = transcript_file(attempt_id)

    if not os.path.exists(path):
        return

    try:
        with open(path, "r") as f:
            data = json.load(f)

        data.setdefault("conversation", []).append({
            "sender": "system",
            "text": f"⚠ {message}",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print("⚠ save_anomaly_to_transcript error:", e)


# -----------------------
# Face helpers
# -----------------------
def get_face_embedding(img_bgr):
    """Return insightface embedding (numpy float32) or None."""
    try:
        if img_bgr is None:
            print("get_face_embedding: img is None")
            return None
        h, w = img_bgr.shape[:2]
        if h < 20 or w < 20:
            print(f"get_face_embedding: image too small ({w}x{h})")
            return None

        faces = face_rec.get(img_bgr)  # expects BGR
        if not faces:
            # no faces found by insightface
            return None

        emb = faces[0].embedding
        if emb is None:
            return None
        return np.array(emb, dtype=np.float32)
    except Exception as e:
        print("get_face_embedding exception:", e)
        return None


def load_registered_face(candidate_name: str, candidate_id: str, attempt_id: int):
    """
    Load registered face for THIS attempt only
    """
    candidate_safe = re.sub(r"[^A-Za-z0-9_]", "_", candidate_id)
    key = f"{candidate_safe}__attempt_{attempt_id}"

    if key in REGISTERED_EMBEDS:
        return REGISTERED_EMBEDS[key]

    fp = SAVE_DIR / f"{key}.png"
    if not fp.exists():
        print(f"🗂 No registered face found: {fp}")
        return None

    img = cv2.imread(str(fp))
    emb = get_face_embedding(img)
    if emb is not None:
        REGISTERED_EMBEDS[key] = emb
        print(f"🗂 Registered face loaded: {fp}")
        return emb

    return None



def compute_similarity(emb1, emb2):
    """Cosine similarity between two embeddings; returns float or None."""
    if emb1 is None or emb2 is None:
        return None
    emb1 = np.array(emb1, dtype=np.float32)
    emb2 = np.array(emb2, dtype=np.float32)
    denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    if denom == 0:
        return None
    return float(np.dot(emb1, emb2) / denom)

def compute_best_similarity(live_emb, reg_emb):
    """
    Supports single embedding OR list of embeddings
    Returns best cosine similarity
    """
    if live_emb is None or reg_emb is None:
        return None

    if isinstance(reg_emb, list):
        sims = [
            compute_similarity(live_emb, e)
            for e in reg_emb
            if e is not None
        ]
        return max(sims) if sims else None

    return compute_similarity(live_emb, reg_emb)


# EAR / pose helpers
def compute_EAR(landmarks, eye_ids):
    """Eye Aspect Ratio using Mediapipe normalized landmarks."""
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_ids])
    dist_vert = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    dist_hori = np.linalg.norm(pts[0] - pts[3])
    if dist_hori == 0:
        return 0.0
    return dist_vert / (2.0 * dist_hori)


def estimate_head_pose(face_landmarks, img_w, img_h):
    """
    Estimate head pose (pitch, yaw, roll) using a small set of facial points.
    Returns pitch, yaw, roll in degrees.
    """
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),              # Nose tip
        (0.0, -63.6, -12.5),          # Chin
        (-43.3, 32.7, -26.0),         # Left eye left corner
        (43.3, 32.7, -26.0),          # Right eye right corner
        (-28.9, -28.9, -24.1),        # Left mouth corner
        (28.9, -28.9, -24.1),         # Right mouth corner
    ])

    image_points = np.array([
        (face_landmarks[1].x * img_w, face_landmarks[1].y * img_h),     # Nose
        (face_landmarks[152].x * img_w, face_landmarks[152].y * img_h), # Chin
        (face_landmarks[263].x * img_w, face_landmarks[263].y * img_h), # Right eye
        (face_landmarks[33].x * img_w, face_landmarks[33].y * img_h),   # Left eye
        (face_landmarks[287].x * img_w, face_landmarks[287].y * img_h), # Right mouth
        (face_landmarks[57].x * img_w, face_landmarks[57].y * img_h),   # Left mouth
    ], dtype=np.float64)

    focal = img_w
    cam_matrix = np.array([
        [focal, 0, img_w / 2],
        [0, focal, img_h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    try:
        success, rot_vec, trans_vec = cv2.solvePnP(
            MODEL_POINTS, image_points, cam_matrix, np.zeros((4,1)),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        rmat, _ = cv2.Rodrigues(rot_vec)
        proj_mat = np.hstack((rmat, trans_vec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)
        pitch, yaw, roll = euler.flatten()
        return float(pitch), float(yaw), float(roll)
    except Exception as e:
        print("estimate_head_pose error:", e)
        return 0.0, 0.0, 0.0


# -----------------------
# Main frame processing
# -----------------------
def process_frame(sess: dict, img_bgr: np.ndarray, candidate_name: str, candidate_id: str):
    """
    Process a single BGR frame, update session state and return anomalies, boxes, etc.
    """
    # Debug header
    print("📸 Frame received for:", candidate_id)
    if img_bgr is None:
        print("❌ img_bgr is None")
        return {"anomalies": [], "faces": 0, "frame": img_bgr, "boxes": []}

    sess["frames_processed"] = sess.get("frames_processed", 0) + 1
    h, w = img_bgr.shape[:2]
    print("🔍 Frame shape:", (w, h))

    anomalies = []
    boxes = []
    detected_faces = []

    # Convert to RGB for Mediapipe
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ---------- Face detection ----------
    det = face_detection.process(rgb)
    if det and det.detections:
        for d in det.detections:
            bb = d.location_data.relative_bounding_box
            x1 = max(0, int(bb.xmin * w))
            y1 = max(0, int(bb.ymin * h))
            x2 = min(w, x1 + int(bb.width * w))
            y2 = min(h, y1 + int(bb.height * h))
            detected_faces.append((x1, y1, x2, y2))
            boxes.append({"top": y1, "left": x1, "bottom": y2, "right": x2})

    # ---------- No face handling ----------
    if len(detected_faces) == 0:
        sess["no_face_count"] += 1
        if sess["no_face_count"] >= 3 and should_trigger(sess, "absence", cooldown=10):
            anomaly = {"type": "absence", "msg": "No face detected."}
            anomalies.append(anomaly)

            # ⭐ FIX: update session counters before return
            sess["anomaly_counts"]["absence"] = sess["anomaly_counts"].get("absence", 0) + 1
            sess["anomalies"].append(anomaly)

        return {
            "anomalies": anomalies,
            "faces": 0,
            "frame": img_bgr,
            "boxes": boxes,
            "anomaly_counts": sess.get("anomaly_counts", {})
        }


    # reset no face counter
    sess["no_face_count"] = 0
    print("🟦 Detected faces:", len(detected_faces), "boxes:", boxes)

    # Use primary face (largest area)
    areas = [( (x2-x1)*(y2-y1), (x1,y1,x2,y2) ) for (x1,y1,x2,y2) in detected_faces]
    areas.sort(reverse=True)
    _, (x1,y1,x2,y2) = areas[0]

    # Multiple faces anomaly
    if len(detected_faces) > 1 and should_trigger(sess, "multi_face", cooldown=30):
        # anomalies.append({"type": "multi_face", "msg": "Multiple faces detected."})
        anomaly = {"type": "multi_face", "msg": "Multiple faces detected."}
        anomalies.append(anomaly)
        sess["anomaly_counts"]["multi_face"] = sess["anomaly_counts"].get("multi_face", 0) + 1
        sess["anomalies"].append(anomaly)


    


    # ---------- Face mesh & liveness (blink + pose) ----------
    mesh = face_mesh.process(rgb)
    pitch = yaw = roll = 0.0
    ear_l = ear_r = EAR = 0.0

    if mesh and mesh.multi_face_landmarks:
        lm = mesh.multi_face_landmarks[0].landmark

        # EAR / blink detection
        try:
            ear_l = compute_EAR(lm, LEFT_EYE)
            ear_r = compute_EAR(lm, RIGHT_EYE)
            EAR = (ear_l + ear_r) / 2.0
        except Exception as e:
            print("EAR compute error:", e)
            EAR = 0.0

        sess.setdefault("ear_history", [])
        sess.setdefault("blink_count", 0)
        sess["ear_history"].append(EAR)
        if len(sess["ear_history"]) > 3:
            sess["ear_history"] = sess["ear_history"][-3:]

        if len(sess["ear_history"]) == 3:
            prev, mid, cur = sess["ear_history"]
            if prev > BLINK_RECOVERY and mid < BLINK_THRESHOLD and cur > BLINK_RECOVERY:
                sess["blink_count"] = sess.get("blink_count", 0) + 1
                print(f"👁 Blink detected (count={sess['blink_count']}) - EAR avg={EAR:.3f}")

        # head pose
        pitch, yaw, roll = estimate_head_pose(lm, w, h)
        print(f"🧭 Head pose pitch={pitch:.1f} yaw={yaw:.1f} roll={roll:.1f}")

    # No blink anomaly after many frames
    if sess.get("blink_count", 0) == 0 and sess["frames_processed"] > BLINK_TRIGGER_FRAMES and should_trigger(sess, "no_blink", cooldown=120):
        anomalies.append({"type": "no_blink", "msg": "No blink detected — possible spoof."})


    # ---------- Face recognition (InsightFace) ----------
    # ---------- Face recognition (InsightFace) ----------
    try:
        faces_full = face_rec.get(img_bgr)
    except Exception as e:
        dbg("❌ insightface get error:", e)
        faces_full = []

    live_emb = None
    if faces_full:
        # use largest detected face
        faces_full.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        live_emb = faces_full[0].embedding

    reg_emb = load_registered_face(candidate_name, candidate_id, sess["attempt_id"])

    dbg("\n================ FACE MATCH DEBUG ================")
    dbg("🧑 Candidate:", candidate_name, "| ID:", candidate_id)
    dbg("📸 Frame:", sess.get("frames_processed"))
    dbg("👤 Live embedding:", "OK" if live_emb is not None else "NONE")
    dbg("🗂 Registered embedding:", "OK" if reg_emb is not None else "NONE")

    if live_emb is not None and reg_emb is not None:
        sim = compute_best_similarity(live_emb, reg_emb)

        dbg(f"🔍 Similarity score: {sim:.4f}" if sim is not None else "🔍 Similarity: NONE")

        # Track similarity history
        sess["face_sim_history"].append(sim)
        if len(sess["face_sim_history"]) > FACE_SIM_HISTORY_SIZE:
            sess["face_sim_history"] = sess["face_sim_history"][-FACE_SIM_HISTORY_SIZE:]

        avg_sim = (
            np.mean([s for s in sess["face_sim_history"] if s is not None])
            if sess["face_sim_history"]
            else None
        )

        dbg("📈 Similarity history (last 5):", sess["face_sim_history"][-5:])
        dbg("📊 Avg similarity:", round(avg_sim, 4) if avg_sim is not None else None)

        # --------------------------------------------------
        # 🟡 INSTANT FACE MISMATCH (COUNTED + UI VISIBLE)
        # --------------------------------------------------
        if (
            sim is not None
            and sim < FACE_MISMATCH_INSTANT_THRESHOLD
            and should_trigger(sess, "face_mismatch", cooldown=FACE_MISMATCH_COOLDOWN)
        ):
            dbg("⚠️ FACE MISMATCH DETECTED (instant)")

            anomaly = {
                "type": "face_mismatch",
                "severity": "warning",
                "msg": "Live face does not match registered face",
                "similarity": float(sim),
            }

            anomalies.append(anomaly)

            # ✅ EXPLICIT COUNT INCREMENT (THIS FIXES UI)
            sess.setdefault("anomaly_counts", {})
            sess["anomaly_counts"]["face_mismatch"] = (
                sess["anomaly_counts"].get("face_mismatch", 0) + 1
            )

            # keep history consistent
            sess.setdefault("anomalies", []).append(anomaly)

            dbg("📊 face_mismatch count →", sess["anomaly_counts"]["face_mismatch"])

        # --------------------------------------------------
        # Temporal mismatch streak
        # --------------------------------------------------
        if sim is not None and sim < FACE_MATCH_SUSPICIOUS:
            sess["face_mismatch_streak"] += 1
            dbg("⚠️ Mismatch streak +1 →", sess["face_mismatch_streak"])
        else:
            if sess["face_mismatch_streak"] > 0:
                dbg("✅ Match recovered, resetting streak")
            sess["face_mismatch_streak"] = 0

        # Liveness signal
        is_live = (
            sess.get("blink_count", 0) > 0
            and sess.get("still_frames", 0) < int(2 * FPS)
        )

        dbg("👁 Blink count:", sess.get("blink_count", 0))
        dbg("🧍 Still frames:", sess.get("still_frames", 0))
        dbg("🧠 Liveness:", is_live)

        dbg(
            "🎯 Thresholds →",
            f"STRONG>{FACE_MATCH_STRONG}",
            f"SUSP<{FACE_MATCH_SUSPICIOUS}",
            f"CRIT<{FACE_MATCH_CRITICAL}",
        )

        # --------------------------------------------------
        # 🔥 CONFIRMED FACE MISMATCH (CRITICAL)
        # --------------------------------------------------
        if (
            sess["face_mismatch_streak"] >= FACE_MISMATCH_CONFIRM_FRAMES
            and sim < FACE_MATCH_CRITICAL
            and is_live
            and should_trigger(sess, "face_mismatch_confirmed", cooldown=120)
        ):
            dbg("🚨🚨 CONFIRMED FACE MISMATCH TRIGGERED 🚨🚨")

            anomalies.append({
                "type": "face_mismatch_confirmed",
                "severity": "critical",
                "msg": "Confirmed face mismatch with liveness detected",
                "similarity": float(sim),
                "duration_sec": FACE_MISMATCH_CONFIRM_SECONDS,
            })

            sess["face_mismatch_streak"] = 0

        # --------------------------------------------------
        # ⚠️ Identity drift
        # --------------------------------------------------
        if (
            len(sess["face_sim_history"]) == FACE_SIM_HISTORY_SIZE
            and sess["frames_processed"] > FACE_SIM_HISTORY_SIZE * 2
            and avg_sim is not None
            and avg_sim < FACE_MATCH_SUSPICIOUS
            and should_trigger(sess, "identity_drift", cooldown=180)
        ):
            dbg("⚠️ IDENTITY DRIFT DETECTED → avg_sim:", avg_sim)

            anomalies.append({
                "type": "identity_drift",
                "msg": "Face identity drift detected over time",
                "avg_similarity": float(avg_sim),
            })

    dbg("==================================================\n")


    # ---------- Gaze-away detection (continuous) ----------
    sess.setdefault("gaze_away_frames", 0)
    looking_away = False
    # Use head pose (pitch/yaw)
    if abs(yaw) > GAZE_AWAY_YAW_THRESHOLD or abs(pitch) > GAZE_AWAY_PITCH_THRESHOLD:
        sess["gaze_away_frames"] += 1
        looking_away = True
    else:
        sess["gaze_away_frames"] = 0

    if sess["gaze_away_frames"] >= GAZE_AWAY_REQUIRED_FRAMES:
        if should_trigger(sess, "gaze_away_long", cooldown=GAZE_AWAY_REQUIRED_SECONDS):
            anomalies.append({
                "type": "gaze_away_long",
                "msg": f"Candidate looked away for {GAZE_AWAY_REQUIRED_SECONDS} seconds continuously.",
                "duration_sec": GAZE_AWAY_REQUIRED_SECONDS,
                "pitch": float(pitch),
                "yaw": float(yaw)
            })
        # Reset counter after reporting
        sess["gaze_away_frames"] = 0

    # ---------- Micro-movement / motion-based anomalies ----------
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    movement_anomalies = []
    prev = sess.get("prev_centroid")
    if prev is not None:
        dx = cx - prev[0]
        dy = cy - prev[1]
        dist = np.sqrt(dx*dx + dy*dy)

        # Update movement histories
        sess["movement_history"].append((dx, dy))
        if len(sess["movement_history"]) > 300:
            sess["movement_history"] = sess["movement_history"][-300:]

        # STILL / STATIC FACE detection
        if dist < MOVEMENT_STILL_THRESHOLD:
            sess["still_frames"] = sess.get("still_frames", 0) + 1
        else:
            sess["still_frames"] = 0

        if sess["still_frames"] >= int(7 * FPS):  # ~7 seconds of stillness
            if should_trigger(sess, "static_face", cooldown=60):
                movement_anomalies.append({
                    "type": "static_face",
                    "msg": "Face too still — possible spoofing."
                })
            sess["still_frames"] = 0

        # NODDING (vertical oscillation) — continuous for NODDING_REQUIRED_FRAMES
        if abs(dy) > abs(dx) and abs(dy) > DY_NOD_THRESHOLD:
            sess["vertical_nod_count"] = sess.get("vertical_nod_count", 0) + 1
        else:
            sess["vertical_nod_count"] = 0

        if sess["vertical_nod_count"] >= NODDING_REQUIRED_FRAMES:
            if should_trigger(sess, "excessive_nodding_long", cooldown=NODDING_REQUIRED_SECONDS):
                movement_anomalies.append({
                    "type": "excessive_nodding_long",
                    "msg": f"Continuous nodding for {NODDING_REQUIRED_SECONDS} seconds detected.",
                    "duration": NODDING_REQUIRED_SECONDS
                })
            sess["vertical_nod_count"] = 0

        # LEFT-RIGHT SCANNING (continuous for SCANNING_REQUIRED_FRAMES)
        if abs(dx) > abs(dy) and abs(dx) > DX_SCAN_THRESHOLD:
            sess["horizontal_scan_count"] = sess.get("horizontal_scan_count", 0) + 1
        else:
            sess["horizontal_scan_count"] = 0

        if sess["horizontal_scan_count"] >= SCANNING_REQUIRED_FRAMES:
            if should_trigger(sess, "head_scanning_long", cooldown=SCANNING_REQUIRED_SECONDS):
                movement_anomalies.append({
                    "type": "head_scanning_long",
                    "msg": f"Continuous left-right scanning for {SCANNING_REQUIRED_SECONDS} seconds detected.",
                    "duration": SCANNING_REQUIRED_SECONDS
                })
            sess["horizontal_scan_count"] = 0

        # STRESS MOVEMENT (erratic) using stddev of recent distances
        sess["last_distances"].append(dist)
        if len(sess["last_distances"]) > int(20 * FPS):
            sess["last_distances"] = sess["last_distances"][-int(20 * FPS):]
            std_dev = float(np.std(sess["last_distances"]))
            if std_dev > STRESS_STDDEV_THRESHOLD:
                if should_trigger(sess, "stress_movement", cooldown=60):
                    movement_anomalies.append({
                        "type": "stress_movement",
                        "msg": "Erratic head movement — possible stress or hesitation."
                    })
                sess["last_distances"] = []

    # Save last centroid
    sess["prev_centroid"] = (cx, cy)

    # ---------- Collect anomalies ----------
    all_anomalies = anomalies + movement_anomalies

    # Update counters and session history (and print debug)
    sess.setdefault("anomaly_counts", {})
    for a in all_anomalies:
        t = a.get("type", "unknown")
        sess["anomaly_counts"][t] = sess["anomaly_counts"].get(t, 0) + 1
        sess["anomalies"].append(a)
        print("⚠ Anomaly triggered:", a)

    # Return structured response
    return {
        "anomalies": all_anomalies,
        "faces": len(detected_faces),
        "frame": img_bgr,
        "boxes": boxes
    }



@router.post("/face-monitor")
async def face_monitor(
    attempt_id: int = Form(None),
    frame: UploadFile = File(None),
    event_type: str = Form(None),
    event_msg: str = Form(None),
):
    """
    Safe face monitoring endpoint — never blocks UI, never throws.
    """
    print(f"\n🟢 [FACE-MONITOR RAW] attempt={attempt_id} event={event_type} frame={'YES' if frame else 'NO'}")

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n🟢 [FACE-MONITOR] {ts} attempt={attempt_id} event={event_type}")

    # -------------------- SAFETY WRAPPER --------------------
    try:
        if not attempt_id:
            print("⚠ Missing attempt_id")
            return {"ok": True}

        # ---------------- DB LOOKUP ----------------
        db = SessionLocal()
        try:
            attempt = db.query(InterviewAttempts).get(attempt_id)
            if not attempt:
                print(f"⚠ Invalid attempt_id {attempt_id}")
                return {"ok": True}

            if attempt.status not in ("IN_PROGRESS", "SCHEDULED"):
                print(f"⚠ Attempt {attempt_id} not active (status={attempt.status})")
                return {"ok": True}

            candidate_id = attempt.candidate_id
            candidate_name = attempt.candidate_id

        finally:
            db.close()

        sess = ensure_session(attempt_id)

        # ---------------- SYSTEM EVENT ----------------
        if event_type:
            print(f"🟡 Event: {event_type} msg={event_msg}")

            anomaly = {
                "type": event_type,
                "msg": event_msg or event_type,
                "timestamp": ts,
            }

            sess.setdefault("anomaly_counts", {})
            sess["anomaly_counts"][event_type] = sess["anomaly_counts"].get(event_type, 0) + 1
            sess.setdefault("anomalies", []).append(anomaly)

            append_to_log({**anomaly, "attempt_id": attempt_id, "candidate": candidate_name})
            save_anomaly_to_transcript(attempt_id, anomaly["msg"])
            save_anomaly_to_db(attempt_id, anomaly)

            return {
                "ok": True,
                "anomalies": [anomaly],
                "anomaly_counts": sess["anomaly_counts"],
                "faces": 0,
                "boxes": [],
                "frame_base64": None,
            }

        # ---------------- FRAME HANDLING ----------------
        if not frame:
            print("⚠ No frame received")
            return {"ok": True}

        content = await frame.read()
        if not content:
            print("⚠ Empty frame")
            return {"ok": True}

        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print("❌ Frame decode error:", e)
            return {"ok": True}

        # ---------------- PROCESS FRAME ----------------
        result = process_frame(sess, img_bgr, candidate_name, candidate_id)

        # ---------------- STORE ANOMALIES ----------------
        for a in result.get("anomalies", []):
            anomaly = {
                "type": a.get("type", "unknown"),
                "msg": a.get("msg", ""),
                "timestamp": ts,
            }

            print(f"🔴 Anomaly detected: {anomaly}")

            append_to_log({**anomaly, "attempt_id": attempt_id, "candidate": candidate_name})
            save_anomaly_to_transcript(attempt_id, anomaly["msg"])
            save_anomaly_to_db(attempt_id, anomaly)

        # ---------------- ENCODE FRAME ----------------
        frame_base64 = None
        try:
            _, buf = cv2.imencode(".jpg", result.get("frame", img_bgr))
            frame_base64 = base64.b64encode(buf).decode("utf-8")
        except Exception:
            pass

        return {
            "ok": True,
            "faces": result.get("faces", 0),
            "boxes": result.get("boxes", []),
            "anomalies": result.get("anomalies", []),
            "anomaly_counts": sess.get("anomaly_counts", {}),
            "frame_base64": frame_base64,
        }

    except Exception as e:
        print("🔥 FACE MONITOR INTERNAL ERROR:", e)
        # Never propagate error
        return {"ok": True}


@router.get("/live")
def get_live_insights(attempt_id: int):
    """
    Fetch live anomalies + counters for a specific interview attempt.
    """
    sess = SESSION_STORE.get(attempt_id)

    if not sess:
        return {
            "attempt_id": attempt_id,
            "anomaly_counts": {},
            "anomalies": [],
            "transcript_tail": [],
        }

    # Load transcript tail
    tail = []
    try:
        with open(transcript_file(attempt_id), "r", encoding="utf-8") as f:
            data = json.load(f)
            tail = data.get("conversation", [])[-5:]
    except Exception:
        pass

    return {
        "attempt_id": attempt_id,
        "anomaly_counts": sess.get("anomaly_counts", {}),
        "anomalies": sess.get("anomalies", [])[-10:],
        "transcript_tail": tail,
    }
