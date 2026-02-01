import os
import random
import bcrypt
import jwt
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from app.db import SessionLocal
from app.mcp.tools.company_auth_model import CompanyUser
from app.mcp.utils.mail import send_mail


JWT_SECRET = os.getenv("JWT_SECRET")

router = APIRouter()

# ------------------------
# Request Models
# ------------------------
class SignupRequest(BaseModel):
    company_name: str
    email: EmailStr
    mobile: str
    password: str

class OTPVerifyRequest(BaseModel):
    email: EmailStr
    otp: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# ------------------------
# Signup → Send OTP
# ------------------------
@router.post("/auth/signup")
async def signup(payload: SignupRequest):
    db = SessionLocal()
    try:
        print("🔥🔥🔥 NEW company_auth.py LOADED 🔥🔥🔥")
        existing_email = db.query(CompanyUser).filter(
            CompanyUser.email == payload.email
        ).first()

        existing_mobile = db.query(CompanyUser).filter(
            CompanyUser.mobile == payload.mobile
        ).first()

        if existing_email and existing_mobile:
            raise HTTPException(
                status_code=400,
                detail="Email and mobile already exist"
            )

        if existing_email:
            raise HTTPException(
                status_code=400,
                detail="Email already exists"
            )

        if existing_mobile:
            raise HTTPException(
                status_code=400,
                detail="Mobile already exists"
            )


        otp = str(random.randint(100000, 999999))
        otp_hash = bcrypt.hashpw(otp.encode(), bcrypt.gensalt()).decode()
        expires = datetime.utcnow() + timedelta(minutes=10)

        if not existing:
            user = CompanyUser(
                company_name=payload.company_name,
                email=payload.email,
                mobile=payload.mobile,
                password_hash=bcrypt.hashpw(payload.password.encode(), bcrypt.gensalt()).decode(),
                otp_hash=otp_hash,
                otp_expires_at=expires,
            )
            db.add(user)
        else:
            existing.otp_hash = otp_hash
            existing.otp_expires_at = expires

        db.commit()
        print("🔥 NEW COMPANY_AUTH CODE LOADED")
        #DO NOT await
        send_mail(
            to_email=payload.email,
            subject="PrimeHire Verification Code",
            body=f"Your PrimeHire OTP is {otp}. Valid for 10 minutes."
        )

        return {"ok": True}

    finally:
        db.close()


# ------------------------
# Verify OTP
# ------------------------
@router.post("/auth/verify-otp") 
async def verify_otp(payload: OTPVerifyRequest):
    db = SessionLocal()
    try:
        user = db.query(CompanyUser).filter_by(email=payload.email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if not user.otp_hash or datetime.utcnow() > user.otp_expires_at:
            raise HTTPException(status_code=400, detail="OTP expired")

        if not bcrypt.checkpw(payload.otp.encode(), user.otp_hash.encode()):
            raise HTTPException(status_code=400, detail="Invalid OTP")

        user.is_verified = True
        user.otp_hash = None
        user.otp_expires_at = None
        db.commit()

        return {"ok": True}

    finally:
        db.close()

# ------------------------
# Login
# ------------------------
@router.post("/auth/login")
async def login(payload: LoginRequest):
    db = SessionLocal()
    try:
        user = db.query(CompanyUser).filter_by(email=payload.email).first()
        if not user or not user.is_verified:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        if not bcrypt.checkpw(payload.password.encode(), user.password_hash.encode()):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = jwt.encode(
            {"user_id": user.id, "email": user.email,"company_id": user.id, "exp": datetime.utcnow() + timedelta(days=1)},
            JWT_SECRET,
            algorithm="HS256",
        )

        return {"ok": True, "token": token}

    finally:
        db.close()
