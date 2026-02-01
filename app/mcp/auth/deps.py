import jwt
from fastapi import Depends, HTTPException, Header
import os

JWT_SECRET = os.getenv("JWT_SECRET")

def get_current_company_id(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token")

    token = authorization.replace("Bearer ", "")

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload["company_id"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token") 
