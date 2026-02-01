from sqlalchemy import Column, Integer, String, DateTime, Boolean
from datetime import datetime, timedelta
from app.db import Base

class CompanyUser(Base):
    __tablename__ = "company_users"

    id = Column(Integer, primary_key=True)
    company_name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    mobile = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)

    is_verified = Column(Boolean, default=False)

    otp_hash = Column(String, nullable=True)
    otp_expires_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
