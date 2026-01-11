# app/db.py

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# --------------------------------------------------------------------
# DATABASE URL
# --------------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://primehire_user_naresh:primehire_user_naresh@localhost/primehirebrain_db"
)

# --------------------------------------------------------------------
# ENGINE
# --------------------------------------------------------------------
engine = create_engine(
    DATABASE_URL,
    echo=False,          # disable verbose SQL logs
    pool_pre_ping=True,  # auto-reconnect dropped connections
)

# --------------------------------------------------------------------
# GLOBAL SHARED BASE
# --------------------------------------------------------------------
Base = declarative_base()

# --------------------------------------------------------------------
# SESSION FACTORY
# --------------------------------------------------------------------
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

# --------------------------------------------------------------------
# UTILITY — dependency injection style (optional)
# --------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
