import os
import smtplib
from email.message import EmailMessage
import logging

logger = logging.getLogger("mail")

def send_mail(
    to_email: str,
    subject: str,
    body: str
):
    SMTP_HOST = os.getenv("SMTP_HOST", "smtpout.secureserver.net")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASS = os.getenv("SMTP_PASS")

    if not SMTP_USER or not SMTP_PASS:
        logger.error("SMTP credentials missing")
        raise RuntimeError("SMTP not configured")

    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
        smtp.login(SMTP_USER, SMTP_PASS)
        smtp.send_message(msg)

    logger.info(f"Mail sent to {to_email}")

