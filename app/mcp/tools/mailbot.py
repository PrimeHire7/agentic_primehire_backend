import smtplib
from email.message import EmailMessage
from fastapi import APIRouter, HTTPException, UploadFile, Form


@router.post("/automate_mailbot")
async def mailmind_send(
    to_email: str = Form(...),
    subject: str = Form(...),
    body: str = Form(...),
    attachment: UploadFile | None = None,
):
    creds = MailCreds(
        email="naresh@primehire.ai",
        password="Techdeveloper$",
        platform="godaddy",
        imap_server="imap.secureserver.net",
    )

    smtp_server = PLATFORM_SMTP.get(creds.platform.lower())
    if not smtp_server:
        raise HTTPException(status_code=400, detail="Invalid SMTP platform")

    try:
        msg = EmailMessage()
        msg["From"] = creds.email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)

        if attachment:
            file_data = await attachment.read()
            msg.add_attachment(
                file_data,
                maintype="application",
                subtype="octet-stream",
                filename=attachment.filename,
            )

        # Connect and send email
        with smtplib.SMTP_SSL(smtp_server, 465) as smtp:
            smtp.login(creds.email, creds.password)
            smtp.send_message(msg)

        return {"message": f"Email successfully sent to {to_email}!"}

    except smtplib.SMTPAuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid SMTP credentials")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Mail send failed: {e}")
