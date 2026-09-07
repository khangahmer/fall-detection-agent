"""
Standalone email test — isolates the SMTP problem from the whole video pipeline.
Run: python test_email.py
Should finish in a few seconds if your network/credentials are fine.
"""
import os
import time
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

GMAIL_SENDER    = os.environ["GMAIL_SENDER"]
GMAIL_APP_PASS  = os.environ["GMAIL_APP_PASS"]
ALERT_RECIPIENT = os.environ["ALERT_RECIPIENT"]

print(f"Sender:    {GMAIL_SENDER}")
print(f"Recipient: {ALERT_RECIPIENT}")
print(f"App pass length: {len(GMAIL_APP_PASS)} chars (should be 16 or 19 with spaces)")
print("Connecting to smtp.gmail.com:587 ...")

start = time.time()
try:
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
        print(f"[{time.time()-start:.1f}s] TCP connected, starting TLS...")
        server.ehlo()
        server.starttls()
        server.ehlo()
        print(f"[{time.time()-start:.1f}s] TLS up, logging in...")
        server.login(GMAIL_SENDER, GMAIL_APP_PASS)
        print(f"[{time.time()-start:.1f}s] Login OK, sending test email...")
        msg = MIMEText("This is a test email from the fall-detection agent setup.")
        msg["From"], msg["To"], msg["Subject"] = GMAIL_SENDER, ALERT_RECIPIENT, "Test email - fall detection agent"
        server.sendmail(GMAIL_SENDER, ALERT_RECIPIENT, msg.as_string())
        print(f"[{time.time()-start:.1f}s] SUCCESS — check {ALERT_RECIPIENT}'s inbox (and spam folder).")
except Exception as e:
    print(f"[{time.time()-start:.1f}s] FAILED: {type(e).__name__}: {e}")
    print("\nIf this took ~15s to fail with a timeout, your network is likely blocking outbound SMTP.")
    print("Try again on a mobile hotspot to confirm.")