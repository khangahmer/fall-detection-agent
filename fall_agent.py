"""
Fall-Detection Agent (pose-based, no custom training required)
================================================================
Perception (YOLO26 pose keypoints) -> geometric fall heuristic
  -> Reasoning (LLM, tool use) -> Action (email / log)

Why pose instead of a custom "fall" class:
- yolo26n-pose.pt is a stock pretrained model (auto-downloads, no training).
- We compute torso angle from vertical + bbox aspect ratio from the
  17 COCO keypoints. This is an EXPLAINABLE trigger you can defend in a
  demo ("torso was 62 degrees from vertical, that's why it fired"),
  instead of trusting an opaque custom-trained class.
- The geometry only decides WHEN to ask the LLM. The LLM still makes the
  final call (real fall vs false positive) by looking at the actual frame.

Setup:
    pip install -r requirements.txt
    cp .env.example .env   # fill in your keys
"""

import os
import cv2
import json
import base64
import smtplib
import queue
import threading
import numpy as np
from datetime import datetime
from collections import defaultdict
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
GMAIL_SENDER    = os.environ["GMAIL_SENDER"]
GMAIL_APP_PASS  = os.environ["GMAIL_APP_PASS"]
ALERT_RECIPIENT = os.environ["ALERT_RECIPIENT"]
SMTP_PORT       = 587

DEBOUNCE_FRAMES   = 5      # consecutive "probable fall" frames before asking the LLM
PERSON_CONF       = 0.5    # YOLO person-detection confidence gate
ANGLE_THRESHOLD   = 55.0   # degrees from vertical -> torso is "lying down"-ish
ASPECT_THRESHOLD  = 1.3    # bbox width/height -> person is wider than tall
KPT_CONF_MIN      = 0.3    # per-keypoint confidence to trust it
DEBUG             = os.environ.get("DEBUG", "false").lower() == "true"

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")  # "groq" or "anthropic"
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
GROQ_TEXT_MODEL = os.environ.get("GROQ_TEXT_MODEL", "openai/gpt-oss-20b")
CONFIRM_WITH_VISION = os.environ.get("CONFIRM_WITH_VISION", "false").lower() == "true"
LLM_MAX_TOKENS = 200  # tool-call responses are short; keeps you under OTPM limits
TEXT_LLM_MAX_TOKENS = 600  # gpt-oss models "think" before answering — needs more room
                            # or it can burn its budget on reasoning and never emit the tool call

if LLM_PROVIDER == "groq":
    from groq import Groq
    llm_client = Groq(api_key=os.environ["GROQ_API_KEY"])
else:
    import anthropic
    llm_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# COCO 17-keypoint indices
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

# ---------------------------------------------------------------------
# Fall heuristic (pure geometry, no model training)
# ---------------------------------------------------------------------
def torso_angle_from_vertical(kpts):
    """kpts: (17, 3) array of [x, y, conf]. Returns degrees from vertical, or None."""
    ls, rs, lh, rh = kpts[L_SHOULDER], kpts[R_SHOULDER], kpts[L_HIP], kpts[R_HIP]
    if any(p[2] < KPT_CONF_MIN for p in (ls, rs, lh, rh)):
        return None
    shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))  # 0=vertical, 90=horizontal

def is_probable_fall(kpts, bbox):
    x1, y1, x2, y2 = bbox
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    aspect = w / h
    angle = torso_angle_from_vertical(kpts)
    fall_by_angle = angle is not None and angle > ANGLE_THRESHOLD
    fall_by_aspect = aspect > ASPECT_THRESHOLD
    return (fall_by_angle or fall_by_aspect), angle, aspect

# ---------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------
TOOL_SPECS = [
    {
        "name": "send_email_alert",
        "description": "Send an urgent email alert to the on-call contact. Use only when confident this is a genuine fall requiring human attention.",
        "params": {
            "severity": {"type": "string", "enum": ["high", "medium"]},
            "reason": {"type": "string", "description": "Short justification for the decision."}
        },
        "required": ["severity", "reason"]
    },
    {
        "name": "log_incident",
        "description": "Log the event without alerting anyone. Use for likely false positives (sitting, bending, stretching, pets, lying down deliberately) or low-confidence cases.",
        "params": {
            "reason": {"type": "string"}
        },
        "required": ["reason"]
    },
]

def anthropic_tools():
    return [{"name": t["name"], "description": t["description"],
             "input_schema": {"type": "object", "properties": t["params"], "required": t["required"]}}
            for t in TOOL_SPECS]

def openai_style_tools():
    return [{"type": "function", "function": {
                "name": t["name"], "description": t["description"],
                "parameters": {"type": "object", "properties": t["params"], "required": t["required"]}}}
            for t in TOOL_SPECS]

# ---------------------------------------------------------------------
# Action executors
# ---------------------------------------------------------------------
class GmailSender:
    """Connects fresh for each send instead of keeping one long-lived connection —
    emails here are rare (only on flagged falls), so the latency cost of reconnecting
    is irrelevant, and it avoids stale/idle-timeout connections dropping silently."""

    def send(self, track_id, severity, reason, frame):
        msg = MIMEMultipart()
        msg["From"], msg["To"] = GMAIL_SENDER, ALERT_RECIPIENT
        msg["Subject"] = f"[FALL ALERT - {severity.upper()}] Person ID {track_id}"
        msg.attach(MIMEText(f"Reason (from agent): {reason}\nTime: {datetime.now()}", "plain"))
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            att = MIMEImage(buf.tobytes(), _subtype="jpeg")
            att.add_header("Content-Disposition", "attachment", filename=f"fall_{track_id}.jpg")
            msg.attach(att)

        last_err = None
        for attempt in range(2):  # try twice before giving up
            try:
                with smtplib.SMTP("smtp.gmail.com", SMTP_PORT, timeout=15) as server:
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    server.login(GMAIL_SENDER, GMAIL_APP_PASS)
                    server.sendmail(GMAIL_SENDER, ALERT_RECIPIENT, msg.as_string())
                return  # success
            except Exception as e:
                last_err = e
                print(f"[SMTP] attempt {attempt + 1} failed: {type(e).__name__}: {e}")
        raise last_err

gmail = GmailSender()
action_queue = queue.Queue()

def action_worker():
    while True:
        job = action_queue.get()
        if job is None:
            break
        kind, payload = job
        print(f"[WORKER] picked up '{kind}' job for ID {payload.get('track_id')}, processing...")
        try:
            if kind == "email":
                gmail.send(**payload)
                print(f"[ACTION] emailed alert for ID {payload['track_id']} ({payload['severity']})")
            elif kind == "log":
                with open("incident_log.jsonl", "a") as f:
                    f.write(json.dumps(payload) + "\n")
                print(f"[ACTION] logged incident for ID {payload['track_id']}")
        except Exception as e:
            print(f"[ACTION ERROR] {kind} failed for ID {payload.get('track_id')}: {type(e).__name__}: {e}")
        action_queue.task_done()

threading.Thread(target=action_worker, daemon=True).start()

# ---------------------------------------------------------------------
# Reasoning step
# ---------------------------------------------------------------------
def _prompt_text(track_id, angle, aspect):
    angle_str = f"{angle:.1f} deg from vertical" if angle is not None else "unavailable (keypoints unclear)"
    return (f"A pose-based heuristic flagged person track ID {track_id} as a probable fall: "
            f"torso angle = {angle_str} (fall threshold {ANGLE_THRESHOLD} deg), "
            f"bounding-box aspect ratio (width/height) = {aspect:.2f} (fall threshold {ASPECT_THRESHOLD}). "
            f"Look at the actual frame and decide whether this is a genuine fall needing an "
            f"urgent alert, or a false positive (sitting down, bending over, stretching, lying "
            f"down deliberately, etc). Call exactly one tool with your decision.")

def reason_about_fall(track_id, angle, aspect, frame):
    name, args = None, None
    prompt = _prompt_text(track_id, angle, aspect)

    # Default path: TEXT-ONLY reasoning over the heuristic numbers.
    # The geometric heuristic already decided "this looks like a fall" —
    # the LLM's job here is judgment on severity/action, not re-detecting
    # the fall from pixels. No image tokens, much higher rate limits.
    if LLM_PROVIDER == "groq" and not CONFIRM_WITH_VISION:
        try:
            completion = llm_client.chat.completions.create(
                model=GROQ_TEXT_MODEL,
                max_tokens=TEXT_LLM_MAX_TOKENS,
                tools=openai_style_tools(),
                tool_choice="required",
                messages=[{"role": "user", "content": prompt}]
            )
            msg = completion.choices[0].message
            if msg.tool_calls:
                call = msg.tool_calls[0]
                name, args = call.function.name, json.loads(call.function.arguments)
            else:
                print(f"[WARN] ID {track_id}: model returned no tool call. Raw content: {msg.content!r}")
        except Exception as e:
            # Never let an LLM/API hiccup silently drop a flagged fall — fall back
            # to logging it so a human can review, and keep the pipeline running.
            print(f"[LLM ERROR] ID {track_id}: {e}")
            name, args = "log_incident", {"reason": f"LLM call failed ({type(e).__name__}); needs manual review"}

    # Optional path: vision confirmation (set CONFIRM_WITH_VISION=true).
    # Uses more tokens and hits tighter vision-model rate limits, so only
    # turn this on if you want the LLM to actually look at the frame.
    elif LLM_PROVIDER == "groq":
        ok, buf = cv2.imencode(".jpg", frame)
        img_b64 = base64.b64encode(buf.tobytes()).decode()
        try:
            completion = llm_client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                tools=openai_style_tools(),
                tool_choice="required",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }]
            )
            msg = completion.choices[0].message
            if msg.tool_calls:
                call = msg.tool_calls[0]
                name, args = call.function.name, json.loads(call.function.arguments)
            else:
                print(f"[WARN] ID {track_id}: model returned no tool call. Raw content: {msg.content!r}")
        except Exception as e:
            print(f"[LLM ERROR] ID {track_id}: {e}")
            name, args = "log_incident", {"reason": f"LLM call failed ({type(e).__name__}); needs manual review"}

    else:  # anthropic, vision-based
        ok, buf = cv2.imencode(".jpg", frame)
        img_b64 = base64.b64encode(buf.tobytes()).decode()
        response = llm_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=LLM_MAX_TOKENS,
            tools=anthropic_tools(),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        for block in response.content:
            if block.type == "tool_use":
                name, args = block.name, block.input
                break

    if name == "send_email_alert":
        print(f"[DECISION] ID {track_id}: send_email_alert (severity={args['severity']}) — reason: {args['reason']}")
        action_queue.put(("email", {"track_id": track_id, "frame": frame,
                                     "severity": args["severity"], "reason": args["reason"]}))
    elif name == "log_incident":
        print(f"[DECISION] ID {track_id}: log_incident — reason: {args['reason']}")
        action_queue.put(("log", {"track_id": track_id, "reason": args["reason"],
                                   "timestamp": datetime.now().isoformat()}))
    else:
        print(f"[DECISION] ID {track_id}: no action taken (no tool call returned)")
    return name, args

# ---------------------------------------------------------------------
# Perception loop (headless — safe for server/Docker)
# ---------------------------------------------------------------------
def run(model_path="yolo26n-pose.pt", source=0, on_frame=None):
    """on_frame(frame) optional callback — used by the Streamlit demo to stream frames."""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")

    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 25
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    consecutive = defaultdict(int)
    reasoned_ids = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = model.track(frame, verbose=False, conf=PERSON_CONF, classes=[0], persist=True)[0]
            ids = result.boxes.id
            if ids is not None and result.keypoints is not None:
                kpts_all = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
                for box, tid, kpts in zip(result.boxes.xyxy.tolist(), ids, kpts_all):
                    tid = int(tid)
                    x1, y1, x2, y2 = map(int, box)

                    fall, angle, aspect = is_probable_fall(kpts, box)
                    if DEBUG:
                        angle_str = f"{angle:.1f}" if angle is not None else "None (low kpt conf)"
                        print(f"[DEBUG] ID {tid}: angle={angle_str} aspect={aspect:.2f} fall={fall} consecutive={consecutive[tid]}")
                    color = (0, 0, 255) if fall else (186, 0, 221)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    if angle is not None:
                        cv2.putText(frame, f"ID{tid} angle:{angle:.0f}", (x1, max(y1 - 10, 0)),
                                    0, 0.7, color, 2)

                    if fall:
                        consecutive[tid] += 1
                    else:
                        consecutive[tid] = 0

                    if consecutive[tid] >= DEBOUNCE_FRAMES and tid not in reasoned_ids:
                        reasoned_ids.add(tid)
                        print(f"[EVENT] ID {tid} sustained fall pose (angle={angle}, aspect={aspect:.2f}), asking agent...")
                        reason_about_fall(tid, angle, aspect, frame.copy())
            else:
                consecutive.clear()

            out.write(frame)
            if on_frame:
                on_frame(frame)
    finally:
        cap.release()
        out.release()
        action_queue.put(None)

if __name__ == "__main__":
    # source=0 for a live webcam, or a path/URL to a video file / RTSP stream
    run(model_path="yolo26n-pose.pt", source="videos/person-fall-2.mp4")