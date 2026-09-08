"""
Streamlit demo for the pose-based fall-detection agent.
Run: streamlit run app_streamlit.py
"""
import streamlit as st
import tempfile
import cv2
import fall_agent as agent

st.set_page_config(page_title="Fall Detection Agent", layout="wide")
st.title("🚨 Fall Detection Agent")
st.caption("YOLO26 pose keypoints + geometric heuristic + LLM reasoning (tool use) + email/log action")

model_path = st.text_input("Model path", "yolo26n-pose.pt")

frame_slot = st.empty()
status_slot = st.empty()

def show(frame):
    frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded and st.button("Run agent"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded.read())
    tmp.close()
    status_slot.info("Running... open 'Manage app' in the bottom right to watch live logs.")
    agent.run(model_path=model_path, source=tmp.name, on_frame=show)
    status_slot.success("Done. Check incident_log.jsonl and your inbox.")
