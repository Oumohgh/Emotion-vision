import os
import time
import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace

# Suppress TensorFlow logging in terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(
    page_title="Face Cadre & Feeling Analytics", 
    page_icon="🖼️", 
    layout="wide"
)

st.title("🖼️ Face Cadre & Real-Time Emotion Analyzer")

# Sidebar Controls
run_tracker = st.sidebar.checkbox("Start Camera", value=False)
box_color_hex = st.sidebar.color_picker("Cadre Frame Color", "#00FF00")

# Convert Hex Color to BGR
def hex_to_bgr(hex_str):
    hex_str = hex_str.lstrip('#')
    r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)

bgr_color = hex_to_bgr(box_color_hex)

# Layout Columns
col_cam, col_stats = st.columns([2, 1])

with col_cam:
    image_placeholder = st.empty()

with col_stats:
    st.subheader("📊 Detected Feeling Score")
    stats_placeholder = st.empty()


def draw_styled_cadre(img, x, y, w, h, color, label=""):
    """Draws a main box + stylized corner brackets (cadre) around the face."""
    # Main bounding rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # Stylized corner brackets
    line_len = int(min(w, h) * 0.2)
    thickness = 4

    # Top-Left
    cv2.line(img, (x, y), (x + line_len, y), color, thickness)
    cv2.line(img, (x, y), (x, y + line_len), color, thickness)
    # Top-Right
    cv2.line(img, (x + w, y), (x + w - line_len, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + line_len), color, thickness)
    # Bottom-Left
    cv2.line(img, (x, y + h), (x + line_len, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - line_len), color, thickness)
    # Bottom-Right
    cv2.line(img, (x + w, y + h), (x + w - line_len, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - line_len), color, thickness)

    # Label Banner
    if label:
        banner_y = y - 30 if y - 30 > 0 else y
        cv2.rectangle(img, (x, banner_y), (x + w, y), color, -1)
        cv2.putText(
            img,
            label,
            (x + 5, y - 8 if y - 30 > 0 else y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )


def draw_no_face_warning(img):
    """Draws a 'NO FACE DETECTED' overlay banner at the top of the video feed."""
    h, w, _ = img.shape
    text = "NO FACE DETECTED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # Get text width/height for centering
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    # Top banner position
    margin = 15
    rect_x1, rect_y1 = (w - text_w) // 2 - margin, 15
    rect_x2, rect_y2 = (w + text_w) // 2 + margin, 15 + text_h + margin

    # Draw semi-transparent background box & red border
    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 200), -1)
    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), 2)

    # Draw white text
    text_x = (w - text_w) // 2
    text_y = rect_y1 + text_h + 5
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)


if run_tracker:
    cap = cv2.VideoCapture(0)
    last_results = []
    frame_count = 0

    try:
        while run_tracker:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot access camera.")
                break

            frame_count += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Analyze face cadre & emotions every 5 frames
            if frame_count % 5 == 0 or len(last_results) == 0:
                try:
                    analyses = DeepFace.analyze(
                        img_path=rgb_frame,
                        actions=['emotion'],
                        detector_backend='fastmtcnn',
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if isinstance(analyses, dict):
                        analyses = [analyses]
                        
                    last_results = analyses
                except Exception:
                    last_results = []

            # Filter valid faces that have non-zero dimensions
            valid_faces = [
                res for res in last_results 
                if res.get('region', {}).get('w', 0) > 0 and res.get('region', {}).get('h', 0) > 0
            ]

            # Render UI updates
            with stats_placeholder.container():
                if valid_faces:
                    for idx, res in enumerate(valid_faces):
                        region = res.get('region', {})
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        emotions = res.get('emotion', {})
                        dominant_emotion = res.get('dominant_emotion', 'UNKNOWN').upper()

                        # Draw cadre frame on detected face
                        draw_styled_cadre(rgb_frame, x, y, w, h, bgr_color, label=dominant_emotion)

                        # Display emotion progress breakdown in side column
                        st.write(f"**Face #{idx + 1} Cadre:**")
                        for emotion_name, score in sorted(emotions.items(), key=lambda item: item[1], reverse=True):
                            st.write(f"{emotion_name.capitalize()}: `{score:.1f}%`")
                            st.progress(min(int(score), 100))
                else:
                    # Draw NO FACE overlay on the video feed
                    draw_no_face_warning(rgb_frame)
                    st.warning("⚠️ No face detected in camera view.")

            # Render live frame
            image_placeholder.image(rgb_frame, channels="RGB", width="stretch")

            # Prevents WebSocket socket crash (WinError 10054)
            time.sleep(0.03)

    finally:
        cap.release()
else:
    image_placeholder.info("Check 'Start Camera' in the sidebar to launch the app.")