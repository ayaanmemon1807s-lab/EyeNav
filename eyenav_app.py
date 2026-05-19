import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math

st.set_page_config(page_title="EyeNav", page_icon="👁️", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1 style="text-align:center;">👁️ EyeNav - Eye Controlled Mouse System</h1><p style="text-align:center;">Semester Project</p></div>', unsafe_allow_html=True)

# Initialize MediaPipe
@st.cache_resource
def load_models():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

face_mesh = load_models()

# Eye indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(landmarks, indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    v1 = math.dist(pts[1], pts[5])
    v2 = math.dist(pts[2], pts[4])
    h_dist = math.dist(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h_dist + 1e-6)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    
    # Webcam capture
    camera_image = st.camera_input("Position your face in frame", key="camera")
    
    if camera_image is not None:
        # Convert to OpenCV format
        bytes_data = camera_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.COLOR_RGB2BGR)
        h, w = cv2_img.shape[:2]
        
        rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        gaze_x, gaze_y = 0.5, 0.5
        ear = 0.0
        blink_left = False
        blink_right = False
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            # Calculate EAR
            ear_l = eye_aspect_ratio(lm, LEFT_EYE, w, h)
            ear_r = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear = (ear_l + ear_r) / 2
            
            # Blink detection
            if ear < 0.21:
                blink_left = True
                blink_right = True
            
            # Get gaze
            li_x = sum([lm[i].x for i in LEFT_IRIS]) / 4
            li_y = sum([lm[i].y for i in LEFT_IRIS]) / 4
            ri_x = sum([lm[i].x for i in RIGHT_IRIS]) / 4
            ri_y = sum([lm[i].y for i in RIGHT_IRIS]) / 4
            gaze_x = (li_x + ri_x) / 2
            gaze_y = (li_y + ri_y) / 2
            
            # Draw gaze point
            gx_px = int(gaze_x * w)
            gy_px = int(gaze_y * h)
            cv2.circle(cv2_img, (gx_px, gy_px), 8, (108, 99, 255), -1)
            cv2.circle(cv2_img, (gx_px, gy_px), 10, (0, 229, 199), 2)
            
            # Draw eye landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                cv2_img, results.multi_face_landmarks[0],
                mp.solutions.face_mesh.FACEMESH_IRISES)
            
            # Display metrics
            colm1, colm2, colm3 = st.columns(3)
            with colm1:
                st.metric("Gaze X", f"{gaze_x:.2f}")
            with colm2:
                st.metric("Gaze Y", f"{gaze_y:.2f}")
            with colm3:
                st.metric("EAR", f"{ear:.3f}")
            
            # Gesture indicators
            st.subheader("🎮 Gesture Controls")
            gcol1, gcol2, gcol3 = st.columns(3)
            with gcol1:
                if blink_left:
                    st.success("😉 LEFT WINK - Left Click!")
                else:
                    st.info("😉 Left Wink")
            with gcol2:
                if blink_right:
                    st.success("😜 RIGHT WINK - Right Click!")
                else:
                    st.info("😜 Right Wink")
            with gcol3:
                st.info("😑 Both Eyes - Pause")
        
        # Display processed image
        st.image(cv2_img, channels="BGR", use_container_width=True)

with col2:
    st.subheader("🎮 Control Panel")
    
    st.subheader("📖 Instructions")
    st.markdown("""
    **Step-by-Step Guide:**
    
    1. ✅ Allow camera access
    2. ✅ Position your face in frame
    3. 👀 Move eyes to control cursor
    4. 😉 Left wink = Left Click
    5. 😜 Right wink = Right Click
    6. 😑 Both eyes = Pause
    """)
    
    st.subheader("📊 Metrics Explained")
    st.info("""
    - **Gaze X/Y**: Eye position (0-1 range)
    - **EAR**: Eye Aspect Ratio (lower = blink)
    - < 0.21 = Blink detected
    """)
    
    st.subheader("📧 Project Info")
    st.info("""
    **EyeNav - Eye Controlled Mouse System**
    - Computer Vision based HCI
    - Real-time eye tracking
    - Blink detection for clicks
    - Semester Project
    """)

st.markdown("---")
st.markdown("<p style='text-align:center;'>EyeNav - Semester Project | Eye Controlled Mouse System</p>", unsafe_allow_html=True)
