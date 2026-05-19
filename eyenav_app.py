"""
EyeNav Backend Server
=====================
Runs two servers simultaneously:
  - Flask  on port 5000  → MJPEG video stream  (http://localhost:5000/video)
  - WebSocket on port 8765 → real-time JSON data  (ws://localhost:8765)

Install dependencies:
    pip install opencv-python mediapipe flask websockets pyautogui numpy

Run:
    python eyenav_server.py
"""

import asyncio
import json
import math
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from flask import Flask, Response
import websockets

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
WS_HOST         = "localhost"
WS_PORT         = 8765
FLASK_PORT      = 5000
CAM_INDEX       = 0
TARGET_FPS      = 30
BLINK_EAR_THRESH = 0.21      # Eye Aspect Ratio threshold for blink detection
BLINK_CONSEC    = 2          # consecutive frames below threshold = blink
SCREEN_W, SCREEN_H = pyautogui.size()

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0          # no delay between pyautogui calls

# ─────────────────────────────────────────────
# SHARED STATE  (thread-safe via lock)
# ─────────────────────────────────────────────
state_lock = threading.Lock()
shared = {
    # tracking data
    "gaze_x":      0.5,      # 0.0–1.0 normalised screen fraction
    "gaze_y":      0.5,
    "ear_left":    0.0,
    "ear_right":   0.0,
    "ear_avg":     0.0,
    "blink_left":  False,
    "blink_right": False,
    "blink_both":  False,
    "fps":         0,
    "confidence":  0.0,
    "tracking":    False,
    # latest annotated JPEG bytes for MJPEG stream
    "frame_bytes": b"",
    # settings (written by WebSocket when UI sends control msgs)
    "sensitivity":     2.5,
    "blink_click":     True,
    "dwell_click":     False,
    "smooth_tracking": True,
    "paused":          False,
}

# connected WebSocket clients
ws_clients: set = set()


# ─────────────────────────────────────────────
# MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# MediaPipe landmark indices for left / right eye
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Iris landmarks (refined)
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices]
    # vertical distances
    v1 = math.dist(pts[1], pts[5])
    v2 = math.dist(pts[2], pts[4])
    # horizontal distance
    h  = math.dist(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def iris_centre(landmarks, iris_indices, img_w, img_h):
    xs = [landmarks[i].x * img_w for i in iris_indices]
    ys = [landmarks[i].y * img_h for i in iris_indices]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def smooth(prev, curr, alpha=0.25):
    """Exponential moving average."""
    return prev * (1 - alpha) + curr * alpha


# ─────────────────────────────────────────────
# CALIBRATION  (simple 5-point)
# ─────────────────────────────────────────────
class Calibration:
    """
    Collects raw iris positions at known screen points and fits a simple
    affine transform (least-squares) to map iris → screen coords.
    """
    def __init__(self):
        self.src_pts = []   # raw iris positions
        self.dst_pts = []   # corresponding screen positions
        self.M       = None  # 2×3 affine matrix

    def add_point(self, iris_x, iris_y, screen_x, screen_y):
        self.src_pts.append([iris_x, iris_y])
        self.dst_pts.append([screen_x, screen_y])

    def fit(self):
        if len(self.src_pts) < 4:
            return False
        src = np.array(self.src_pts, dtype=np.float32)
        dst = np.array(self.dst_pts, dtype=np.float32)
        self.M, _ = cv2.estimateAffinePartial2D(src, dst)
        return self.M is not None

    def map(self, iris_x, iris_y):
        if self.M is None:
            return iris_x, iris_y
        pt = np.array([[[iris_x, iris_y]]], dtype=np.float32)
        result = cv2.transform(pt, self.M)
        return float(result[0][0][0]), float(result[0][0][1])

    def reset(self):
        self.src_pts.clear()
        self.dst_pts.clear()
        self.M = None


calibration = Calibration()


# ─────────────────────────────────────────────
# TRACKING THREAD
# ─────────────────────────────────────────────
def tracking_thread():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    blink_counter_l = 0
    blink_counter_r = 0
    prev_gx = 0.5
    prev_gy = 0.5
    fps_counter = 0
    fps_timer   = time.time()
    current_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # FPS measurement
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_timer   = time.time()

        img_h, img_w = frame.shape[:2]

        # --- MediaPipe inference ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        detected   = results.multi_face_landmarks is not None
        confidence = 0.0
        gx, gy     = prev_gx, prev_gy
        ear_l = ear_r = ear_avg = 0.0
        blink_l = blink_r = blink_both = False

        if detected:
            lms = results.multi_face_landmarks[0].landmark
            confidence = 0.95

            # EAR
            ear_l = eye_aspect_ratio(lms, LEFT_EYE,  img_w, img_h)
            ear_r = eye_aspect_ratio(lms, RIGHT_EYE, img_w, img_h)
            ear_avg = (ear_l + ear_r) / 2.0

            # Blink detection
            if ear_l < BLINK_EAR_THRESH:
                blink_counter_l += 1
            else:
                if blink_counter_l >= BLINK_CONSEC:
                    blink_l = True
                blink_counter_l = 0

            if ear_r < BLINK_EAR_THRESH:
                blink_counter_r += 1
            else:
                if blink_counter_r >= BLINK_CONSEC:
                    blink_r = True
                blink_counter_r = 0

            blink_both = blink_l and blink_r

            # Iris gaze estimation
            li_x, li_y = iris_centre(lms, LEFT_IRIS,  img_w, img_h)
            ri_x, ri_y = iris_centre(lms, RIGHT_IRIS, img_w, img_h)
            avg_iris_x = (li_x + ri_x) / 2.0
            avg_iris_y = (li_y + ri_y) / 2.0

            # Map to screen via calibration (or fallback normalise)
            if calibration.M is not None:
                sx, sy = calibration.map(avg_iris_x, avg_iris_y)
            else:
                # simple normalise without calibration
                sx = avg_iris_x / img_w * SCREEN_W
                sy = avg_iris_y / img_h * SCREEN_H

            gx = sx / SCREEN_W
            gy = sy / SCREEN_H

            # Smooth
            with state_lock:
                alpha = 0.25 if shared["smooth_tracking"] else 1.0
            gx = smooth(prev_gx, gx, alpha)
            gy = smooth(prev_gy, gy, alpha)
            prev_gx, prev_gy = gx, gy

            # Draw landmarks on frame
            mp_drawing.draw_landmarks(
                frame,
                results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_IRISES,
                None,
                mp_styles.get_default_face_mesh_iris_connections_style()
            )

            # Draw gaze point
            gx_px = int(gx * img_w)
            gy_px = int(gy * img_h)
            cv2.circle(frame, (gx_px, gy_px), 8, (108, 99, 255), -1)
            cv2.circle(frame, (gx_px, gy_px), 10, (0, 229, 199),  2)

            # --- Cursor movement & click actions ---
            with state_lock:
                is_paused = shared["paused"] or blink_both
                sens      = shared["sensitivity"]
                do_blink  = shared["blink_click"]

            if not is_paused:
                # Move cursor (sensitivity scales deviation from centre)
                cx = int(SCREEN_W  * gx)
                cy = int(SCREEN_H  * gy)
                pyautogui.moveTo(cx, cy)

                if do_blink:
                    if blink_l and not blink_r:
                        pyautogui.click()
                    elif blink_r and not blink_l:
                        pyautogui.rightClick()

        # Encode annotated frame as JPEG
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buf.tobytes()

        # Write to shared state
        with state_lock:
            shared.update({
                "gaze_x":     round(gx, 4),
                "gaze_y":     round(gy, 4),
                "ear_left":   round(ear_l, 3),
                "ear_right":  round(ear_r, 3),
                "ear_avg":    round(ear_avg, 3),
                "blink_left":  blink_l,
                "blink_right": blink_r,
                "blink_both":  blink_both,
                "fps":         current_fps,
                "confidence":  round(confidence, 2),
                "tracking":    detected,
                "frame_bytes": frame_bytes,
                "paused":      shared["paused"] or blink_both,
            })
            # Reset pause flag after both-eyes blink handled
            if blink_both:
                shared["paused"] = not shared["paused"]

        time.sleep(max(0, 1 / TARGET_FPS - 0.002))

    cap.release()


# ─────────────────────────────────────────────
# WEBSOCKET SERVER
# ─────────────────────────────────────────────
async def ws_handler(websocket):
    ws_clients.add(websocket)
    print(f"[WS] Client connected. Total: {len(ws_clients)}")
    try:
        async for message in websocket:
            # Receive settings / control messages from the UI
            try:
                msg = json.loads(message)
                if msg.get("type") == "settings":
                    with state_lock:
                        if "sensitivity"     in msg: shared["sensitivity"]     = float(msg["sensitivity"])
                        if "blink_click"     in msg: shared["blink_click"]     = bool(msg["blink_click"])
                        if "dwell_click"     in msg: shared["dwell_click"]     = bool(msg["dwell_click"])
                        if "smooth_tracking" in msg: shared["smooth_tracking"] = bool(msg["smooth_tracking"])
                elif msg.get("type") == "calibrate":
                    calibration.reset()
                    print("[CAL] Calibration reset, starting new session")
                elif msg.get("type") == "calibrate_point":
                    # UI sends: { type: "calibrate_point", iris_x, iris_y, screen_x, screen_y }
                    with state_lock:
                        iris_x = shared.get("_raw_iris_x", 320)
                        iris_y = shared.get("_raw_iris_y", 240)
                    calibration.add_point(iris_x, iris_y,
                                          msg["screen_x"], msg["screen_y"])
                    if len(calibration.src_pts) >= 5:
                        ok = calibration.fit()
                        print(f"[CAL] Fitted: {ok}")
            except Exception as e:
                print(f"[WS] Bad message: {e}")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        ws_clients.discard(websocket)
        print(f"[WS] Client disconnected. Total: {len(ws_clients)}")


async def broadcast_loop():
    """Push tracking data to all connected clients at TARGET_FPS."""
    while True:
        if ws_clients:
            with state_lock:
                payload = {
                    "type":       "tracking",
                    "gaze_x":     shared["gaze_x"],
                    "gaze_y":     shared["gaze_y"],
                    "ear":        shared["ear_avg"],
                    "blink_left":  shared["blink_left"],
                    "blink_right": shared["blink_right"],
                    "blink_both":  shared["blink_both"],
                    "fps":         shared["fps"],
                    "confidence":  shared["confidence"],
                    "tracking":    shared["tracking"],
                    "paused":      shared["paused"],
                }
            msg = json.dumps(payload)
            dead = set()
            for ws in list(ws_clients):
                try:
                    await ws.send(msg)
                except Exception:
                    dead.add(ws)
            ws_clients -= dead
        await asyncio.sleep(1 / TARGET_FPS)


async def ws_main():
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        print(f"[WS] Listening on ws://{WS_HOST}:{WS_PORT}")
        await broadcast_loop()


def ws_thread():
    asyncio.run(ws_main())


# ─────────────────────────────────────────────
# FLASK MJPEG STREAM
# ─────────────────────────────────────────────
flask_app = Flask(__name__)

def generate_mjpeg():
    while True:
        with state_lock:
            frame = shared["frame_bytes"]
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame +
                b"\r\n"
            )
        time.sleep(1 / TARGET_FPS)


@flask_app.route("/video")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@flask_app.route("/")
def index():
    return "<h2>EyeNav server running.<br>Video: <a href='/video'>/video</a></h2>"


def flask_thread():
    flask_app.run(host="0.0.0.0", port=FLASK_PORT, threaded=True, use_reloader=False)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  EyeNav Server")
    print(f"  Video stream : http://localhost:{FLASK_PORT}/video")
    print(f"  WebSocket    : ws://localhost:{WS_PORT}")
    print("=" * 50)

    # Start tracking in background thread
    t_track = threading.Thread(target=tracking_thread, daemon=True)
    t_track.start()

    # Start Flask in background thread
    t_flask = threading.Thread(target=flask_thread, daemon=True)
    t_flask.start()

    # Run WebSocket in main thread (asyncio event loop)
    ws_thread()
