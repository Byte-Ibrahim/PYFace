import cv2
import face_recognition
import numpy as np
import time
from collections import deque

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SCALE          = 0.25          # Downscale factor for detection (1/4 size)
PROCESS_EVERY  = 2             # Only run face_recognition every N frames
SHOW_FPS       = True          # Overlay FPS counter
MESH_COLOR     = (255, 255, 255)
DOT_COLOR      = (0, 255, 0)
DOT_RADIUS     = 2
LINE_THICKNESS = 1

# Features that form closed loops
CLOSED_FEATURES = frozenset({'left_eye', 'right_eye', 'top_lip', 'bottom_lip'})
# ──────────────────────────────────────────────────────────────────────────────


def build_overlay(frame: np.ndarray, face_landmarks_list: list) -> None:
    """
    Draw facial mesh landmarks directly onto `frame` (in-place).
    Scales coordinates from the 1/4-size detection space back to full resolution.
    """
    scale_inv = int(1 / SCALE)   # = 4 — avoids repeated float division in loop

    for face_landmarks in face_landmarks_list:
        for feature, points in face_landmarks.items():
            # Vectorised coordinate scaling — no Python-level loop
            pts = np.array(points, dtype=np.int32)
            pts *= scale_inv                        # broadcast multiply: fast
            pts_poly = pts.reshape((-1, 1, 2))

            is_closed = feature in CLOSED_FEATURES
            cv2.polylines(frame, [pts_poly], is_closed, MESH_COLOR, LINE_THICKNESS)

            # Draw dots — still a loop but unavoidable with cv2.circle
            for pt in pts:
                cv2.circle(frame, (pt[0], pt[1]), DOT_RADIUS, DOT_COLOR, -1)


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    # Favour MJPG encoding for higher USB bandwidth efficiency
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    print("Optimized Facial Mesh — Press 'q' to quit.")

    # State carried across iterations
    frame_idx          = 0
    cached_landmarks   = []          # Last computed landmarks, reused on skipped frames
    fps_times: deque   = deque(maxlen=30)   # Rolling window for FPS calculation

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame — retrying…")
            continue

        # ── DETECTION (throttled) ────────────────────────────────────────────
        if frame_idx % PROCESS_EVERY == 0:
            # Downscale once; reuse for both colour conversion and detection
            small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)

            # face_recognition expects RGB; cvtColor is faster than slicing
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            cached_landmarks = face_recognition.face_landmarks(rgb_small)

        # ── DRAW ─────────────────────────────────────────────────────────────
        build_overlay(frame, cached_landmarks)

        # ── FPS OVERLAY ──────────────────────────────────────────────────────
        if SHOW_FPS:
            fps_times.append(time.perf_counter())
            if len(fps_times) >= 2:
                fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
                cv2.putText(
                    frame, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                    cv2.LINE_AA
                )

        cv2.imshow('Optimized Facial Mesh', frame)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()