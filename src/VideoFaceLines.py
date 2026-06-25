# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
import numpy as np
import time
import urllib.request
import os
from collections import deque
# pyrefly: ignore [missing-import]
from mediapipe.tasks import python as mp_python
# pyrefly: ignore [missing-import]
from mediapipe.tasks.python import vision as mp_vision

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SHOW_FPS      = True
SNAPSHOT_DIR  = "snapshots"   # Folder where cropped face images are saved
SNAPSHOT_SIZE = 256           # Output square size in pixels (256×256)
PADDING       = 0.30          # Extra padding around face bounding box (30%)
MODEL_PATH    = "face_landmarker.task"
MODEL_URL     = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# BGR colours
MESH_COLOR    = (180, 180, 180)
CONTOUR_COLOR = (0, 200, 255)
DOT_COLOR     = (0, 255, 0)

# FaceLandmarksConnections — contours only (no dense tessellation)
CONTOURS  = mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
LEFT_EYE  = mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
RIGHT_EYE = mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE
LIPS      = mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS

# Sparse set of key anatomical landmarks to draw as dots (~20 points)
KEY_LANDMARKS = [
    1,    # nose tip
    4,    # nose bridge
    33,   # left eye outer corner
    133,  # left eye inner corner
    362,  # right eye outer corner
    263,  # right eye inner corner
    61,   # left mouth corner
    291,  # right mouth corner
    17,   # chin bottom
    152,  # chin centre
    10,   # forehead top
    234,  # left cheek
    454,  # right cheek
    70,   # left brow outer
    107,  # left brow inner
    336,  # right brow inner
    300,  # right brow outer
    13,   # upper lip centre
    14,   # lower lip centre
]
# ──────────────────────────────────────────────────────────────────────────────


def download_model() -> None:
    if os.path.exists(MODEL_PATH):
        return
    print(f"Downloading face landmarker model → {MODEL_PATH} …")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def draw_connections(frame: np.ndarray, lms, connections, color, thickness=1) -> None:
    h, w = frame.shape[:2]
    for conn in connections:
        s, e = conn.start, conn.end
        x1, y1 = int(lms[s].x * w), int(lms[s].y * h)
        x2, y2 = int(lms[e].x * w), int(lms[e].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def draw_dots(frame: np.ndarray, lms, color, radius=1) -> None:
    h, w = frame.shape[:2]
    for lm in lms:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)


def draw_key_dots(frame: np.ndarray, lms, color, radius=3) -> None:
    """Draw dots only at sparse key anatomical landmarks."""
    h, w = frame.shape[:2]
    for idx in KEY_LANDMARKS:
        if idx < len(lms):
            cx, cy = int(lms[idx].x * w), int(lms[idx].y * h)
            cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)


def draw_face(frame: np.ndarray, lms) -> None:
    draw_connections(frame, lms, CONTOURS,  CONTOUR_COLOR, thickness=1)
    draw_connections(frame, lms, LEFT_EYE,  CONTOUR_COLOR, thickness=1)
    draw_connections(frame, lms, RIGHT_EYE, CONTOUR_COLOR, thickness=1)
    draw_connections(frame, lms, LIPS,      CONTOUR_COLOR, thickness=1)
    draw_key_dots(frame, lms, DOT_COLOR, radius=3)


def face_bounding_box(lms, frame_h: int, frame_w: int, padding: float = PADDING):
    """Returns a tight square (x0, y0, x1, y1) around all face landmarks."""
    xs = [lm.x * frame_w for lm in lms]
    ys = [lm.y * frame_h for lm in lms]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = (x_max - x_min) * padding
    pad_y = (y_max - y_min) * padding
    x_min -= pad_x;  x_max += pad_x
    y_min -= pad_y;  y_max += pad_y
    size = max(x_max - x_min, y_max - y_min)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    x0 = max(0, int(cx - size / 2))
    y0 = max(0, int(cy - size / 2))
    x1 = min(frame_w, int(cx + size / 2))
    y1 = min(frame_h, int(cy + size / 2))
    return x0, y0, x1, y1


def draw_face_grid(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                   rows: int = 3, cols: int = 3,
                   color=(0, 255, 200), thickness: int = 1) -> None:
    """
    Draw the bounding box + a rows×cols grid inside it, live on the frame.
    """
    box_w = x1 - x0
    box_h = y1 - y0

    # Outer bounding box
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)

    # Vertical dividers
    for c in range(1, cols):
        x = x0 + int(box_w * c / cols)
        cv2.line(frame, (x, y0), (x, y1), color, thickness, cv2.LINE_AA)

    # Horizontal dividers
    for r in range(1, rows):
        y = y0 + int(box_h * r / rows)
        cv2.line(frame, (x0, y), (x1, y), color, thickness, cv2.LINE_AA)




def split_into_grid(image: np.ndarray, rows: int = 3, cols: int = 3) -> list:
    """
    Divide `image` into rows×cols equal segments.

    Returns a 2-D list  grid[row][col]  where each element is a numpy
    sub-array (a view, not a copy — zero extra memory until you write them).

    Maths:
        cell_h = H // rows   (integer division; any remainder is trimmed)
        cell_w = W // cols
        grid[r][c] = image[r*cell_h : (r+1)*cell_h,
                           c*cell_w : (c+1)*cell_w]
    """
    H, W = image.shape[:2]
    cell_h = H // rows
    cell_w = W // cols

    grid = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            y_start = r * cell_h
            y_end   = (r + 1) * cell_h
            x_start = c * cell_w
            x_end   = (c + 1) * cell_w
            cell = image[y_start:y_end, x_start:x_end]
            row_cells.append(cell)
        grid.append(row_cells)

    return grid          # grid[row][col], each shape: (cell_h, cell_w, 3)


def save_snapshot(frame: np.ndarray, lms_list: list, snap_count: int) -> int:
    """Crop & save a square face image for each detected face, then split into 3×3 grid."""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    h, w = frame.shape[:2]
    ts = time.strftime("%Y%m%d_%H%M%S")
    for i, lms in enumerate(lms_list):
        x0, y0, x1, y1 = face_bounding_box(lms, h, w)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop_sq = cv2.resize(crop, (SNAPSHOT_SIZE, SNAPSHOT_SIZE),
                             interpolation=cv2.INTER_LANCZOS4)

        # ── Save full face ────────────────────────────────────────────────────
        path = os.path.join(SNAPSHOT_DIR, f"face_{ts}_f{i+1}_{snap_count:04d}.jpg")
        cv2.imwrite(path, crop_sq)
        print(f"[SNAP] Saved → {path}  ({crop_sq.shape[1]}×{crop_sq.shape[0]} px)")

        # ── Split into 3×3 grid ───────────────────────────────────────────────
        grid = split_into_grid(crop_sq, rows=3, cols=3)   # list[3][3] of ndarrays

        cell_h, cell_w = grid[0][0].shape[:2]
        print(f"       Grid: 3×3 cells, each {cell_w}×{cell_h} px")

        grid_dir = os.path.join(SNAPSHOT_DIR, f"grid_{ts}_f{i+1}_{snap_count:04d}")
        os.makedirs(grid_dir, exist_ok=True)
        for r in range(3):
            for c in range(3):
                cell_path = os.path.join(grid_dir, f"cell_r{r}_c{c}.jpg")
                cv2.imwrite(cell_path, grid[r][c])
        print(f"       Grid tiles saved → {grid_dir}/")

        snap_count += 1
    return snap_count



def main() -> None:
    download_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=4,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    print("Facial Mesh — SPACE = snapshot | 'q' = quit")

    fps_times: deque = deque(maxlen=30)
    frame_ts_ms  = 0
    snap_count   = 0
    flash_frames = 0   # countdown for white-flash effect after snapshot

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame — retrying…")
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            frame_ts_ms += 33
            result = landmarker.detect_for_video(mp_image, frame_ts_ms)

            if result.face_landmarks:
                for face_lms in result.face_landmarks:
                    draw_face(frame, face_lms)
                # Draw bounding box preview around each face
                for face_lms in result.face_landmarks:
                    x0, y0, x1, y1 = face_bounding_box(face_lms, h, w)
                    draw_face_grid(frame, x0, y0, x1, y1, rows=3, cols=3, color=(0, 255, 200), thickness=1)

            # White flash effect on snapshot
            if flash_frames > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (255, 255, 255), -1)
                alpha = flash_frames / 8.0
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                flash_frames -= 1

            # Status bar
            face_count = len(result.face_landmarks) if result.face_landmarks else 0
            cv2.putText(frame,
                        f"Faces: {face_count}  Snaps: {snap_count}  SPACE=snap  Q=quit",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1, cv2.LINE_AA)

            if SHOW_FPS:
                fps_times.append(time.perf_counter())
                if len(fps_times) >= 2:
                    fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
                    cv2.putText(
                        frame, f"FPS: {fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA,
                    )

            cv2.imshow("Facial Mesh", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):          # SPACEBAR → snapshot
                if result.face_landmarks:
                    snap_count = save_snapshot(frame, result.face_landmarks, snap_count)
                    flash_frames = 8
                else:
                    print("[SNAP] No face detected — skipping.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {snap_count} snapshot(s) saved to '{SNAPSHOT_DIR}/'")


if __name__ == '__main__':
    main()