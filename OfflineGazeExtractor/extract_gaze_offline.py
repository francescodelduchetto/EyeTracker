import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import mediapipe as mp


# Face mesh landmark indices
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_UPPER = 159
LEFT_EYE_LOWER = 145
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
RIGHT_EYE_UPPER = 386
RIGHT_EYE_LOWER = 374
FOREHEAD = 10
CHIN = 152
NOSE_TIP = 1


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v.astype(float)
    return (v / n).astype(float)


def _load_face_mesh_module():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return mp.solutions.face_mesh
    try:
        from mediapipe.python.solutions import face_mesh as face_mesh_module

        return face_mesh_module
    except Exception as exc:
        raise ImportError(
            "MediaPipe FaceMesh API not found. Install an official mediapipe build "
            "that includes FaceMesh solutions."
        ) from exc


def _as_px(lm, w: int, h: int) -> np.ndarray:
    return np.array([lm.x * w, lm.y * h, lm.z * w], dtype=float)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _estimate_head_axes(lms, w: int, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_outer = _as_px(lms[LEFT_EYE_OUTER], w, h)
    right_outer = _as_px(lms[RIGHT_EYE_OUTER], w, h)
    eye_mid = (left_outer + right_outer) * 0.5
    forehead = _as_px(lms[FOREHEAD], w, h)
    chin = _as_px(lms[CHIN], w, h)

    x_axis = _normalize(right_outer - left_outer)
    up_raw = _normalize(forehead - chin)
    z_axis = _normalize(np.cross(x_axis, up_raw))
    y_axis = _normalize(np.cross(z_axis, x_axis))

    # Keep forward mostly toward camera-positive z for stability.
    if z_axis[2] < 0:
        z_axis = -z_axis
        y_axis = -y_axis

    return eye_mid, x_axis, z_axis


def _estimate_gaze_3d(lms, w: int, h: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """Estimate gaze direction using MediaPipe's 3D iris landmarks and a simple eyeball model.

    Rather than converting 2D iris pixel offsets to arbitrary angles, this function
    works directly in 3D camera space:
      1. Eye socket center = midpoint of inner/outer eye corners (in 3D).
      2. Eyeball center = socket center pushed back in depth by ~half the eye width
         (anatomical prior: eyeball radius ≈ half of visible eye width in pixels,
         in the same scale that MediaPipe uses for its z coordinate).
      3. Gaze direction = normalize(iris_3d - eyeball_center_3d).

    MediaPipe z convention (with refine_landmarks=True):
      - z is in the same scale as x (normalised by image width).
      - More positive z = farther from camera (deeper into the face).

    The returned vector is in camera image space (x right, y down, z away from camera),
    so a person looking straight at the camera produces a vector with a negative z
    component.
    """
    li = _as_px(lms[LEFT_IRIS_CENTER], w, h)
    ri = _as_px(lms[RIGHT_IRIS_CENTER], w, h)
    lo = _as_px(lms[LEFT_EYE_OUTER], w, h)
    lii = _as_px(lms[LEFT_EYE_INNER], w, h)
    ro = _as_px(lms[RIGHT_EYE_OUTER], w, h)
    rii = _as_px(lms[RIGHT_EYE_INNER], w, h)

    left_socket = (lo + lii) * 0.5
    right_socket = (ro + rii) * 0.5

    left_eye_w = float(np.linalg.norm(lo[:2] - lii[:2]))
    right_eye_w = float(np.linalg.norm(ro[:2] - rii[:2]))

    if min(left_eye_w, right_eye_w) < 1e-6:
        return np.array([0.0, 0.0, -1.0]), {
            "eye_yaw_proxy": 0.0, "eye_pitch_proxy": 0.0,
            "left_dx": 0.0, "right_dx": 0.0, "left_dy": 0.0, "right_dy": 0.0,
        }

    # Push eyeball center behind the socket along the depth axis.
    left_eyeball = left_socket.copy()
    left_eyeball[2] += left_eye_w * 0.5
    right_eyeball = right_socket.copy()
    right_eyeball[2] += right_eye_w * 0.5

    left_gaze = _normalize(li - left_eyeball)
    right_gaze = _normalize(ri - right_eyeball)
    gaze = _normalize(left_gaze + right_gaze)

    # Approximate yaw/pitch from camera-space gaze for logging.
    gz = gaze[2] if abs(gaze[2]) > 1e-6 else -1e-6
    eye_yaw = float(math.degrees(math.atan2(gaze[0], -gz)))
    eye_pitch = float(math.degrees(math.atan2(-gaze[1], -gz)))

    left_dx = float(_clamp((li[0] - left_socket[0]) / (left_eye_w * 0.5), -1.0, 1.0))
    right_dx = float(_clamp((ri[0] - right_socket[0]) / (right_eye_w * 0.5), -1.0, 1.0))
    left_dy = float(_clamp((li[1] - left_socket[1]) / (left_eye_w * 0.5), -1.0, 1.0))
    right_dy = float(_clamp((ri[1] - right_socket[1]) / (right_eye_w * 0.5), -1.0, 1.0))

    return gaze, {
        "eye_yaw_proxy": eye_yaw,
        "eye_pitch_proxy": eye_pitch,
        "left_dx": left_dx,
        "right_dx": right_dx,
        "left_dy": left_dy,
        "right_dy": right_dy,
    }


def _confidence_from_eyes(lms, w: int, h: int) -> float:
    lu = _as_px(lms[LEFT_EYE_UPPER], w, h)
    ll = _as_px(lms[LEFT_EYE_LOWER], w, h)
    ru = _as_px(lms[RIGHT_EYE_UPPER], w, h)
    rl = _as_px(lms[RIGHT_EYE_LOWER], w, h)
    lo = _as_px(lms[LEFT_EYE_OUTER], w, h)
    lii = _as_px(lms[LEFT_EYE_INNER], w, h)
    ro = _as_px(lms[RIGHT_EYE_OUTER], w, h)
    rii = _as_px(lms[RIGHT_EYE_INNER], w, h)

    left_ratio = float(np.linalg.norm(lu[:2] - ll[:2]) / max(np.linalg.norm(lo[:2] - lii[:2]), 1e-6))
    right_ratio = float(np.linalg.norm(ru[:2] - rl[:2]) / max(np.linalg.norm(ro[:2] - rii[:2]), 1e-6))
    avg = (left_ratio + right_ratio) * 0.5

    # Map approximately-open eye ratio to [0,1].
    return _clamp((avg - 0.08) / 0.20, 0.0, 1.0)


def _draw_wire_box(frame, corners, color, thickness=2):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    h, w = frame.shape[:2]
    pts = [(int(p[0]), int(p[1])) for p in corners]
    for i, j in edges:
        ok, a, b = cv2.clipLine((0, 0, w, h), pts[i], pts[j])
        if ok:
            cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)


def _draw_main_overlay(
    frame,
    eye_mid: np.ndarray,
    gaze_dir: np.ndarray,
    head_right: np.ndarray,
    confidence: float,
    text: str,
    conf_threshold: float = 0.45,
):
    h, w = frame.shape[:2]

    gaze_n = _normalize(gaze_dir)
    right_n = _normalize(head_right)
    up_n = _normalize(np.cross(right_n, gaze_n))
    if np.linalg.norm(up_n) < 1e-6:
        up_n = np.array([0.0, -1.0, 0.0], dtype=float)

    near_d = 70.0
    far_d = 190.0
    half_w_near = 30.0
    half_h_near = 22.0
    half_w_far = 52.0
    half_h_far = 38.0

    c_near = eye_mid + gaze_n * near_d
    c_far = eye_mid + gaze_n * far_d

    n0 = c_near - right_n * half_w_near - up_n * half_h_near
    n1 = c_near + right_n * half_w_near - up_n * half_h_near
    n2 = c_near + right_n * half_w_near + up_n * half_h_near
    n3 = c_near - right_n * half_w_near + up_n * half_h_near

    f0 = c_far - right_n * half_w_far - up_n * half_h_far
    f1 = c_far + right_n * half_w_far - up_n * half_h_far
    f2 = c_far + right_n * half_w_far + up_n * half_h_far
    f3 = c_far - right_n * half_w_far + up_n * half_h_far

    box_color = (0, 220, 0) if confidence >= conf_threshold else (0, 0, 255)
    _draw_wire_box(frame, [n0, n1, n2, n3, f0, f1, f2, f3], box_color, thickness=1)

    # Re-include a center orientation line to make direction explicit.
    p0 = (int(eye_mid[0]), int(eye_mid[1]))
    p1 = (int((eye_mid + gaze_n * (far_d + 35.0))[0]), int((eye_mid + gaze_n * (far_d + 35.0))[1]))
    ok_line, cl0, cl1 = cv2.clipLine((0, 0, w, h), p0, p1)
    if ok_line:
        cv2.arrowedLine(frame, cl0, cl1, box_color, 2, tipLength=0.2)

    origin = (int(eye_mid[0]), int(eye_mid[1]))
    cv2.circle(frame, origin, 4, box_color, -1)

    conf_state = "HIGH" if confidence >= conf_threshold else "LOW"
    cv2.putText(frame, "GAZE (relative) box", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"confidence={confidence:.2f} ({conf_state}) threshold={conf_threshold:.2f}",
        (10, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        box_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(frame, text, (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "space=pause/resume, left/right (or A/D)=step, q=quit", (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def _is_left_key(key_code: int) -> bool:
    low8 = key_code & 0xFF
    low16 = key_code & 0xFFFF
    return key_code in (81, 2424832, 65361, 123, 63234) or low8 == 81 or low16 in (81, 123)


def _is_right_key(key_code: int) -> bool:
    low8 = key_code & 0xFF
    low16 = key_code & 0xFFFF
    return key_code in (83, 2555904, 65363, 124, 63235) or low8 == 83 or low16 in (83, 124)


def _read_frame_at(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return ok, frame


def process_video(
    input_path: str,
    csv_path: str,
    json_path: str,
    preview: bool,
    invert_yaw: bool = False,
    invert_pitch: bool = False,
) -> None:
    # invert_yaw and invert_pitch are accepted for CLI backward-compatibility
    # but are no longer used; the 3D eyeball model gets signs right by construction.
    face_mesh_module = _load_face_mesh_module()
    face_mesh = face_mesh_module.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1e-6:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        total_frames = 0

    rows_by_frame = {}
    frame_idx = 0
    playing = True

    while True:
        if preview:
            if total_frames > 0:
                frame_idx = int(_clamp(frame_idx, 0, total_frames - 1))
            ok, frame = _read_frame_at(cap, frame_idx)
        else:
            ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        timestamp = frame_idx / fps

        row = {
            "frame": frame_idx,
            "time_sec": timestamp,
            "face_detected": 0,
            "head_forward_x": 0.0,
            "head_forward_y": 0.0,
            "head_forward_z": 0.0,
            "gaze_rel_x": 0.0,
            "gaze_rel_y": 0.0,
            "gaze_rel_z": 0.0,
            "eye_yaw_proxy_deg": 0.0,
            "eye_pitch_proxy_deg": 0.0,
            "left_dx": 0.0,
            "right_dx": 0.0,
            "left_dy": 0.0,
            "right_dy": 0.0,
            "confidence": 0.0,
        }

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            eye_mid, head_right, head_forward = _estimate_head_axes(lms, w, h)
            gaze_rel, metrics = _estimate_gaze_3d(lms, w, h)
            conf = _confidence_from_eyes(lms, w, h)

            row.update(
                {
                    "face_detected": 1,
                    "head_forward_x": float(head_forward[0]),
                    "head_forward_y": float(head_forward[1]),
                    "head_forward_z": float(head_forward[2]),
                    "gaze_rel_x": float(gaze_rel[0]),
                    "gaze_rel_y": float(gaze_rel[1]),
                    "gaze_rel_z": float(gaze_rel[2]),
                    "eye_yaw_proxy_deg": float(metrics["eye_yaw_proxy"]),
                    "eye_pitch_proxy_deg": float(metrics["eye_pitch_proxy"]),
                    "left_dx": float(metrics["left_dx"]),
                    "right_dx": float(metrics["right_dx"]),
                    "left_dy": float(metrics["left_dy"]),
                    "right_dy": float(metrics["right_dy"]),
                    "confidence": float(conf),
                }
            )

            if preview:
                status = f"f={frame_idx} t={timestamp:.3f}s conf={conf:.2f}"
                _draw_main_overlay(frame, eye_mid, gaze_rel, head_right, conf, status)

        rows_by_frame[frame_idx] = row

        if preview:
            cv2.imshow("Offline Gaze Extractor", frame)
            key_code = cv2.waitKeyEx(1)
            key_ascii = key_code & 0xFF

            if key_ascii == ord("q"):
                break

            if key_ascii == ord(" "):
                playing = not playing
            elif _is_left_key(key_code) or key_ascii == ord("a"):
                playing = False
                frame_idx -= 1
            elif _is_right_key(key_code) or key_ascii == ord("d"):
                playing = False
                frame_idx += 1
            elif playing:
                frame_idx += 1

            if total_frames > 0 and frame_idx >= total_frames:
                break
            if frame_idx < 0:
                frame_idx = 0
        else:
            frame_idx += 1

    cap.release()
    face_mesh.close()
    if preview:
        cv2.destroyAllWindows()

    rows = [rows_by_frame[i] for i in sorted(rows_by_frame.keys())]

    fieldnames = list(rows[0].keys()) if rows else [
        "frame",
        "time_sec",
        "face_detected",
        "head_forward_x",
        "head_forward_y",
        "head_forward_z",
        "gaze_rel_x",
        "gaze_rel_y",
        "gaze_rel_z",
        "eye_yaw_proxy_deg",
        "eye_pitch_proxy_deg",
        "left_dx",
        "right_dx",
        "left_dy",
        "right_dy",
        "confidence",
    ]

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Processed {len(rows)} frames")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def _find_videos(folder: str) -> List[Path]:
    return sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _process_one_worker(args_tuple) -> str:
    """Top-level function so it is picklable by multiprocessing."""
    input_path, csv_path, json_path = args_tuple
    try:
        process_video(input_path, csv_path, json_path, preview=False)
        return f"OK  {input_path}"
    except Exception as exc:
        return f"ERR {input_path}: {exc}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract relative gaze and head direction from prerecorded video without calibration."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Path to a single input video")
    input_group.add_argument("--input-dir", help="Folder of videos to batch-process")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write output files in batch mode (default: same folder as the videos)",
    )
    parser.add_argument(
        "--csv-out",
        default="offline_gaze.csv",
        help="Output CSV path for single-file mode (default: offline_gaze.csv)",
    )
    parser.add_argument(
        "--json-out",
        default="offline_gaze.json",
        help="Output JSON path for single-file mode (default: offline_gaze.json)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live overlay while processing a single file (press q to stop)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel worker processes for batch mode (default: all CPU cores)",
    )
    parser.add_argument(
        "--invert-yaw",
        action="store_true",
        help="Invert yaw sign convention for gaze estimation.",
    )
    parser.add_argument(
        "--invert-pitch",
        action="store_true",
        help="Invert pitch sign convention for gaze estimation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.input:
        # Single-file mode
        process_video(
            args.input,
            args.csv_out,
            args.json_out,
            args.preview,
        )
        return

    # Batch mode
    videos = _find_videos(args.input_dir)
    if not videos:
        print(f"No video files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    work_items = []
    for v in videos:
        dest = out_dir or v.parent
        work_items.append((
            str(v),
            str(dest / (v.stem + "_gaze.csv")),
            str(dest / (v.stem + "_gaze.json")),
        ))

    workers = min(args.workers, len(work_items))
    print(f"Batch-processing {len(work_items)} video(s) with {workers} worker(s)...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_process_one_worker, work_items):
            print(result)


if __name__ == "__main__":
    main()
