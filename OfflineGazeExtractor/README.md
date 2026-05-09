# Offline Gaze Extractor (No Calibration)

This tool extracts **relative gaze direction** and **head forward direction** from prerecorded videos.
It is intended for scenarios where calibration is not possible (for example, naturalistic HRI recordings).

## What You Get Per Frame

- `face_detected` (0 or 1)
- `head_forward_x/y/z` (unit vector in camera coordinates)
- `gaze_rel_x/y/z` (unit vector in camera coordinates)
- `eye_yaw_proxy_deg`, `eye_pitch_proxy_deg` (proxy angles from iris offsets)
- `left_dx/right_dx/left_dy/right_dy` (normalized iris offsets)
- `confidence` (0..1 proxy based on eye openness/visibility)

Outputs are written to both CSV and JSON.

## Run

From repo root:

```bash
cd OfflineGazeExtractor
python extract_gaze_offline.py --input /path/to/video.mp4 --csv-out out/gaze.csv --json-out out/gaze.json --preview
```

Press `q` to stop preview early.

## Notes

- This is **not absolute screen point-of-regard**.
- It provides relative gaze/head direction useful for offline analysis, event detection, and HRI behavior coding.
- For absolute gaze in world coordinates, you still need camera geometry and calibration.
