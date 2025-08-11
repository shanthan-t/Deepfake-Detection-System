import cv2
import numpy as np

def detect_video_deepfake(video_path, detector, target_frames=60):
    """
    Sample frames, run the image detector (faces + alignment + TTA),
    and aggregate with mean + 95th percentile.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, total // max(1, target_frames))

    probs = []
    idx = 0
    while idx < total and len(probs) < target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        p = detector.score_frame_np(frame_rgb)   # [0..1]
        probs.append(p)
        idx += step

    cap.release()

    if not probs:
        return False, 0.0, 0

    probs = np.array(probs)
    mean = float(np.mean(probs))
    p95  = float(np.percentile(probs, 95))
    agg  = 0.8*mean + 0.2*p95

    is_fake = agg >= 0.5
    confidence = agg*100 if is_fake else (1.0-agg)*100
    return bool(is_fake), float(confidence), int(len(probs))
