"""
Multimodal Emotion Recognition — Real-time Demo
================================================
Kết hợp Face stream + Audio stream real-time.
- Face  : SCRFD detector + CNN emotion model (ONNX)
- Audio : Microphone + CNN+LSTM model (ONNX)
- Fusion: Weighted average (weight từ fusion_weights.json)

Usage:
    python demo_multimodal.py
    python demo_multimodal.py --face-model  models/emotion_model.onnx
                              --audio-model models/audio_model.onnx
                              --fusion      models/fusion_weights.json
Controls:
    q  — Quit
    s  — Screenshot
    f  — Toggle fusion / face-only mode
    k  — Toggle keypoints
    b  — Toggle confidence bars
"""

import cv2
import numpy as np
import argparse
import os
import sys
import json
import threading
import queue
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from face_emotion_pipeline import FaceEmotionPipeline, EMOTION_COLORS, EMOTIONS
from audio_stream import AudioPredictor, RealtimeAudioBuffer
from fusion import LateFusion

# ─────────────────────────────────────────────
#  DEFAULT PATHS
# ─────────────────────────────────────────────
DEFAULT_DETECTOR    = 'models/det_10g_fp16_dynamic.onnx'
DEFAULT_FACE_MODEL  = 'models/emotion_model.onnx'
DEFAULT_AUDIO_MODEL = 'models/audio_model.onnx'
DEFAULT_CLASSES     = 'models/class_indices.json'
DEFAULT_FUSION      = 'models/fusion_weights.json'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--detector',    default=DEFAULT_DETECTOR)
    p.add_argument('--face-model',  default=DEFAULT_FACE_MODEL)
    p.add_argument('--audio-model', default=DEFAULT_AUDIO_MODEL)
    p.add_argument('--classes',     default=DEFAULT_CLASSES)
    p.add_argument('--fusion',      default=DEFAULT_FUSION)
    p.add_argument('--cam',         type=int, default=0)
    p.add_argument('--det-thresh',  type=float, default=0.45)
    p.add_argument('--no-audio',    action='store_true',
                   help='Tắt audio stream, chỉ dùng face')
    return p.parse_args()


# ─────────────────────────────────────────────
#  DRAW HELPERS
# ─────────────────────────────────────────────
def draw_audio_panel(frame, audio_result, x=10, y=60):
    """Vẽ audio emotion bar ở góc trái."""
    em, conf, probs = audio_result
    color = EMOTION_COLORS.get(em, (200, 200, 200))

    # Label
    label = f"AUDIO: {em.upper()} {conf*100:.0f}%"
    cv2.putText(frame, label, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Mini bars
    for i, (prob, emotion) in enumerate(zip(probs, EMOTIONS)):
        by  = y + 10 + i * 16
        ec  = EMOTION_COLORS.get(emotion, (150, 150, 150))
        filled = int(prob * 80)
        cv2.rectangle(frame, (x, by), (x+80, by+11), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, by), (x+filled, by+11), ec, -1)
        cv2.putText(frame, f"{emotion[:3]}:{prob*100:.0f}%",
                    (x+83, by+10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (200, 200, 200), 1)


def draw_fused_result(frame, fused_em, fused_conf, fused_probs):
    """Vẽ kết quả fusion ở góc phải trên."""
    h, w = frame.shape[:2]
    color = EMOTION_COLORS.get(fused_em, (0, 255, 0))

    # Background panel
    panel_w, panel_h = 220, 35
    px = w - panel_w - 10
    py = 10
    cv2.rectangle(frame, (px-5, py-5), (px+panel_w, py+panel_h),
                  (30, 30, 30), -1)
    cv2.rectangle(frame, (px-5, py-5), (px+panel_w, py+panel_h),
                  color, 2)

    # Fused label
    label = f"FUSED: {fused_em.upper()} {fused_conf*100:.0f}%"
    cv2.putText(frame, label, (px, py+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_stream_status(frame, face_active, audio_active, fusion_mode):
    """Vẽ status indicators."""
    h, w = frame.shape[:2]
    y = h - 35

    statuses = [
        (f"FACE {'ON' if face_active else 'OFF'}",
         (0, 200, 0) if face_active else (100, 100, 100)),
        (f"AUDIO {'ON' if audio_active else 'OFF'}",
         (0, 200, 200) if audio_active else (100, 100, 100)),
        (f"FUSION {'ON' if fusion_mode else 'OFF'}",
         (200, 200, 0) if fusion_mode else (100, 100, 100)),
    ]
    x = 10
    for text, color in statuses:
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        x += 130


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # Kiểm tra files
    for path, name in [(args.detector,   'Face Detector'),
                       (args.face_model, 'Face Emotion Model')]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} not found: {path}")
            return

    os.makedirs('results', exist_ok=True)

    print("\n" + "=" * 55)
    print("  Multimodal Emotion Recognition — Demo")
    print("=" * 55)
    print(f"  Face detector : {os.path.basename(args.detector)}")
    print(f"  Face model    : {args.face_model}")
    print(f"  Audio model   : {args.audio_model}")
    print(f"  Fusion weights: {args.fusion}")
    print("=" * 55)

    # ── Load Face Pipeline ─────────────────────
    face_pipeline = FaceEmotionPipeline(
        detector_path      = args.detector,
        emotion_model_path = args.face_model,
        class_indices_path = args.classes,
        det_conf_threshold = args.det_thresh,
    )

    # ── Load Audio Predictor ───────────────────
    audio_buffer = None
    audio_active = False

    if not args.no_audio and os.path.exists(args.audio_model):
        try:
            audio_predictor = AudioPredictor(args.audio_model, args.classes)
            audio_buffer    = RealtimeAudioBuffer(audio_predictor)
            audio_buffer.start()
            audio_active = True
            print("[AUDIO] Stream started")
        except Exception as e:
            print(f"[AUDIO] Failed to start: {e}")
            print("[AUDIO] Install: pip install sounddevice librosa")
    else:
        print("[AUDIO] Skipped (no model or --no-audio)")

    # ── Load Fusion Weights ────────────────────
    fusion = LateFusion(w_face=0.5, w_audio=0.5, mode='confidence')
    if os.path.exists(args.fusion):
        fusion.load(args.fusion)

    # ── Webcam ─────────────────────────────────
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.cam}")
        return

    print("\n  q: Quit | s: Screenshot | f: Toggle fusion | k: Keypoints | b: Bars")
    print("  m: Toggle fusion mode (confidence / fixed)")
    print("  +/-: Detection threshold\n")

    show_kps   = True
    show_bars  = True
    fusion_mode = True
    screenshot_count = 0
    frame_count = 0
    t_start = datetime.now()

    # Default audio result
    default_audio = ('neutral', 0.0, np.ones(7) / 7)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ── Face Stream ────────────────────────
        face_results = face_pipeline.process(frame)

        # ── Audio Stream ───────────────────────
        audio_result = audio_buffer.get_result() if audio_active else default_audio

        # ── Fusion ─────────────────────────────
        # Ghi de emotion tren bbox bang ket qua fusion
        if fusion_mode and face_results and audio_active:
            audio_probs = audio_result[2]
            for r in face_results:
                fused_em, fused_conf, fused_probs = fusion.fuse(
                    r['probs'], audio_probs)
                # Thay the bang ket qua fusion
                r['emotion']    = fused_em
                r['confidence'] = fused_conf
                r['probs']      = fused_probs

        # Ve frame voi ket qua da duoc fusion
        frame_out = face_pipeline.draw(frame, face_results,
                                       show_kps=show_kps,
                                       show_bars=show_bars)

        # ── FPS & Status ───────────────────────
        frame_count += 1
        elapsed = (datetime.now() - t_start).total_seconds()
        fps = frame_count / elapsed if elapsed > 0 else 0

        cv2.putText(frame_out, f"FPS:{fps:.1f}",
                    (10, frame_out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        draw_stream_status(frame_out, bool(face_results),
                           audio_active, fusion_mode)

        cv2.imshow('Multimodal Emotion Recognition', frame_out)

        # ── Keys ───────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_count += 1
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = f'results/screenshot_{ts}.png'
            cv2.imwrite(path, frame_out)
            print(f"Screenshot: {path}")
        elif key == ord('f'):
            fusion_mode = not fusion_mode
            print(f"Fusion: {'ON' if fusion_mode else 'OFF'}")
        elif key == ord('k'):
            show_kps = not show_kps
        elif key == ord('b'):
            show_bars = not show_bars
        elif key in (ord('+'), ord('=')):
            face_pipeline.detector.conf_thresh = min(
                0.95, face_pipeline.detector.conf_thresh + 0.05)
        elif key == ord('-'):
            face_pipeline.detector.conf_thresh = max(
                0.10, face_pipeline.detector.conf_thresh - 0.05)
        elif key == ord('m'):
            fusion.mode = 'fixed' if fusion.mode == 'confidence' else 'confidence'
            print(f"Fusion mode: {fusion.mode.upper()}")

    # Cleanup
    if audio_buffer:
        audio_buffer.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Demo closed.")


if __name__ == '__main__':
    main()
