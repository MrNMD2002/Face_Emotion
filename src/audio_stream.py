"""
Audio Emotion Stream
====================
MFCC extraction + CNN+LSTM model cho speech emotion recognition.
Dataset: RAVDESS
"""

import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
EMOTIONS    = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
SAMPLE_RATE = 16000
DURATION    = 2.5        # giây mỗi clip
N_MFCC      = 40         # số MFCC coefficients
HOP_LENGTH  = 512
MAX_FRAMES  = int(SAMPLE_RATE * DURATION / HOP_LENGTH) + 1   # ~79 frames

# RAVDESS emotion mapping → FER2013 labels
RAVDESS_MAP = {
    '01': 'neutral',
    '02': 'neutral',   # calm → neutral
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise'
}


# ─────────────────────────────────────────────
#  MFCC EXTRACTION
# ─────────────────────────────────────────────
def extract_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, max_frames=MAX_FRAMES):
    """
    Chuyển waveform → MFCC tensor.

    Returns:
        tensor: [1, 1, N_MFCC, max_frames] float32
    """
    # Đảm bảo đủ độ dài
    target_len = int(sr * DURATION)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mfcc = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=HOP_LENGTH
    )

    # Normalize
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

    # Pad/crop theo time axis
    if mfcc.shape[1] < max_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_frames]

    # [N_MFCC, T] → [1, 1, N_MFCC, T]
    return mfcc[np.newaxis, np.newaxis].astype(np.float32)


def load_audio_file(path, sr=SAMPLE_RATE):
    """Load file .wav về numpy array."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


# ─────────────────────────────────────────────
#  AUDIO CNN + LSTM MODEL
# ─────────────────────────────────────────────
class AudioEmotionNet(nn.Module):
    """
    CNN trích local features từ MFCC
    LSTM học temporal pattern của giọng nói.

    Input : [batch, 1, 40, T]  — MFCC spectrogram
    Output: [batch, 7]         — emotion logits
    """
    def __init__(self, n_mfcc=N_MFCC, num_classes=7, dropout=0.3):
        super().__init__()

        # CNN — trích features theo tần số
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(dropout),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),   # chỉ pool theo tần số, giữ time
            nn.Dropout2d(dropout),
        )

        # Tính output size của CNN
        cnn_freq = n_mfcc // 8       # 40 // 8 = 5
        cnn_ch   = 128
        lstm_in  = cnn_freq * cnn_ch  # 5 × 128 = 640

        # LSTM — học temporal pattern
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Classifier
        self.head = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, 1, 40, T]
        B, C, F, T = x.shape

        # CNN features
        x = self.cnn(x)           # [B, 128, F', T']

        # Reshape cho LSTM: [B, T', 128*F']
        B2, C2, F2, T2 = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B2, T2, C2 * F2)

        # LSTM
        x, _ = self.lstm(x)       # [B, T', 512]

        # Lấy hidden state cuối
        x = x[:, -1, :]           # [B, 512]

        return self.head(x)       # [B, 7]

    def get_features(self, x):
        """Trả về feature vector [512] để dùng cho fusion."""
        B, C, F, T = x.shape
        cnn_out = self.cnn(x)
        B2, C2, F2, T2 = cnn_out.shape
        cnn_out = cnn_out.permute(0, 3, 1, 2).reshape(B2, T2, C2 * F2)
        lstm_out, _ = self.lstm(cnn_out)
        return lstm_out[:, -1, :]   # [B, 512]


# ─────────────────────────────────────────────
#  ONNX PREDICTOR (inference)
# ─────────────────────────────────────────────
class AudioPredictor:
    """
    Load audio model từ ONNX, predict từ audio chunk real-time.
    """
    def __init__(self, model_path, class_indices_path=None):
        from scipy.special import softmax as sp_softmax
        self._softmax = sp_softmax

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session    = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        if class_indices_path and os.path.exists(class_indices_path):
            import json
            with open(class_indices_path) as f:
                self.idx_to_class = {v: k for k, v in json.load(f).items()}
        else:
            self.idx_to_class = {i: e for i, e in enumerate(EMOTIONS)}

        print(f"AudioPredictor loaded: {os.path.basename(model_path)}")
        print(f"  Providers: {self.session.get_providers()}")

    def predict_from_audio(self, audio_chunk, sr=SAMPLE_RATE):
        """
        Predict từ raw audio numpy array.

        Args:
            audio_chunk: numpy array [N samples]
        Returns:
            (emotion, confidence, probs[7])
        """
        mfcc_tensor = extract_mfcc(audio_chunk, sr=sr)
        return self._predict_mfcc(mfcc_tensor)

    def predict_from_file(self, audio_path):
        """Predict từ file .wav."""
        audio = load_audio_file(audio_path)
        return self.predict_from_audio(audio)

    def _predict_mfcc(self, mfcc_tensor):
        logits = self.session.run(None, {self.input_name: mfcc_tensor})[0][0]
        probs  = self._softmax(logits).astype(np.float32)
        idx    = int(np.argmax(probs))
        return self.idx_to_class[idx], float(probs[idx]), probs


# ─────────────────────────────────────────────
#  REALTIME AUDIO RECORDER
# ─────────────────────────────────────────────
class RealtimeAudioBuffer:
    """
    Sliding window buffer cho real-time microphone input.
    Predict mỗi STRIDE giây.
    """
    N_BANDS  = 28          # số cột spectrum
    FFT_SIZE = 1024        # độ phân giải FFT

    def __init__(self, predictor, sr=SAMPLE_RATE,
                 window=DURATION, stride=0.5):
        self.predictor  = predictor
        self.sr         = sr
        self.window     = window
        self.stride     = stride
        self.buffer     = np.zeros(int(sr * window))
        self.result     = ('neutral', 0.0,
                           np.ones(7) / 7)   # default
        self.volume     = 0.0                # RMS [0–1]
        self.freq_bands = np.zeros(self.N_BANDS)  # spectrum [0–1] per band

        # Bin edges chia theo log-scale (bỏ DC, lấy đến sr/2)
        fft_freqs  = np.fft.rfftfreq(self.FFT_SIZE, d=1.0 / sr)
        n_fft_bins = len(fft_freqs)
        edges = np.logspace(
            np.log10(max(fft_freqs[1], 80)),   # ~80 Hz
            np.log10(sr / 2),                  # Nyquist
            self.N_BANDS + 1
        )
        self._bin_edges = np.searchsorted(fft_freqs, edges).clip(1, n_fft_bins - 1)

        # Smoothing — giữ 30% giá trị cũ để tránh nhảy quá giật
        self._smooth = np.zeros(self.N_BANDS)

        self._running = False
        self._thread  = None

    def _callback(self, indata, frames, time, status):
        chunk = indata[:, 0]
        self.buffer = np.roll(self.buffer, -len(chunk))
        self.buffer[-len(chunk):] = chunk

        # ── RMS volume ──────────────────────────
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        self.volume = min(rms * 10, 1.0)

        # ── FFT spectrum → N_BANDS ───────────────
        # Zero-pad chunk lên FFT_SIZE
        padded   = np.zeros(self.FFT_SIZE)
        n        = min(len(chunk), self.FFT_SIZE)
        padded[:n] = chunk[:n] * np.hanning(n)   # Hann window giảm spectral leak
        mag      = np.abs(np.fft.rfft(padded))   # magnitude

        bands = np.zeros(self.N_BANDS)
        for i in range(self.N_BANDS):
            lo, hi = self._bin_edges[i], self._bin_edges[i + 1]
            if hi > lo:
                bands[i] = np.mean(mag[lo:hi])
            else:
                bands[i] = mag[lo]

        # Normalize + log-scale cho đẹp
        bands = np.log1p(bands * 20)
        peak  = bands.max()
        if peak > 1e-6:
            bands /= peak

        # Smooth: 70% mới + 30% cũ
        self._smooth = 0.7 * bands + 0.3 * self._smooth
        self.freq_bands = self._smooth.copy()

        try:
            em, conf, probs = self.predictor.predict_from_audio(
                self.buffer.copy(), sr=self.sr)
            self.result = (em, conf, probs)
        except Exception:
            pass

    def start(self):
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            blocksize=int(self.sr * self.stride),
            callback=self._callback
        )
        self._stream.start()
        print("[AUDIO] Microphone stream started")

    def stop(self):
        if hasattr(self, '_stream'):
            self._stream.stop()
            self._stream.close()
        print("[AUDIO] Microphone stream stopped")

    def get_result(self):
        return self.result

    def get_volume(self):
        """Trả về RMS volume mới nhất [0.0 – 1.0]."""
        return self.volume

    def get_freq_bands(self):
        """Trả về spectrum [N_BANDS] float32 [0–1]."""
        return self.freq_bands.copy()
