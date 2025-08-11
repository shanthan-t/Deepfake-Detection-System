import os
import io
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datetime import datetime
import librosa
import librosa.display
from priva_main.config import TEMP_DIR

def get_file_details(file):
    return {
        "filename": file.name,
        "file_type": getattr(file, "type", "unknown"),
        "file_size": f"{getattr(file, 'size', 0) / 1024:.2f} KB",
    }

def save_uploaded_file(uploaded_file):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=str(TEMP_DIR))
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name

def create_audio_visualization(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        fig = plt.figure(figsize=(10, 8))

        ax1 = fig.add_subplot(3, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        ax1.set_title('Waveform')

        ax2 = fig.add_subplot(3, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
        img2 = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax2)
        fig.colorbar(img2, ax=ax2, format='%+2.0f dB')
        ax2.set_title('Log-frequency spectrogram')

        ax3 = fig.add_subplot(3, 1, 3)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        img3 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax3)
        fig.colorbar(img3, ax=ax3)
        ax3.set_title('MFCC')

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    except Exception:
        blank = np.full((480, 960, 3), 230, dtype=np.uint8)
        return Image.fromarray(blank)

def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
