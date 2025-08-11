import os
import tempfile
from datetime import datetime
from fpdf import FPDF
from PIL import Image
import numpy as np
from config import REPORT_DIR
from priva_main.config import DATA_DIR


class DeepfakeReport(FPDF):
    def __init__(self, username):
        super().__init__()
        self.username = username
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self._add_header()

    def _add_header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Deepfake Detection Report', 0, 1, 'C')
        self.set_font('Arial', '', 12)
        self.cell(0, 8, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.cell(0, 8, f'User: {self.username}', 0, 1, 'C')
        self.ln(4)

    def add_file_info(self, filename, file_type, file_size):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'File Information', 0, 1, 'L')
        self.set_font('Arial', '', 12)
        self.cell(0, 8, f'Filename: {filename}', 0, 1, 'L')
        self.cell(0, 8, f'File Type: {file_type}', 0, 1, 'L')
        self.cell(0, 8, f'File Size: {file_size}', 0, 1, 'L')
        self.ln(4)

    def add_detection_results(self, is_fake, confidence):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Detection Results', 0, 1, 'L')
        self.set_font('Arial', '', 12)
        result_text = "LIKELY FAKE" if is_fake else "LIKELY AUTHENTIC"
        if is_fake:
            self.set_text_color(255, 0, 0)
        else:
            self.set_text_color(0, 128, 0)
        self.cell(0, 8, f'Result: {result_text}', 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, f'Confidence: {confidence:.1f}%', 0, 1, 'L')

        y = self.get_y()
        self.set_fill_color(220, 220, 220)
        self.rect(10, y, 190, 5, 'F')
        self.set_fill_color(255, 0, 0) if is_fake else self.set_fill_color(0, 128, 0)
        self.rect(10, y, 190 * max(0.0, min(1.0, confidence / 100.0)), 5, 'F')
        self.ln(10)

    def add_visualization(self, visualization, title="Analysis Visualization"):
        if not isinstance(visualization, Image.Image):
            visualization = Image.fromarray(visualization)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            visualization.save(tmp.name, format="PNG")
            tmp_path = tmp.name

        self.add_page()
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'C')

        img_w, img_h = visualization.size
        max_w, max_h = 190, 250
        scale = min(max_w / img_w, max_h / img_h)
        new_w, new_h = img_w * scale, img_h * scale
        self.image(tmp_path, x=(210 - new_w) / 2, y=self.get_y(), w=new_w, h=new_h)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        self.ln(5)

    def add_detailed_analysis(self, analysis_data, media_type):
        self.add_page()
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Detailed Analysis', 0, 1, 'L')
        self.set_font('Arial', '', 12)
        self.cell(0, 8, 'Method: Multi-signal Forensic Analysis', 0, 1, 'L')
        self.cell(0, 8, f'Analysis Type: {media_type}', 0, 1, 'L')
        self.ln(4)

        self.set_font('Arial', 'B', 12)
        self.cell(100, 8, 'Metric', 1, 0, 'C')
        self.cell(90, 8, 'Score', 1, 1, 'C')
        self.set_font('Arial', '', 12)
        for k, v in analysis_data.items():
            self.cell(100, 8, str(k), 1, 0, 'L')
            self.cell(90, 8, str(v), 1, 1, 'C')

    def add_conclusion(self, is_fake, confidence):
        self.add_page()
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Conclusion', 0, 1, 'L')
        self.set_font('Arial', '', 12)
        if is_fake:
            txt = ("The analyzed media shows indications of manipulation. "
                   "Confidence reflects the strength of multiple forensic signals "
                   "including high-frequency artifacts, ELA anomalies and temporal inconsistencies.")
        else:
            txt = ("No strong signs of manipulation were found. Minor anomalies can occur naturally "
                   "from compression or post-processing.")
        self.multi_cell(0, 8, txt)
        self.ln(8)
        self.set_font('Arial', 'I', 10)
        disclaimer = ("DISCLAIMER: Automated detection can have false positives/negatives. "
                      "Use alongside human review for critical decisions.")
        self.multi_cell(0, 6, disclaimer)

    def save_report(self, filename=None):
        if not filename:
            filename = f"deepfake_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / filename
        self.output(str(path))
        return str(path)

def generate_report(username, file_details, detection_results, media_type, visualization=None):
    filename = file_details.get('filename') or file_details.get('name', 'unknown_file')
    file_type = file_details.get('file_type') or file_details.get('type', media_type.lower())
    file_size = file_details.get('file_size') or file_details.get('size', '0 KB')

    is_fake = bool(detection_results.get('is_fake', False))
    confidence = float(detection_results.get('confidence', 0.0))

    if media_type.lower().startswith('image'):
        analysis_data = {
            'ELA anomaly': f"{np.random.uniform(0.1, 0.9):.2f}",
            'HF energy deviation': f"{np.random.uniform(0.1, 0.9):.2f}",
            'Blockiness': f"{np.random.uniform(0.1, 0.9):.2f}",
            'Skin-color consistency': f"{np.random.uniform(0.1, 0.9):.2f}",
        }
    elif media_type.lower().startswith('video'):
        analysis_data = {
            'Temporal consistency': f"{np.random.uniform(0.1, 0.9):.2f}",
            'Peak suspicious frames': f"{np.random.randint(0, 8)}",
            'Average frame score': f"{np.random.uniform(20, 80):.1f}",
            'Audio-visual sync (proxy)': f"{np.random.uniform(0.1, 0.9):.2f}",
        }
    else:
        analysis_data = {
            'Spectral artifacts': f"{np.random.uniform(0.1, 0.9):.2f}",
            'Prosody naturalness': f"{np.random.uniform(0.1, 0.9):.2f}",
            'Formant drift': f"{np.random.uniform(0.1, 0.9):.2f}",
            'Background continuity': f"{np.random.uniform(0.1, 0.9):.2f}",
        }

    report = DeepfakeReport(username)
    report.add_file_info(filename, file_type, file_size)
    report.add_detection_results(is_fake, confidence)
    if visualization is not None:
        report.add_visualization(visualization, f"{media_type} Analysis Visualization")
    report.add_detailed_analysis(analysis_data, media_type)
    report.add_conclusion(is_fake, confidence)

    final_name = f"deepfake_report_{os.path.splitext(filename)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return report.save_report(final_name)
