import sys
import queue
import threading
import warnings
import numpy as np
import soundcard as sc
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel, QPushButton
from PyQt6.QtCore import QThread, pyqtSignal

warnings.filterwarnings("ignore", message="data discontinuity in recording")

SAMPLE_RATE = 16000
CHUNK_SECONDS = 8
SUMMARY_EVERY = 4  # update summary after this many new transcript lines

audio_queue = queue.Queue()


# ---- Continuous recorder: runs in its own thread, never waits for AI ----
def record_stream(label, device, stop_event):
    frames = int(SAMPLE_RATE * CHUNK_SECONDS)
    with device.recorder(samplerate=SAMPLE_RATE) as r:
        while not stop_event.is_set():
            data = r.record(numframes=frames)
            mono = data.mean(axis=1) if data.ndim > 1 else data
            audio_queue.put((label, mono.astype(np.float32)))


# ---- Worker thread: transcribes and summarizes while recording continues ----
class PipelineWorker(QThread):
    transcript_updated = pyqtSignal(str)
    summary_updated = pyqtSignal(str)
    status_updated = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.transcript_buffer = ""

    def run(self):
        self.status_updated.emit("Loading Whisper model...")
        whisper_model = whisper.load_model("small")

        self.status_updated.emit("Loading summarization model...")
        model_name = "microsoft/Phi-3.5-mini-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

        def summarize(text):
            prompt = f"""Summarize this meeting transcript in 2-3 bullet points. Then list any action items.

Transcript:
{text}

Summary:"""
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = llm.generate(**inputs, max_new_tokens=200)
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        def is_silent(audio, threshold=0.01):
            return np.abs(audio).mean() < threshold

        # Start recording only after models are loaded
        mic = sc.default_microphone()
        loopback = sc.get_microphone(
            id=str(sc.default_speaker().name), include_loopback=True
        )
        threading.Thread(
            target=record_stream, args=("You", mic, self.stop_event), daemon=True
        ).start()
        threading.Thread(
            target=record_stream, args=("Others", loopback, self.stop_event), daemon=True
        ).start()

        lines_since_summary = 0

        while not self.stop_event.is_set():
            self.status_updated.emit(f"Listening... ({audio_queue.qsize()} chunks waiting)")
            try:
                label, audio = audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            if is_silent(audio):
                continue

            result = whisper_model.transcribe(audio, fp16=False, language="en")
            text = result["text"].strip()
            if not text:
                continue

            self.transcript_buffer += f"\n[{label}]: {text}"
            self.transcript_updated.emit(self.transcript_buffer)
            lines_since_summary += 1

            if lines_since_summary >= SUMMARY_EVERY:
                self.status_updated.emit("Updating summary... (still recording)")
                self.summary_updated.emit(summarize(self.transcript_buffer))
                lines_since_summary = 0

    def stop(self):
        self.stop_event.set()


# ---- Main window ----
class MeetingCompanionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Offline AI Meeting Companion")
        self.resize(1000, 600)
        self.apply_theme()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header.setObjectName("header")
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(20, 15, 20, 15)

        logo_label = QLabel("🎙")
        logo_label.setObjectName("logo")
        title_label = QLabel("OFFLINE AI MEETING COMPANION")
        title_label.setObjectName("title")

        header_layout.addWidget(logo_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header.setLayout(header_layout)
        layout.addWidget(header)

        self.status_label = QLabel("● Not started")
        self.status_label.setObjectName("status")
        self.status_label.setContentsMargins(20, 8, 20, 8)
        layout.addWidget(self.status_label)

        panes_layout = QHBoxLayout()
        panes_layout.setContentsMargins(20, 10, 20, 10)
        panes_layout.setSpacing(20)

        transcript_col = QVBoxLayout()
        transcript_title = QLabel("LIVE TRANSCRIPT")
        transcript_title.setObjectName("paneTitle")
        transcript_col.addWidget(transcript_title)
        self.transcript_box = QTextEdit()
        self.transcript_box.setReadOnly(True)
        self.transcript_box.setObjectName("pane")
        transcript_col.addWidget(self.transcript_box)

        summary_col = QVBoxLayout()
        summary_title = QLabel("SUMMARY & ACTION ITEMS")
        summary_title.setObjectName("paneTitle")
        summary_col.addWidget(summary_title)
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setObjectName("pane")
        summary_col.addWidget(self.summary_box)

        panes_layout.addLayout(transcript_col)
        panes_layout.addLayout(summary_col)
        layout.addLayout(panes_layout)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(20, 10, 20, 20)
        button_row.setSpacing(15)

        self.start_button = QPushButton("▶  START")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self.start_pipeline)
        button_row.addWidget(self.start_button)

        self.export_button = QPushButton("⬇  EXPORT NOTES")
        self.export_button.setObjectName("exportButton")
        self.export_button.clicked.connect(self.export_notes)
        button_row.addWidget(self.export_button)

        layout.addLayout(button_row)
        self.setLayout(layout)

        self.worker = None

    def apply_theme(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #0a0a0a;
                color: #e0e0e0;
                font-family: 'Segoe UI', sans-serif;
            }
            #header {
                background-color: #0d0d0d;
                border-bottom: 2px solid #00ffcc;
            }
            #logo {
                font-size: 24px;
                margin-right: 10px;
            }
            #title {
                font-size: 16px;
                font-weight: bold;
                letter-spacing: 2px;
                color: #00ffcc;
            }
            #status {
                color: #00ffcc;
                font-size: 12px;
                background-color: #111111;
            }
            #paneTitle {
                color: #00ffcc;
                font-size: 12px;
                font-weight: bold;
                letter-spacing: 1px;
                padding-bottom: 5px;
            }
            #pane {
                background-color: #111111;
                border: 1px solid #222222;
                border-radius: 6px;
                padding: 10px;
                color: #e0e0e0;
                font-size: 13px;
            }
            QPushButton#startButton {
                background-color: #00ffcc;
                color: #0a0a0a;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 6px;
                border: none;
                letter-spacing: 1px;
            }
            QPushButton#startButton:hover {
                background-color: #33ffd9;
            }
            QPushButton#startButton:disabled {
                background-color: #333333;
                color: #666666;
            }
            QPushButton#exportButton {
                background-color: transparent;
                color: #00ffcc;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 6px;
                border: 1px solid #00ffcc;
                letter-spacing: 1px;
            }
            QPushButton#exportButton:hover {
                background-color: #00ffcc22;
            }
        """)

    def start_pipeline(self):
        self.start_button.setEnabled(False)
        self.worker = PipelineWorker()
        self.worker.transcript_updated.connect(self.transcript_box.setPlainText)
        self.worker.summary_updated.connect(self.summary_box.setPlainText)
        self.worker.status_updated.connect(lambda s: self.status_label.setText(f"● {s}"))
        self.worker.start()

    def export_notes(self):
        with open("meeting_notes.md", "w", encoding="utf-8") as f:
            f.write("# Meeting Transcript\n\n")
            f.write(self.transcript_box.toPlainText())
            f.write("\n\n# Summary & Action Items\n\n")
            f.write(self.summary_box.toPlainText())
        self.status_label.setText("● Notes exported to meeting_notes.md")

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MeetingCompanionApp()
    window.show()
    sys.exit(app.exec())