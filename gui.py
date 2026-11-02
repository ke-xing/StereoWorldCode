"""StereoWorld Studio - desktop GUI for the low-VRAM StereoWorld pipeline.

Runs main_lowvram.py in a child process (so a crash or OOM never takes the
GUI down and VRAM is fully released after each job), streams its log, parses
progress markers and keeps a queue of videos.

    python gui.py
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox,
    QSplitter, QStatusBar, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton,
    QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parent
SETTINGS_FILE = ROOT / "gui_settings.json"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".flv"}

MODELS = [
    # (label, path relative to ROOT, HF repo, local dir)
    ("StereoWorld LoRA", "models/stereo.safetensors", "KXingLab/stereoworld", "models"),
    ("Wan2.1 DiT", "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
     "Wan-AI/Wan2.1-T2V-1.3B", "models/Wan-AI/Wan2.1-T2V-1.3B"),
    ("Wan2.1 T5 encoder", "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
     "Wan-AI/Wan2.1-T2V-1.3B", "models/Wan-AI/Wan2.1-T2V-1.3B"),
    ("Wan2.1 VAE", "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
     "Wan-AI/Wan2.1-T2V-1.3B", "models/Wan-AI/Wan2.1-T2V-1.3B"),
    ("VideoLLaMA3-7B (captions)", "models/VideoLLaMA3-7B/config.json",
     "DAMO-NLP-SG/VideoLLaMA3-7B", "models/VideoLLaMA3-7B"),
]

STAGES = {
    "load_video": "Reading video", "caption": "Captioning", "load_models": "Loading models",
    "encode": "Encoding", "denoise": "Generating right eye", "decode": "Decoding",
    "write": "Writing SBS video",
}

STYLE = """
QWidget { font-size: 10pt; }
QMainWindow, QWidget#central { background: #15171c; color: #e6e6e6; }
QGroupBox { border: 1px solid #2c3039; border-radius: 6px; margin-top: 14px; padding: 10px 8px 8px 8px; color: #e6e6e6; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9aa4b2; font-weight: 600; }
QLabel { color: #d0d4da; }
QLabel#hint { color: #8a93a0; font-size: 9pt; }
QLabel#title { font-size: 15pt; font-weight: 700; color: #ffffff; }
QLabel#stage { font-size: 11pt; font-weight: 600; color: #ffffff; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTableWidget {
  background: #1d2027; color: #e6e6e6; border: 1px solid #2f343e; border-radius: 4px; padding: 4px; selection-background-color: #3a6df0; }
QComboBox QAbstractItemView { background: #1d2027; color: #e6e6e6; }
QHeaderView::section { background: #22262e; color: #9aa4b2; border: none; padding: 4px; }
QPushButton, QToolButton { background: #2a2f39; color: #e6e6e6; border: 1px solid #363c48; border-radius: 4px; padding: 6px 12px; }
QPushButton:hover, QToolButton:hover { background: #333946; }
QPushButton:disabled { color: #6b7280; background: #22262d; }
QPushButton#primary { background: #3a6df0; border-color: #3a6df0; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #4b7cf5; }
QPushButton#danger { background: #a83838; border-color: #a83838; color: white; }
QTabWidget::pane { border: 1px solid #2c3039; border-radius: 6px; top: -1px; }
QTabBar::tab { background: #1d2027; color: #9aa4b2; padding: 6px 14px; border: 1px solid #2c3039; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
QTabBar::tab:selected { background: #262a33; color: #ffffff; }
QProgressBar { background: #1d2027; border: 1px solid #2f343e; border-radius: 4px; text-align: center; color: white; height: 18px; }
QProgressBar::chunk { background: #3a6df0; border-radius: 3px; }
QCheckBox { color: #d0d4da; }
QStatusBar { color: #9aa4b2; }
"""


# --------------------------------------------------------------------- model
@dataclass
class Job:
    input: str
    output: str
    prompt: str = ""
    status: str = "queued"  # queued | running | done | failed | stopped
    started: float = 0.0
    message: str = ""


def find_env_python(env: str = "stereoworld") -> str | None:
    """Locate the python of a conda env without relying on `conda activate`."""
    exe = "python.exe" if sys.platform == "win32" else "bin/python"
    roots = []
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        p = Path(prefix)
        roots += [p.parent.parent if p.parent.name == "envs" else p]
    home = Path.home()
    for name in ("miniconda3", "anaconda3", "miniforge3", "mambaforge"):
        roots += [home / name, Path(os.environ.get("LOCALAPPDATA", home)) / name,
                  Path(os.environ.get("ProgramData", "C:/ProgramData")) / name, Path("C:/") / name]
    for r in roots:
        cand = r / "envs" / env / exe
        if cand.exists():
            return str(cand)
    return None


REQUIRED_MODULES = ("torch", "einops", "imageio", "safetensors", "huggingface_hub")


def python_ready(python: str) -> bool:
    """True if `python` has the packages the pipeline and the model download need."""
    code = "import " + ", ".join(REQUIRED_MODULES)
    try:
        return subprocess.run([python, "-c", code], capture_output=True, timeout=90,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode == 0
    except Exception:
        return False


def default_output(inp: str, out_dir: str) -> str:
    p = Path(inp)
    d = Path(out_dir) if out_dir else p.parent
    return str(d / f"{p.stem}_SBS.mp4")


def hint(text: str) -> QLabel:
    l = QLabel(text)
    l.setObjectName("hint")
    l.setWordWrap(True)
    return l


class DropTable(QTableWidget):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__(0, 3)
        self.setAcceptDrops(True)
        self.setHorizontalHeaderLabels(["Input", "Status", "Output"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    dragMoveEvent = dragEnterEvent

    def dropEvent(self, e):
        files = []
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            if p.is_dir():
                files += [str(f) for f in sorted(p.iterdir()) if f.suffix.lower() in VIDEO_EXT]
            elif p.suffix.lower() in VIDEO_EXT:
                files.append(str(p))
        if files:
            self.files_dropped.emit(files)


# ------------------------------------------------------------------- window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StereoWorld Studio")
        self.resize(1280, 860)
        self.jobs: list[Job] = []
        self.current: int | None = None
        self.proc: QProcess | None = None
        self.proc_kind = ""  # "run" | "plan" | "download" | "env"
        self.stop_requested = False
        self.buf = ""
        self.seg_total = 0
        self.step_total = 0

        central = QWidget(objectName="central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 10, 14, 10)

        head = QHBoxLayout()
        t = QLabel("StereoWorld Studio", objectName="title")
        head.addWidget(t)
        head.addStretch()
        self.gpu_label = QLabel("GPU: detecting…")
        head.addWidget(self.gpu_label)
        root.addLayout(head)

        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)
        split.addWidget(self._build_left())
        split.addWidget(self._build_right())
        split.setSizes([560, 720])

        self.setStatusBar(QStatusBar())
        self._load_settings()
        self._refresh_models()
        self._update_mode_widgets()
        self._update_buttons()
        QTimer.singleShot(200, self._detect_gpu)

        self.timer = QTimer(self, interval=1000, timeout=self._tick)
        self.timer.start()

    # ................................................................ layout
    def _build_left(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 6, 0)

        q = QGroupBox("Queue")
        qv = QVBoxLayout(q)
        self.table = DropTable()
        self.table.files_dropped.connect(self.add_files)
        self.table.itemSelectionChanged.connect(self._on_select)
        qv.addWidget(self.table)
        qv.addWidget(hint("Drag videos or folders here. Output is side-by-side: left = original, right = generated eye."))
        row = QHBoxLayout()
        b = QPushButton("Add videos…"); b.clicked.connect(self._browse_inputs); row.addWidget(b)
        b = QPushButton("Remove"); b.clicked.connect(self._remove_selected); row.addWidget(b)
        b = QPushButton("Reset status"); b.clicked.connect(self._reset_selected); row.addWidget(b)
        b = QPushButton("Clear finished"); b.clicked.connect(self._clear_done); row.addWidget(b)
        qv.addLayout(row)
        v.addWidget(q, 2)

        o = QGroupBox("Selected video")
        f = QFormLayout(o)
        self.sel_out = QLineEdit(); self.sel_out.editingFinished.connect(self._sel_changed)
        ob = QToolButton(text="…"); ob.clicked.connect(self._browse_sel_out)
        r = QHBoxLayout(); r.addWidget(self.sel_out); r.addWidget(ob)
        f.addRow("Output file", r)
        self.sel_prompt = QLineEdit(); self.sel_prompt.setPlaceholderText("optional - overrides captioning for this video")
        self.sel_prompt.editingFinished.connect(self._sel_changed)
        f.addRow("Prompt", self.sel_prompt)
        self.out_dir = QLineEdit(); self.out_dir.setPlaceholderText("same folder as input")
        db = QToolButton(text="…"); db.clicked.connect(self._browse_out_dir)
        r = QHBoxLayout(); r.addWidget(self.out_dir); r.addWidget(db)
        f.addRow("Default output folder", r)
        v.addWidget(o)

        m = QGroupBox("Models")
        mg = QGridLayout(m)
        self.model_labels = []
        for i, (name, *_rest) in enumerate(MODELS):
            a = QLabel(name); s = QLabel("…")
            mg.addWidget(a, i, 0); mg.addWidget(s, i, 1)
            self.model_labels.append(s)
        r = QHBoxLayout()
        self.dl_btn = QPushButton("Download missing"); self.dl_btn.clicked.connect(self._download_models)
        rb = QPushButton("Re-check"); rb.clicked.connect(self._refresh_models)
        r.addWidget(self.dl_btn); r.addWidget(rb); r.addStretch()
        mg.addLayout(r, len(MODELS), 0, 1, 2)
        mg.addWidget(hint("~32 GB total. VideoLLaMA3 is only needed when captioning automatically."), len(MODELS) + 1, 0, 1, 2)
        v.addWidget(m)
        return w

    def _build_right(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 0, 0, 0)

        vs = QSplitter(Qt.Vertical)
        v.addWidget(vs)
        top = QWidget(); tv = QVBoxLayout(top); tv.setContentsMargins(0, 0, 0, 0)
        tabs = QTabWidget()
        tabs.addTab(self._tab_basic(), "Basic")
        tabs.addTab(self._tab_vram(), "VRAM / tiling")
        tabs.addTab(self._tab_caption(), "Captioning")
        tabs.addTab(self._tab_env(), "Environment")
        tv.addWidget(tabs)

        p = QGroupBox("Progress")
        pv = QVBoxLayout(p)
        self.stage = QLabel("Idle", objectName="stage")
        pv.addWidget(self.stage)
        self.bar_seg = QProgressBar(); self.bar_seg.setFormat("Segment %v / %m")
        self.bar_step = QProgressBar(); self.bar_step.setFormat("Step %v / %m")
        self.bar_queue = QProgressBar(); self.bar_queue.setFormat("Queue %v / %m")
        for b in (self.bar_step, self.bar_seg, self.bar_queue):
            b.setValue(0); b.setMaximum(1); pv.addWidget(b)
        self.eta = QLabel(""); pv.addWidget(self.eta)
        row = QHBoxLayout()
        self.plan_btn = QPushButton("Show plan"); self.plan_btn.clicked.connect(self.show_plan)
        self.start_btn = QPushButton("Start queue", objectName="primary"); self.start_btn.clicked.connect(self.start_queue)
        self.stop_btn = QPushButton("Stop", objectName="danger"); self.stop_btn.clicked.connect(self.stop)
        self.open_btn = QPushButton("Open output"); self.open_btn.clicked.connect(self._open_output)
        for b in (self.plan_btn, self.start_btn, self.stop_btn, self.open_btn):
            row.addWidget(b)
        pv.addLayout(row)
        tv.addWidget(p)
        vs.addWidget(top)

        lg = QGroupBox("Log")
        lv = QVBoxLayout(lg)
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(20000)
        mono = QFont("Consolas" if sys.platform == "win32" else "Monospace"); mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(9)
        self.log.setFont(mono)
        lv.addWidget(self.log)
        r = QHBoxLayout()
        self.cmd_preview = QLineEdit(readOnly=True); self.cmd_preview.setPlaceholderText("command line")
        cb = QPushButton("Copy command"); cb.clicked.connect(lambda: QApplication.clipboard().setText(self.cmd_preview.text()))
        clr = QPushButton("Clear log"); clr.clicked.connect(self.log.clear)
        r.addWidget(self.cmd_preview, 1); r.addWidget(cb); r.addWidget(clr)
        lv.addLayout(r)
        vs.addWidget(lg)
        vs.setStretchFactor(0, 3)
        vs.setStretchFactor(1, 2)
        return w

    def _tab_basic(self) -> QWidget:
        w = QWidget(); f = QFormLayout(w)
        self.preset = QComboBox(); self.preset.addItems(["auto", "24gb", "16gb", "12gb", "10gb", "8gb", "6gb"])
        f.addRow("VRAM preset", self.preset)
        f.addRow("", hint("auto = use the detected VRAM of the GPU below. Pick a smaller preset if you get out-of-memory errors."))
        self.res_mode = QComboBox(); self.res_mode.addItems(["Source (max 1080p)", "Custom"])
        self.res_h = QSpinBox(); self.res_h.setRange(256, 2160); self.res_h.setSingleStep(16); self.res_h.setValue(480)
        self.res_w = QSpinBox(); self.res_w.setRange(256, 3840); self.res_w.setSingleStep(16); self.res_w.setValue(832)
        self.res_mode.currentIndexChanged.connect(self._update_mode_widgets)
        r = QHBoxLayout(); r.addWidget(self.res_mode); r.addWidget(QLabel("W")); r.addWidget(self.res_w); r.addWidget(QLabel("H")); r.addWidget(self.res_h)
        f.addRow("Processing resolution", r)
        f.addRow("", hint("The model was trained around 480p. Lower resolution is the biggest speed and VRAM lever; sizes are rounded to multiples of 16."))
        self.steps = QSpinBox(); self.steps.setRange(4, 100); self.steps.setValue(50)
        f.addRow("Denoising steps", self.steps)
        self.cfg = QDoubleSpinBox(); self.cfg.setRange(1.0, 15.0); self.cfg.setSingleStep(0.5); self.cfg.setValue(5.0)
        f.addRow("Guidance (CFG)", self.cfg)
        f.addRow("", hint("CFG 1.0 skips the negative pass and halves the time per step, at some quality cost."))
        self.seed = QSpinBox(); self.seed.setRange(0, 2**31 - 1); self.seed.setValue(0)
        f.addRow("Seed", self.seed)
        self.open_when_done = QCheckBox("Open output folder when the queue finishes")
        f.addRow("", self.open_when_done)
        return w

    def _tab_vram(self) -> QWidget:
        w = QWidget(); f = QFormLayout(w)
        self.vram_override = QDoubleSpinBox(); self.vram_override.setRange(0, 96); self.vram_override.setSingleStep(0.5)
        self.vram_override.setSpecialValueText("detect"); self.vram_override.setSuffix(" GB")
        f.addRow("Plan for VRAM", self.vram_override)
        self.max_tokens = QSpinBox(); self.max_tokens.setRange(0, 400000); self.max_tokens.setSingleStep(4000)
        self.max_tokens.setSpecialValueText("auto")
        f.addRow("DiT tokens per call", self.max_tokens)
        f.addRow("", hint("Directly sets the tile budget. Raise it if 'peak alloc' in the log stays well under your VRAM."))

        self.manual_tiles = QCheckBox("Manual spatial tiles"); self.manual_tiles.toggled.connect(self._update_mode_widgets)
        f.addRow(self.manual_tiles)
        self.tile_w = QSpinBox(); self.tile_w.setRange(256, 3840); self.tile_w.setSingleStep(16); self.tile_w.setValue(864)
        self.tile_h = QSpinBox(); self.tile_h.setRange(128, 2160); self.tile_h.setSingleStep(16); self.tile_h.setValue(480)
        r = QHBoxLayout(); r.addWidget(QLabel("W")); r.addWidget(self.tile_w); r.addWidget(QLabel("H")); r.addWidget(self.tile_h)
        f.addRow("Tile size (px)", r)
        self.ov_w = QSpinBox(); self.ov_w.setRange(0, 1024); self.ov_w.setSingleStep(16); self.ov_w.setValue(160)
        self.ov_h = QSpinBox(); self.ov_h.setRange(0, 1024); self.ov_h.setSingleStep(16); self.ov_h.setValue(96)
        r = QHBoxLayout(); r.addWidget(QLabel("W")); r.addWidget(self.ov_w); r.addWidget(QLabel("H")); r.addWidget(self.ov_h)
        f.addRow("Tile overlap (px)", r)
        f.addRow("", hint("Keep tiles wide: stereo disparity is horizontal, narrow tiles lose the content that shifts into view."))

        self.manual_temporal = QCheckBox("Manual temporal chunking"); self.manual_temporal.toggled.connect(self._update_mode_widgets)
        f.addRow(self.manual_temporal)
        self.t_chunk = QComboBox(); self.t_chunk.addItems(["whole segment (81)", "49", "33", "25", "17"])
        f.addRow("Frames per window", self.t_chunk)
        self.t_overlap = QSpinBox(); self.t_overlap.setRange(4, 32); self.t_overlap.setSingleStep(4); self.t_overlap.setValue(16)
        f.addRow("Window overlap (frames)", self.t_overlap)
        f.addRow("", hint("Last resort for very small cards: frames in different windows can't attend to each other."))

        self.vae_tile = QComboBox(); self.vae_tile.addItems(["auto", "416x240", "272x160", "160x96"])
        f.addRow("VAE tile (WxH)", self.vae_tile)
        self.chunk_tokens = QComboBox(); self.chunk_tokens.addItems(["auto", "32768", "16384", "8192", "4096", "off"])
        f.addRow("Attention/FFN chunk", self.chunk_tokens)
        self.vram_limit = QDoubleSpinBox(); self.vram_limit.setRange(0, 96); self.vram_limit.setSingleStep(0.5)
        self.vram_limit.setSpecialValueText("auto"); self.vram_limit.setSuffix(" GB")
        f.addRow("Weight offload limit", self.vram_limit)
        return w

    def _tab_caption(self) -> QWidget:
        w = QWidget(); f = QFormLayout(w)
        self.cap_mode = QComboBox()
        self.cap_mode.addItems(["Automatic (VideoLLaMA3)", "Fixed prompt for all videos"])
        self.cap_mode.currentIndexChanged.connect(self._update_mode_widgets)
        f.addRow("Prompt source", self.cap_mode)
        self.global_prompt = QPlainTextEdit(); self.global_prompt.setFixedHeight(70)
        self.global_prompt.setPlaceholderText("Describe the scene, e.g. 'a crowded night market, handheld camera'")
        f.addRow("Fixed prompt", self.global_prompt)
        f.addRow("", hint("A per-video prompt in the queue always wins over both."))
        self.cap_quant = QComboBox(); self.cap_quant.addItems(["4bit", "8bit", "none"])
        f.addRow("Captioner precision", self.cap_quant)
        f.addRow("", hint("4bit ≈ 6 GB, 8bit ≈ 9 GB, none (bf16) ≈ 16 GB. Needs bitsandbytes for 4/8bit."))
        self.cap_device = QComboBox(); self.cap_device.addItems(["same as GPU", "cpu"])
        f.addRow("Captioner device", self.cap_device)
        self.cap_frames = QSpinBox(); self.cap_frames.setRange(0, 180); self.cap_frames.setSpecialValueText("auto")
        f.addRow("Max frames seen", self.cap_frames)
        return w

    def _tab_env(self) -> QWidget:
        w = QWidget(); f = QFormLayout(w)
        self.python = QLineEdit(sys.executable)
        pb = QToolButton(text="…"); pb.clicked.connect(self._browse_python)
        r = QHBoxLayout(); r.addWidget(self.python); r.addWidget(pb)
        f.addRow("Python", r)
        f.addRow("", hint("Interpreter of the stereoworld environment. Default: the one running this GUI."))
        self.device = QComboBox(); self.device.setEditable(True); self.device.addItems(["cuda:0", "cuda:1"])
        f.addRow("GPU device", self.device)
        self.extra_args = QLineEdit(); self.extra_args.setPlaceholderText("extra command-line flags for main_lowvram.py")
        f.addRow("Extra arguments", self.extra_args)
        r = QHBoxLayout()
        b = QPushButton("Check environment"); b.clicked.connect(self._check_env)
        r.addWidget(b); r.addStretch()
        f.addRow("", r)
        return w

    # .............................................................. settings
    def _settings_widgets(self):
        return {
            "preset": self.preset, "res_mode": self.res_mode, "res_h": self.res_h, "res_w": self.res_w,
            "steps": self.steps, "cfg": self.cfg, "seed": self.seed, "open_when_done": self.open_when_done,
            "vram_override": self.vram_override, "max_tokens": self.max_tokens, "manual_tiles": self.manual_tiles,
            "tile_w": self.tile_w, "tile_h": self.tile_h, "ov_w": self.ov_w, "ov_h": self.ov_h,
            "manual_temporal": self.manual_temporal, "t_chunk": self.t_chunk, "t_overlap": self.t_overlap,
            "vae_tile": self.vae_tile, "chunk_tokens": self.chunk_tokens, "vram_limit": self.vram_limit,
            "cap_mode": self.cap_mode, "global_prompt": self.global_prompt, "cap_quant": self.cap_quant,
            "cap_device": self.cap_device, "cap_frames": self.cap_frames, "python": self.python,
            "device": self.device, "extra_args": self.extra_args, "out_dir": self.out_dir,
        }

    def _save_settings(self):
        data = {}
        for k, w in self._settings_widgets().items():
            if isinstance(w, QComboBox):
                data[k] = w.currentText()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                data[k] = w.value()
            elif isinstance(w, QCheckBox):
                data[k] = w.isChecked()
            elif isinstance(w, QPlainTextEdit):
                data[k] = w.toPlainText()
            elif isinstance(w, QLineEdit):
                data[k] = w.text()
        data["jobs"] = [j.__dict__ for j in self.jobs if j.status != "done"]
        try:
            SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _load_settings(self):
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for k, w in self._settings_widgets().items():
            if k not in data:
                continue
            val = data[k]
            try:
                if isinstance(w, QComboBox):
                    i = w.findText(str(val))
                    if i >= 0:
                        w.setCurrentIndex(i)
                    elif w.isEditable():
                        w.setEditText(str(val))
                elif isinstance(w, QSpinBox):
                    w.setValue(int(val))
                elif isinstance(w, QDoubleSpinBox):
                    w.setValue(float(val))
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(val))
                elif isinstance(w, QPlainTextEdit):
                    w.setPlainText(str(val))
                elif isinstance(w, QLineEdit):
                    w.setText(str(val))
            except (TypeError, ValueError):
                pass
        if self.python.text() and not Path(self.python.text()).exists():
            self.python.setText(sys.executable)
        for jd in data.get("jobs", []):
            j = Job(**{k: jd[k] for k in ("input", "output", "prompt") if k in jd})
            if Path(j.input).exists():
                self.jobs.append(j)
        self._refresh_table()

    def closeEvent(self, e):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            if QMessageBox.question(self, "Job running", "A job is still running. Stop it and quit?") != QMessageBox.Yes:
                e.ignore()
                return
            self.stop_requested = True
            self._kill()
        self._save_settings()
        e.accept()

    # ................................................................. queue
    def add_files(self, files):
        known = {j.input for j in self.jobs}
        for f in files:
            if f not in known:
                self.jobs.append(Job(f, default_output(f, self.out_dir.text().strip())))
        self._refresh_table()
        if self.table.rowCount() and not self.table.selectedItems():
            self.table.selectRow(0)
        self._update_buttons()

    def _browse_inputs(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Add videos", "",
                                                "Videos (" + " ".join("*" + e for e in sorted(VIDEO_EXT)) + ")")
        if files:
            self.add_files(files)

    def _browse_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Default output folder", self.out_dir.text())
        if d:
            self.out_dir.setText(d)
            for j in self.jobs:
                if j.status == "queued":
                    j.output = default_output(j.input, d)
            self._refresh_table()

    def _browse_sel_out(self):
        rows = self._selected_rows()
        if not rows:
            return
        j = self.jobs[rows[0]]
        f, _ = QFileDialog.getSaveFileName(self, "Output file", j.output, "MP4 (*.mp4)")
        if f:
            self.sel_out.setText(f)
            self._sel_changed()

    def _browse_python(self):
        f, _ = QFileDialog.getOpenFileName(self, "Python interpreter", self.python.text())
        if f:
            self.python.setText(f)

    def _selected_rows(self):
        return sorted({i.row() for i in self.table.selectedIndexes()})

    def _on_select(self):
        rows = self._selected_rows()
        if rows:
            j = self.jobs[rows[0]]
            self.sel_out.setText(j.output)
            self.sel_prompt.setText(j.prompt)

    def _sel_changed(self):
        for r in self._selected_rows()[:1]:
            j = self.jobs[r]
            j.output = self.sel_out.text().strip() or j.output
            j.prompt = self.sel_prompt.text().strip()
        self._refresh_table()

    def _remove_selected(self):
        for r in reversed(self._selected_rows()):
            if r != self.current:
                del self.jobs[r]
        self._refresh_table()
        self._update_buttons()

    def _reset_selected(self):
        for r in self._selected_rows():
            if r != self.current:
                self.jobs[r].status, self.jobs[r].message = "queued", ""
        self._refresh_table()
        self._update_buttons()

    def _clear_done(self):
        self.jobs = [j for i, j in enumerate(self.jobs) if j.status != "done" or i == self.current]
        self._refresh_table()

    def _refresh_table(self):
        colors = {"queued": "#9aa4b2", "running": "#6ea8ff", "done": "#5bd18b", "failed": "#ff6b6b", "stopped": "#e5b454"}
        sel = self._selected_rows()
        self.table.setRowCount(len(self.jobs))
        for i, j in enumerate(self.jobs):
            items = [QTableWidgetItem(Path(j.input).name), QTableWidgetItem(j.status + (f" - {j.message}" if j.message else "")),
                     QTableWidgetItem(j.output)]
            items[0].setToolTip(j.input)
            items[1].setForeground(QColor(colors.get(j.status, "#e6e6e6")))
            if j.prompt:
                items[0].setText(Path(j.input).name + "  ✎")
            for c, it in enumerate(items):
                self.table.setItem(i, c, it)
        for r in sel:
            if r < self.table.rowCount():
                self.table.selectRow(r)

    # .............................................................. commands
    def _base_args(self, job: Job | None) -> list[str]:
        a = ["main_lowvram.py"]
        if job:
            a += ["--video_path", job.input, "--output_path", job.output]
        a += ["--device", self.device.currentText().strip() or "cuda:0", "--preset", self.preset.currentText()]
        if self.vram_override.value() > 0:
            a += ["--vram-gb", f"{self.vram_override.value():g}"]
        if self.max_tokens.value() > 0:
            a += ["--max-tokens", str(self.max_tokens.value())]
        if self.res_mode.currentIndex() == 1:
            a += ["--resolution", f"{self.res_h.value() // 16 * 16}x{self.res_w.value() // 16 * 16}"]
        if self.manual_tiles.isChecked():
            a += ["--tile-size", f"{self.tile_h.value() // 16 * 16}x{self.tile_w.value() // 16 * 16}",
                  "--tile-overlap", f"{self.ov_h.value() // 16 * 16}x{self.ov_w.value() // 16 * 16}"]
        if self.manual_temporal.isChecked():
            t = self.t_chunk.currentText().split()[0]
            a += ["--temporal-chunk", "0" if t == "whole" else t, "--temporal-overlap", str(self.t_overlap.value())]
        if self.vae_tile.currentText() != "auto":
            vw, vh = self.vae_tile.currentText().split("x")
            a += ["--vae-tile", f"{vh}x{vw}"]
        ct = self.chunk_tokens.currentText()
        if ct != "auto":
            a += ["--chunk-tokens", "0" if ct == "off" else ct]
        if self.vram_limit.value() > 0:
            a += ["--vram-limit", f"{self.vram_limit.value():g}"]
        prompt = (job.prompt if job and job.prompt else "") or (
            self.global_prompt.toPlainText().strip() if self.cap_mode.currentIndex() == 1 else "")
        if prompt:
            a += ["--prompt", prompt]
        else:
            a += ["--caption-quant", self.cap_quant.currentText()]
            if self.cap_device.currentText() == "cpu":
                a += ["--caption-device", "cpu"]
            if self.cap_frames.value() > 0:
                a += ["--caption-max-frames", str(self.cap_frames.value())]
        a += ["--steps", str(self.steps.value()), "--cfg", f"{self.cfg.value():g}", "--seed", str(self.seed.value())]
        extra = self.extra_args.text().strip()
        if extra:
            a += shlex.split(extra, posix=(sys.platform != "win32"))
        return a

    def _cmdline(self, args):
        exe = self.python.text().strip() or sys.executable
        return subprocess.list2cmdline([exe, *args]) if sys.platform == "win32" else shlex.join([exe, *args])

    def _launch(self, args: list[str], kind: str):
        exe = self.python.text().strip() or sys.executable
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(ROOT))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("STEREOWORLD_PROGRESS", "1")
        env.insert("PYTORCH_CUDA_ALLOC_CONF", env.value("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"))
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(self._proc_error)
        self.proc_kind = kind
        self.buf = ""
        self.cmd_preview.setText(self._cmdline(args))
        self._append(f"\n$ {self._cmdline(args)}\n", "#6ea8ff")
        self.proc.start(exe, args)
        self._update_buttons()

    def show_plan(self):
        rows = self._selected_rows()
        job = self.jobs[rows[0]] if rows else next((j for j in self.jobs), None)
        if not job:
            QMessageBox.information(self, "Plan", "Add a video first.")
            return
        self.stage.setText("Planning…")
        self._launch(self._base_args(job) + ["--dry-run"], "plan")

    def start_queue(self):
        missing = [MODELS[i][0] for i in self._missing_models(required_only=True)]
        if missing:
            if QMessageBox.question(self, "Models missing",
                                    "Missing: " + ", ".join(missing) + ".\nStart anyway?") != QMessageBox.Yes:
                return
        self.stop_requested = False
        self._save_settings()
        self._next_job()

    def _next_job(self):
        idx = next((i for i, j in enumerate(self.jobs) if j.status == "queued"), None)
        total = len(self.jobs)
        done = sum(j.status in ("done", "failed", "stopped") for j in self.jobs)
        self.bar_queue.setMaximum(max(1, total)); self.bar_queue.setValue(done)
        if idx is None or self.stop_requested:
            self.current = None
            self.stage.setText("Queue finished" if idx is None else "Stopped")
            self.eta.setText("")
            self._update_buttons()
            if idx is None and self.open_when_done.isChecked():
                outs = [j.output for j in self.jobs if j.status == "done"]
                if outs:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(outs[-1]).parent)))
            return
        j = self.jobs[idx]
        Path(j.output).parent.mkdir(parents=True, exist_ok=True)
        self.current = idx
        j.status, j.message, j.started = "running", "", time.time()
        self.seg_total = self.step_total = 0
        for b in (self.bar_seg, self.bar_step):
            b.setMaximum(1); b.setValue(0)
        self._refresh_table()
        self.table.selectRow(idx)
        self._launch(self._base_args(j), "run")

    def stop(self):
        if not self.proc or self.proc.state() == QProcess.NotRunning:
            return
        self.stop_requested = True
        self._append("\n[gui] stopping…\n", "#e5b454")
        self._kill()

    def _kill(self):
        if not self.proc:
            return
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.processId())],
                           capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            self.proc.terminate()
            if not self.proc.waitForFinished(5000):
                self.proc.kill()

    # ................................................................ output
    def _append(self, text, color=None):
        cur = self.log.textCursor()
        cur.movePosition(QTextCursor.End)
        fmt = cur.charFormat()
        fmt.setForeground(QColor(color or "#d0d4da"))
        cur.setCharFormat(fmt)
        cur.insertText(text)
        self.log.setTextCursor(cur)
        self.log.ensureCursorVisible()

    def _read(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.buf += data.replace("\r\n", "\n")
        # tqdm uses \r for in-place updates: keep only the last state of such lines
        lines = self.buf.split("\n")
        self.buf = lines.pop()
        for line in lines:
            self._handle_line(line.split("\r")[-1])
        if "\r" in self.buf:
            last = self.buf.split("\r")[-1]
            self.statusBar().showMessage(last.strip()[:200])
            self.buf = last

    def _handle_line(self, line: str):
        if line.startswith("@@"):
            kind, _, rest = line[2:].partition(" ")
            vals = rest.split()
            if kind == "stage":
                self.stage.setText(STAGES.get(vals[0], vals[0]) + "…")
            elif kind == "progress" and len(vals) == 4:
                s, S, k, K = map(int, vals)
                self.seg_total, self.step_total = S, K
                self.bar_seg.setMaximum(S); self.bar_seg.setValue(s - 1 if k < K else s)
                self.bar_step.setMaximum(K); self.bar_step.setValue(k)
                self._eta(s, S, k, K)
            elif kind == "caption" and len(vals) == 2:
                self.bar_seg.setMaximum(int(vals[1])); self.bar_seg.setValue(int(vals[0]))
            return
        color = None
        low = line.lower()
        if "error" in low or "traceback" in low or "outofmemory" in low or "out of memory" in low:
            color = "#ff6b6b"
        elif line.startswith("[plan]"):
            color = "#9ad1ff"
        elif line.startswith("[lowvram]") or line.startswith("[done]"):
            color = "#5bd18b"
        elif line.startswith("[caption"):
            color = "#c9a8ff"
        self._append(line + "\n", color)
        m = re.search(r"peak alloc ([\d.]+) GiB", line)
        if m:
            self.statusBar().showMessage(f"Last segment peak VRAM: {m.group(1)} GiB", 15000)

    def _eta(self, s, S, k, K):
        if self.current is None:
            return
        j = self.jobs[self.current]
        frac = ((s - 1) * K + k) / max(1, S * K)
        el = time.time() - j.started
        if frac > 0.01:
            rem = el / frac - el
            self.eta.setText(f"Elapsed {self._fmt(el)}  ·  remaining ≈ {self._fmt(rem)} (generation only)")

    @staticmethod
    def _fmt(sec):
        sec = int(sec)
        return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"

    def _tick(self):
        if self.current is not None and not self.step_total:
            el = time.time() - self.jobs[self.current].started
            self.eta.setText(f"Elapsed {self._fmt(el)}")

    def _proc_error(self, err):
        if err == QProcess.FailedToStart:
            self._append(f"[gui] could not start {self.python.text()}\n", "#ff6b6b")

    def _finished(self, code, status):
        if self.buf:
            self._handle_line(self.buf)
            self.buf = ""
        kind = self.proc_kind
        self.proc_kind = ""
        if kind == "run" and self.current is not None:
            j = self.jobs[self.current]
            if self.stop_requested:
                j.status = "stopped"
            elif code == 0 and Path(j.output).exists():
                j.status, j.message = "done", self._fmt(time.time() - j.started)
            else:
                j.status = "failed"
                tail = self.log.toPlainText()[-4000:]
                j.message = "out of memory" if ("OutOfMemory" in tail or "out of memory" in tail) else f"exit {code}"
                if j.message == "out of memory":
                    self._append("[gui] OOM - try a smaller VRAM preset, lower resolution or 4bit captioning.\n", "#e5b454")
            self._refresh_table()
            self._save_settings()
            self.current = None
            self._next_job()
            return
        if kind == "plan":
            self.stage.setText("Plan ready" if code == 0 else "Plan failed")
        elif kind == "download":
            self.stage.setText("Download finished" if code == 0 else "Download failed")
            self._refresh_models()
        elif kind == "env":
            self.stage.setText("Environment check finished")
        self._update_buttons()

    # ................................................................ models
    def _missing_models(self, required_only=False):
        need_caption = self.cap_mode.currentIndex() == 0 and not all(j.prompt for j in self.jobs if j.status == "queued")
        out = []
        for i, (name, rel, *_r) in enumerate(MODELS):
            if required_only and "VideoLLaMA" in name and not need_caption:
                continue
            if not (ROOT / rel).exists():
                out.append(i)
        return out

    def _refresh_models(self):
        missing = set(self._missing_models())
        for i, lab in enumerate(self.model_labels):
            ok = i not in missing
            lab.setText("✔ found" if ok else "✖ missing")
            lab.setStyleSheet(f"color: {'#5bd18b' if ok else '#ff6b6b'};")
        self.dl_btn.setEnabled(bool(missing) and not self._busy())

    def _download_models(self):
        repos = []
        for i in self._missing_models():
            _, _, repo, local = MODELS[i]
            if (repo, local) not in repos:
                repos.append((repo, local))
        if not repos:
            return
        code = ("import sys\nfrom huggingface_hub import snapshot_download\n"
                "for repo, local in " + repr(repos) + ":\n"
                "    print(f'downloading {repo} -> {local}', flush=True)\n"
                "    snapshot_download(repo_id=repo, local_dir=local)\n"
                "print('[done] models downloaded', flush=True)\n")
        self.stage.setText("Downloading models…")
        self._launch(["-c", code], "download")

    def _check_env(self):
        code = r"""
import importlib, platform, shutil, sys
print('python', sys.version.split()[0], platform.platform())
try:
    import torch
    print('torch', torch.__version__, '| CUDA build', torch.version.cuda)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f'  cuda:{i} {p.name} {p.total_memory/2**30:.1f} GB')
    else:
        print('  (!) CUDA not available - install the CUDA build of PyTorch')
except Exception as e:
    print('(!) torch:', e)
for m in ['einops','transformers','safetensors','imageio','cv2','modelscope','moviepy','diffusers','peft','bitsandbytes','flash_attn','huggingface_hub']:
    try:
        mod = importlib.import_module(m); print(f'{m:16s} ok', getattr(mod, '__version__', ''))
    except Exception as e:
        print(f'{m:16s} MISSING ({type(e).__name__})')
print('ffmpeg          ', shutil.which('ffmpeg') or 'MISSING')
try:
    import diffsynthfinal.pipelines.wan_video_new  # noqa
    print('diffsynthfinal   ok')
except Exception as e:
    print('(!) diffsynthfinal import failed:', e)
"""
        self.stage.setText("Checking environment…")
        self._launch(["-c", code], "env")

    def _detect_gpu(self):
        exe = self.python.text().strip() or sys.executable
        env_py = find_env_python()
        same = env_py and Path(env_py).resolve() == Path(exe).resolve()
        if env_py and not same and not python_ready(exe):
            # configured Python lacks the pipeline packages, the env exists: use it
            self._append(f"[gui] {exe} is missing StereoWorld packages - switching to {env_py}\n", "#e5b454")
            self.python.setText(env_py)
            self._save_settings()
            exe = env_py
        if not python_ready(exe):
            where = exe if (env_py and Path(env_py).resolve() == Path(exe).resolve()) else f"{exe} (no stereoworld env found)"
            self._append(f"[gui] (!) {where} is missing packages (torch with CUDA, einops, imageio, safetensors, "
                         "huggingface_hub). Install the requirements into the stereoworld env, then press "
                         "'Check environment'.\n", "#ff6b6b")
        code = ("import torch\nif torch.cuda.is_available():\n p=torch.cuda.get_device_properties(0)\n"
                " print(f'{p.name} · {p.total_memory/2**30:.1f} GB')\nelse:\n print('no CUDA GPU')")
        try:
            out = subprocess.run([exe, "-c", code], capture_output=True, text=True, timeout=60, cwd=str(ROOT),
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        except Exception:
            out = ""
        self.gpu_label.setText("GPU: " + (out.splitlines()[-1] if out else "unknown (check Environment tab)"))

    # ................................................................... ui
    def _busy(self):
        return bool(self.proc and self.proc.state() != QProcess.NotRunning)

    def _update_buttons(self):
        busy = self._busy()
        has_q = any(j.status == "queued" for j in self.jobs)
        self.start_btn.setEnabled(not busy and has_q)
        self.plan_btn.setEnabled(not busy and bool(self.jobs))
        self.stop_btn.setEnabled(busy and self.proc_kind in ("run", "download"))
        self.open_btn.setEnabled(any(Path(j.output).exists() for j in self.jobs))
        self.dl_btn.setEnabled(not busy and bool(self._missing_models()))

    def _update_mode_widgets(self):
        custom = self.res_mode.currentIndex() == 1
        self.res_w.setEnabled(custom); self.res_h.setEnabled(custom)
        t = self.manual_tiles.isChecked()
        for w in (self.tile_w, self.tile_h, self.ov_w, self.ov_h):
            w.setEnabled(t)
        tm = self.manual_temporal.isChecked()
        self.t_chunk.setEnabled(tm); self.t_overlap.setEnabled(tm)
        fixed = self.cap_mode.currentIndex() == 1
        self.global_prompt.setEnabled(fixed)
        for w in (self.cap_quant, self.cap_device, self.cap_frames):
            w.setEnabled(not fixed)

    def _open_output(self):
        rows = self._selected_rows()
        cand = [self.jobs[r] for r in rows] + list(reversed(self.jobs))
        for j in cand:
            if Path(j.output).exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(j.output))
                return


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("StereoWorld Studio")
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.add_files([a for a in sys.argv[1:] if Path(a).suffix.lower() in VIDEO_EXT])
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
