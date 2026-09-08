"""FaceBlur window.

Built to Section 7 of the build plan. The apple-design skill is not installed in
this environment, so its rule list was applied by hand:

- Every clickable control is at least 24 px tall. The primary button is 36 px.
- Body text uses the palette text colour, so contrast holds in light and dark.
  Help text is smaller, never fainter.
- No information is carried by colour alone. Every status shows a mark and a word.
- Sentence case everywhere. See ui/strings.py.
- One primary action. Only "Blur faces" carries the accent colour.
- The empty, running, stopped, failed and finished states are all drawn.
- Stop asks first. Replacing existing files is an explicit opt in.
- The window resizes. The drop zone and the progress list grow. Buttons do not.
- Every control is reachable by keyboard and keeps its focus ring.
- Spacing sits on an 8 px grid. Sections 24, controls 8, window margin 16.
"""
from __future__ import annotations

import multiprocessing
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import (QEvent, QSettings, QSize, Qt, QThread, QUrl, Signal)
from PySide6.QtGui import QAction, QDesktopServices, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QFileDialog,
    QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSpinBox,
    QToolButton, QVBoxLayout, QWidget,
)

if __package__ in (None, ""):  # started as `python ui/app.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faceblur.pipeline import (STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED,
                               STATUS_STOPPED, output_path, sidecar_path)
from faceblur.settings import VIDEO_EXT, Settings
from faceblur.video import part_path
from ui import strings as S
from faceblur.batch import PoolSubmit, init_pool_worker, run_video

GRID = 8
MARGIN = 2 * GRID          # 16
SECTION = 3 * GRID         # 24
PRIMARY_HEIGHT = 36
CONTROL_HEIGHT = 28

# The one accent in the window. White text on this blue clears 4.5:1 in both
# palettes, which is why the accent is fixed rather than taken from the palette.
ACCENT = "#1a56db"
ACCENT_HOVER = "#1749b8"
ACCENT_DOWN = "#133c96"
ACCENT_DISABLED = "#9aa4b2"

STATUS_FOR_RECORD = {
    STATUS_DONE: S.STATUS_DONE,
    STATUS_FAILED: S.STATUS_FAILED,
    STATUS_SKIPPED: S.STATUS_SKIPPED,
    STATUS_STOPPED: S.STATUS_STOPPED,
}


def bar_style(palette) -> str:
    """A progress bar that stays visible at zero.

    The platform bar draws an empty groove that reads as a hairline rule against
    a dark window, so a waiting row looks like a divider rather than a bar.
    """
    edge = palette.windowText().color()
    edge.setAlpha(110)
    return (
        "QProgressBar {"
        f" border: 1px solid rgba({edge.red()},{edge.green()},{edge.blue()},{edge.alpha()});"
        f" border-radius: {GRID // 2}px; background: transparent; }}"
        f"QProgressBar::chunk {{ background-color: {ACCENT};"
        f" border-radius: {GRID // 2 - 1}px; }}"
    )


def help_label(text: str) -> QLabel:
    """One line of help under a control. Smaller than body text, never fainter."""
    label = QLabel(text)
    label.setWordWrap(True)
    font = label.font()
    font.setPointSizeF(max(7.5, font.pointSizeF() - 1.0))
    label.setFont(font)
    return label


class BatchRunner(QThread):
    """Runs the batch: one video at a time, its phases spread over a pool.

    The phases (detection in frame ranges, segment encodes) go to worker
    processes through faceblur.batch.PoolSubmit. Tracking and the join run
    here. Progress arrives per finished range or segment.
    """

    progress = Signal(int, str, int, int)   # index, stage, done, total
    one_finished = Signal(int, dict)        # index, audit record
    all_finished = Signal(bool)             # stopped by the user

    def __init__(self, jobs, settings: Settings, workers: int, parent=None):
        super().__init__(parent)
        self.jobs = jobs                    # list of (src Path, dst Path)
        self.settings = settings
        self.workers = max(1, workers)
        self._stop_wanted = False
        self._cancel = threading.Event()

    def stop(self) -> None:
        self._stop_wanted = True
        self._cancel.set()

    def run(self) -> None:  # runs in the worker thread, never touches widgets
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(self.workers, initializer=init_pool_worker, initargs=(self.workers,))
        submit = PoolSubmit(pool)
        try:
            for index, (src, dst) in enumerate(self.jobs):
                if self._stop_wanted:
                    break

                def on_progress(stage, done, total, index=index):
                    self.progress.emit(index, stage, done, total)

                try:
                    record = run_video(Path(src), Path(dst), self.settings, submit,
                                       on_progress=on_progress, cancel=self._cancel,
                                       workers=self.workers).to_dict()
                except Exception as exc:  # a worker died, report it as a failure
                    record = {"status": STATUS_FAILED, "error": str(exc)}
                if self._stop_wanted and record.get("status") != STATUS_DONE:
                    record["status"] = STATUS_STOPPED
                self.one_finished.emit(index, record)
        finally:
            if self._stop_wanted:
                pool.terminate()
                self._remove_partial_files()
            else:
                pool.close()
            pool.join()
        self.all_finished.emit(self._stop_wanted)

    def _remove_partial_files(self) -> None:
        """Stop never leaves half a video, or its segment folder, behind."""
        import shutil
        for _, dst in self.jobs:
            dst = Path(dst)
            if dst.exists() and not sidecar_path(dst).exists():
                dst.unlink(missing_ok=True)
            part_path(dst).unlink(missing_ok=True)
        for folder in {Path(dst).parent for _, dst in self.jobs}:
            for temp in folder.glob("faceblur_*"):
                if temp.is_dir():
                    shutil.rmtree(temp, ignore_errors=True)


class DropZone(QFrame):
    """Large dashed region. Accepts one or more videos, or one folder."""

    dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAccessibleName(S.ACC_DROP_ZONE)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._hover = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SECTION, SECTION, SECTION, SECTION)
        layout.setSpacing(GRID)
        layout.addStretch(1)

        self.title = QLabel(S.DROP_ZONE_EMPTY)
        self.title.setAlignment(Qt.AlignCenter)
        font = self.title.font()
        font.setPointSizeF(font.pointSizeF() + 2.0)
        self.title.setFont(font)
        layout.addWidget(self.title)

        self.hint = QLabel(S.DROP_ZONE_HINT)
        self.hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.hint)
        layout.addStretch(1)
        self._paint()

    def _paint(self) -> None:
        # The border colour comes from the text colour, not from a mid grey.
        # A mid grey vanishes against a dark window, and the dashed edge is the
        # only thing that says this region accepts a drop.
        if self._hover:
            colour = self.palette().highlight().color()
            width = 2
        else:
            colour = self.palette().windowText().color()
            colour.setAlpha(130)
            width = 1
        self.setStyleSheet(
            f"DropZone {{ border: {width}px dashed rgba"
            f"({colour.red()},{colour.green()},{colour.blue()},{colour.alpha()});"
            f" border-radius: {GRID}px; }}"
        )

    def show_summary(self, text: str) -> None:
        self.title.setText(text)
        self.hint.setText(S.DROP_ZONE_HINT)

    def show_empty(self) -> None:
        self.title.setText(S.DROP_ZONE_EMPTY)
        self.hint.setText(S.DROP_ZONE_HINT)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._hover = True
            self._paint()

    def dragLeaveEvent(self, event) -> None:
        self._hover = False
        self._paint()

    def dropEvent(self, event) -> None:
        self._hover = False
        self._paint()
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()
                 if url.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.dropped.emit(paths)

    def changeEvent(self, event) -> None:
        if event.type() == QEvent.PaletteChange:
            self._paint()
        super().changeEvent(event)


class FileRow:
    """One row of the progress list: name, bar, status mark and word."""

    def __init__(self, grid: QGridLayout, row: int, name: str):
        self.name = name
        self.name_label = QLabel(name)
        self.name_label.setToolTip(name)
        self.name_label.setMinimumWidth(120)
        self.name_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(GRID + 4)
        self.bar.setStyleSheet(bar_style(self.bar.palette()))
        self.bar.setAccessibleName(S.ACC_FILE_BAR.format(name=name))

        self.status = QLabel(f"{S.STATUS_MARK[S.STATUS_WAITING]}  {S.STATUS_WAITING}")
        self.status.setMinimumWidth(112)

        grid.addWidget(self.name_label, row, 0)
        grid.addWidget(self.bar, row, 1)
        grid.addWidget(self.status, row, 2)

    def set_status(self, word: str, tooltip: str = "") -> None:
        self.status.setText(f"{S.STATUS_MARK[word]}  {word}")
        self.status.setToolTip(tooltip)
        self.name_label.setToolTip(tooltip or self.name)

    def set_fraction(self, fraction: float) -> None:
        self.bar.setValue(int(round(100 * max(0.0, min(1.0, fraction)))))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(S.WINDOW_TITLE)
        self.resize(760, 720)
        self.setMinimumSize(QSize(560, 520))

        self.settings_store = QSettings("FaceBlur", "FaceBlur")
        self.inputs: list[Path] = []
        self.output_dir: Path | None = None
        self.rows: list[FileRow] = []
        self.jobs: list[tuple[Path, Path]] = []
        self.runner: BatchRunner | None = None
        self.records: dict[int, dict] = {}
        self.started_at = 0.0
        self.bytes_done = 0
        self.bytes_total = 0

        self._build()
        self._restore()
        self._update_start_enabled()

    # ---------------------------------------------------------------- building

    def _build(self) -> None:
        # The whole page sits in one scroll area. Without it, opening advanced
        # settings on a short window squeezes the wrapped help labels until they
        # draw on top of each other.
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QFrame.NoFrame)
        self.setCentralWidget(page)

        central = QWidget()
        page.setWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        outer.setSpacing(SECTION)

        outer.addWidget(self._build_input(), 3)
        outer.addWidget(self._build_output())
        outer.addWidget(self._build_advanced())
        outer.addLayout(self._build_start())
        outer.addWidget(self._build_progress(), 4)
        outer.addLayout(self._build_finish())

        about = QAction(S.MENU_ABOUT, self)
        about.triggered.connect(self._show_about)
        self.menuBar().addMenu(S.APP_NAME).addAction(about)

        self._add_shortcut(QKeySequence("Ctrl+O"), self._choose_files)
        self._add_shortcut(QKeySequence("Ctrl+Shift+O"), self._choose_folder)
        self._add_shortcut(QKeySequence("Ctrl+Return"), self._start_or_stop)
        self._add_shortcut(QKeySequence("Ctrl+Enter"), self._start_or_stop)
        self._add_shortcut(QKeySequence("Esc"), self._escape)

    def _add_shortcut(self, sequence: QKeySequence, slot) -> None:
        action = QAction(self)
        action.setShortcut(sequence)
        action.triggered.connect(slot)
        self.addAction(action)

    def _build_input(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(GRID)

        self.drop_zone = DropZone()
        self.drop_zone.dropped.connect(self._take_paths)
        layout.addWidget(self.drop_zone, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(GRID)
        self.choose_files_button = QPushButton(S.CHOOSE_FILES)
        self.choose_files_button.setMinimumHeight(CONTROL_HEIGHT)
        self.choose_files_button.clicked.connect(self._choose_files)
        self.choose_folder_button = QPushButton(S.CHOOSE_FOLDER)
        self.choose_folder_button.setMinimumHeight(CONTROL_HEIGHT)
        self.choose_folder_button.clicked.connect(self._choose_folder)
        buttons.addWidget(self.choose_files_button)
        buttons.addWidget(self.choose_folder_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return box

    def _build_output(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(GRID)

        row = QHBoxLayout()
        row.setSpacing(GRID)
        label = QLabel(S.OUTPUT_LABEL)
        label.setMinimumWidth(96)
        self.output_value = QLabel(S.OUTPUT_EMPTY)
        self.output_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.output_value.setFrameShape(QFrame.StyledPanel)
        self.output_value.setMinimumHeight(CONTROL_HEIGHT)
        self.output_value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.output_value.setContentsMargins(GRID, 0, GRID, 0)
        self.choose_output_button = QPushButton(S.CHOOSE_OUTPUT)
        self.choose_output_button.setMinimumHeight(CONTROL_HEIGHT)
        self.choose_output_button.clicked.connect(self._choose_output)

        row.addWidget(label)
        row.addWidget(self.output_value, 1)
        row.addWidget(self.choose_output_button)
        layout.addLayout(row)
        layout.addWidget(help_label(S.OUTPUT_HELP))
        return box

    def _build_advanced(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(GRID)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText(S.ADVANCED)
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        self.advanced_toggle.setAutoRaise(True)
        # A checked tool button paints itself in the platform highlight, which
        # puts a second accent on the screen. Only "Blur faces" carries accent.
        self.advanced_toggle.setStyleSheet(
            "QToolButton { background: transparent; border: none;"
            f" padding: {GRID // 2}px {GRID}px; }}"
            "QToolButton:checked { background: transparent; }"
            "QToolButton:hover { text-decoration: underline; }"
        )
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setMinimumHeight(CONTROL_HEIGHT)
        self.advanced_toggle.setAccessibleName(S.ACC_ADVANCED_TOGGLE)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.advanced_toggle, 0, Qt.AlignLeft)

        self.advanced_body = QWidget()
        body = QVBoxLayout(self.advanced_body)
        body.setContentsMargins(SECTION, 0, 0, 0)
        body.setSpacing(SECTION)

        # Detector
        engine = QVBoxLayout()
        engine.setSpacing(GRID // 2)
        engine.addWidget(QLabel(S.ENGINE_LABEL))
        self.engine_standard = QRadioButton(S.ENGINE_STANDARD)
        self.engine_standard.setChecked(True)
        self.engine_max = QRadioButton(S.ENGINE_MAX)
        self.engine_group = QButtonGroup(self)
        self.engine_group.addButton(self.engine_standard)
        self.engine_group.addButton(self.engine_max)
        for button, text in ((self.engine_standard, S.ENGINE_STANDARD_HELP),
                             (self.engine_max, S.ENGINE_MAX_HELP)):
            button.setMinimumHeight(CONTROL_HEIGHT - 4)
            engine.addWidget(button)
            engine.addWidget(help_label(text))
        body.addLayout(engine)

        # Redaction style
        mode = QVBoxLayout()
        mode.setSpacing(GRID // 2)
        mode.addWidget(QLabel(S.MODE_LABEL))
        self.mode_blur = QRadioButton(S.MODE_BLUR)
        self.mode_blur.setChecked(True)
        self.mode_pixelate = QRadioButton(S.MODE_PIXELATE)
        self.mode_solid = QRadioButton(S.MODE_SOLID)
        self.mode_group = QButtonGroup(self)
        for button, text in ((self.mode_blur, S.MODE_BLUR_HELP),
                             (self.mode_pixelate, S.MODE_PIXELATE_HELP),
                             (self.mode_solid, S.MODE_SOLID_HELP)):
            button.setMinimumHeight(CONTROL_HEIGHT - 4)
            self.mode_group.addButton(button)
            mode.addWidget(button)
            mode.addWidget(help_label(text))
        body.addLayout(mode)

        # Stride and workers
        numbers = QGridLayout()
        numbers.setHorizontalSpacing(GRID)
        numbers.setVerticalSpacing(GRID // 2)

        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 30)
        self.stride_spin.setValue(1)
        self.stride_spin.setMinimumHeight(CONTROL_HEIGHT)
        numbers.addWidget(QLabel(S.STRIDE_LABEL), 0, 0)
        numbers.addWidget(self.stride_spin, 0, 1)
        numbers.addWidget(help_label(S.STRIDE_HELP), 1, 0, 1, 3)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, max(1, multiprocessing.cpu_count()))
        from faceblur.detect import default_workers
        self.workers_spin.setValue(default_workers())
        self.workers_spin.setMinimumHeight(CONTROL_HEIGHT)
        numbers.addWidget(QLabel(S.WORKERS_LABEL), 2, 0)
        numbers.addWidget(self.workers_spin, 2, 1)
        numbers.addWidget(help_label(S.WORKERS_HELP), 3, 0, 1, 3)
        numbers.setColumnStretch(2, 1)
        body.addLayout(numbers)

        # Replace existing
        replace = QVBoxLayout()
        replace.setSpacing(GRID // 2)
        self.replace_check = QCheckBox(S.REPLACE_EXISTING)
        self.replace_check.setMinimumHeight(CONTROL_HEIGHT - 4)
        replace.addWidget(self.replace_check)
        replace.addWidget(help_label(S.REPLACE_EXISTING_HELP))
        body.addLayout(replace)

        self.advanced_body.setVisible(False)
        layout.addWidget(self.advanced_body)
        return box

    def _build_start(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.start_help = help_label(S.START_HELP)
        row.addWidget(self.start_help, 1)
        self.start_button = QPushButton(S.START)
        self.start_button.setMinimumHeight(PRIMARY_HEIGHT)
        self.start_button.setMinimumWidth(140)
        self.start_button.setDefault(True)
        self.start_button.setStyleSheet(
            "QPushButton {"
            f" background-color: {ACCENT}; color: #ffffff;"
            " border: 2px solid transparent;"
            f" border-radius: {GRID // 2}px; padding: 0 {SECTION}px;"
            " font-weight: 600; }"
            f"QPushButton:hover {{ background-color: {ACCENT_HOVER}; }}"
            f"QPushButton:pressed {{ background-color: {ACCENT_DOWN}; }}"
            f"QPushButton:disabled {{ background-color: {ACCENT_DISABLED}; }}"
            "QPushButton:focus { outline: none; border-color: #ffffff; }"
        )
        self.start_button.clicked.connect(self._start_or_stop)
        row.addWidget(self.start_button, 0, Qt.AlignRight)
        return row

    def _build_progress(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(GRID)

        heading = QLabel(S.PROGRESS_HEADING)
        font = heading.font()
        font.setBold(True)
        heading.setFont(font)
        layout.addWidget(heading)

        self.overall_bar = QProgressBar()
        self.overall_bar.setRange(0, 100)
        self.overall_bar.setValue(0)
        self.overall_bar.setTextVisible(False)
        self.overall_bar.setFixedHeight(GRID + 6)
        self.overall_bar.setStyleSheet(bar_style(self.overall_bar.palette()))
        self.overall_bar.setAccessibleName(S.ACC_OVERALL_BAR)
        layout.addWidget(self.overall_bar)

        self.overall_label = QLabel(S.OVERALL_IDLE)
        layout.addWidget(self.overall_label)

        # No scroll area here. The whole page scrolls, so a second scrolling
        # region inside it would trap the wheel and confuse the keyboard.
        self.rows_host = QWidget()
        self.rows_grid = QGridLayout(self.rows_host)
        self.rows_grid.setContentsMargins(0, 0, 0, 0)
        self.rows_grid.setHorizontalSpacing(GRID)
        self.rows_grid.setVerticalSpacing(GRID)
        self.rows_grid.setColumnStretch(1, 1)
        layout.addWidget(self.rows_host, 1)
        layout.addStretch(0)
        return box

    def _build_finish(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(GRID)
        self.summary_label = QLabel("")
        self.summary_label.setVisible(False)
        self.open_output_button = QPushButton(S.OPEN_OUTPUT)
        self.open_output_button.setMinimumHeight(CONTROL_HEIGHT)
        self.open_output_button.setVisible(False)
        self.open_output_button.clicked.connect(self._open_output)
        row.addWidget(self.summary_label, 1)
        row.addWidget(self.open_output_button, 0, Qt.AlignRight)
        return row

    # ------------------------------------------------------------------- state

    def _restore(self) -> None:
        stored = self.settings_store.value("output_dir", "")
        if stored and Path(stored).is_dir():
            self.output_dir = Path(stored)
            self.output_value.setText(str(self.output_dir))
        self.advanced_toggle.setChecked(
            self.settings_store.value("advanced_open", False, type=bool))
        if self.settings_store.value("engine", "yunet") == "both":
            self.engine_max.setChecked(True)
        mode = self.settings_store.value("mode", "blur")
        {"blur": self.mode_blur, "pixelate": self.mode_pixelate,
         "solid": self.mode_solid}.get(mode, self.mode_blur).setChecked(True)
        self.stride_spin.setValue(self.settings_store.value("stride", 1, type=int))
        self.workers_spin.setValue(self.settings_store.value(
            "workers", max(1, multiprocessing.cpu_count() // 2), type=int))
        self.replace_check.setChecked(
            self.settings_store.value("replace", False, type=bool))

    def _remember(self) -> None:
        if self.output_dir:
            self.settings_store.setValue("output_dir", str(self.output_dir))
        self.settings_store.setValue("advanced_open", self.advanced_toggle.isChecked())
        self.settings_store.setValue("engine", self._engine())
        self.settings_store.setValue("mode", self._mode())
        self.settings_store.setValue("stride", self.stride_spin.value())
        self.settings_store.setValue("workers", self.workers_spin.value())
        self.settings_store.setValue("replace", self.replace_check.isChecked())

    def _engine(self) -> str:
        return "both" if self.engine_max.isChecked() else "yunet"

    def _mode(self) -> str:
        if self.mode_pixelate.isChecked():
            return "pixelate"
        if self.mode_solid.isChecked():
            return "solid"
        return "blur"

    def _running(self) -> bool:
        return self.runner is not None and self.runner.isRunning()

    # ----------------------------------------------------------------- choosing

    def _toggle_advanced(self, open_now: bool) -> None:
        self.advanced_body.setVisible(open_now)
        self.advanced_toggle.setArrowType(Qt.DownArrow if open_now else Qt.RightArrow)

    def _choose_files(self) -> None:
        if self._running():
            return
        names, _ = QFileDialog.getOpenFileNames(
            self, S.CHOOSE_FILES_TITLE, "", S.VIDEO_FILTER)
        if names:
            self._take_paths([Path(n) for n in names])

    def _choose_folder(self) -> None:
        if self._running():
            return
        name = QFileDialog.getExistingDirectory(self, S.CHOOSE_FOLDER_TITLE)
        if name:
            self._take_paths([Path(name)])

    def _choose_output(self) -> None:
        if self._running():
            return
        name = QFileDialog.getExistingDirectory(self, S.CHOOSE_OUTPUT_TITLE)
        if not name:
            return
        self.output_dir = Path(name)
        self.output_value.setText(str(self.output_dir))
        self._update_start_enabled()

    def _take_paths(self, paths: list[Path]) -> None:
        if self._running():
            return
        videos: list[Path] = []
        for path in paths:
            if path.is_dir():
                videos.extend(sorted(p for p in path.glob("*")
                                     if p.is_file() and p.suffix.lower() in VIDEO_EXT))
            elif path.is_file() and path.suffix.lower() in VIDEO_EXT:
                videos.append(path)
        if not videos:
            QMessageBox.warning(self, S.ERR_INPUT_TITLE, S.INPUT_NONE_FOUND)
            return

        self.inputs = list(dict.fromkeys(videos))
        total = sum(p.stat().st_size for p in self.inputs)
        self.drop_zone.show_summary(S.input_summary(len(self.inputs), total))

        if self.output_dir is None:
            first = paths[0]
            base = first if first.is_dir() else first.parent
            self.output_dir = base.parent / f"{base.name}_blurred"
            self.output_value.setText(str(self.output_dir))

        self._reset_rows()
        self._update_start_enabled()

    def _update_start_enabled(self) -> None:
        ready = bool(self.inputs) and self.output_dir is not None
        self.start_button.setEnabled(ready or self._running())
        self.start_help.setVisible(not ready and not self._running())

    def _reset_rows(self) -> None:
        while self.rows_grid.count():
            item = self.rows_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.rows = [FileRow(self.rows_grid, i, p.name)
                     for i, p in enumerate(self.inputs)]
        self.records.clear()
        self.overall_bar.setValue(0)
        self.overall_label.setText(
            S.OVERALL_RUNNING.format(done=0, total=len(self.inputs),
                                     estimate=S.ESTIMATING)
            if self.inputs else S.OVERALL_IDLE)
        self.summary_label.setVisible(False)
        self.open_output_button.setVisible(False)

    # ------------------------------------------------------------------ running

    def _start_or_stop(self) -> None:
        if self._running():
            self._ask_to_stop()
        elif self.start_button.isEnabled():
            self._start()

    def _start(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            probe = self.output_dir / ".faceblur_write_test"
            probe.write_bytes(b"")
            probe.unlink()
        except OSError:
            QMessageBox.warning(self, S.ERR_OUTPUT_TITLE, S.ERR_OUTPUT_UNWRITABLE)
            return

        settings = Settings(
            engine=self._engine(),
            stride=self.stride_spin.value(),
            mode=self._mode(),
            replace_existing=self.replace_check.isChecked(),
        )
        self.jobs = [(src, output_path(src, self.output_dir, settings.suffix))
                     for src in self.inputs]

        self._reset_rows()
        self._remember()
        self.started_at = time.monotonic()
        self.bytes_done = 0
        self.bytes_total = sum(p.stat().st_size for p in self.inputs)
        self._set_controls_enabled(False)
        self.start_button.setText(S.STOP)

        self.runner = BatchRunner(self.jobs, settings, self.workers_spin.value(), self)
        self.runner.progress.connect(self._on_progress)
        self.runner.one_finished.connect(self._on_one_finished)
        self.runner.all_finished.connect(self._on_all_finished)
        self.runner.start()

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (self.choose_files_button, self.choose_folder_button,
                       self.choose_output_button, self.engine_standard,
                       self.engine_max, self.mode_blur, self.mode_pixelate,
                       self.mode_solid, self.stride_spin, self.workers_spin,
                       self.replace_check):
            widget.setEnabled(enabled)
        self.drop_zone.setAcceptDrops(enabled)

    def _ask_to_stop(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle(S.STOP_CONFIRM_TITLE)
        box.setText(S.STOP_CONFIRM)
        stop = box.addButton(S.STOP_CONFIRM_YES, QMessageBox.DestructiveRole)
        keep = box.addButton(S.STOP_CONFIRM_NO, QMessageBox.RejectRole)
        box.setDefaultButton(keep)
        box.exec()
        if box.clickedButton() is stop and self.runner is not None:
            self.start_button.setEnabled(False)
            self.runner.stop()

    def _escape(self) -> None:
        if self._running():
            self._ask_to_stop()

    def _on_progress(self, index: int, stage: str, done: int, total: int) -> None:
        if index >= len(self.rows):
            return
        row = self.rows[index]
        words = {"detecting": S.STATUS_DETECTING, "writing": S.STATUS_WRITING,
                 "checking": S.STATUS_CHECKING}
        row.set_status(words.get(stage, S.STATUS_WRITING))
        # Each pass walks the whole video, so each is its own part of the row.
        # Checking the copy is a third pass, and only some runs make it.
        shares = {"detecting": (0.0, 0.5), "writing": (0.5, 0.5), "checking": (0.0, 1.0)}
        share, span = shares.get(stage, (0.5, 0.5))
        fraction = share + (span * done / total if total else 0.0)
        row.set_fraction(fraction)

    def _on_one_finished(self, index: int, record: dict) -> None:
        self.records[index] = record
        word = STATUS_FOR_RECORD.get(record.get("status"), S.STATUS_FAILED)
        tooltip = record.get("error", "")
        row = self.rows[index]
        row.set_status(word, tooltip)
        row.set_fraction(1.0 if word != S.STATUS_STOPPED else 0.0)

        self.bytes_done += self.inputs[index].stat().st_size
        self._update_overall()

    def _finished_count(self) -> int:
        """Rows that reached an output. A stopped row never counts as done."""
        return sum(1 for r in self.records.values()
                   if r.get("status") in (STATUS_DONE, STATUS_SKIPPED, STATUS_FAILED))

    def _update_overall(self) -> None:
        done = self._finished_count()
        total = len(self.rows)
        self.overall_bar.setValue(int(round(100 * done / total)) if total else 0)

        # The estimate uses measured throughput, never a guess before the first
        # file finishes.
        estimate = S.ESTIMATING
        elapsed = time.monotonic() - self.started_at
        if done and self.bytes_done and elapsed > 0:
            rate = self.bytes_done / elapsed
            remaining = max(0, self.bytes_total - self.bytes_done)
            if rate > 0:
                estimate = S.time_left(remaining / rate)
        self.overall_label.setText(
            S.OVERALL_RUNNING.format(done=done, total=total, estimate=estimate))

    def _on_all_finished(self, stopped: bool) -> None:
        for index, row in enumerate(self.rows):
            if index not in self.records:
                row.set_status(S.STATUS_STOPPED)
                row.set_fraction(0.0)

        done = sum(1 for r in self.records.values() if r.get("status") == STATUS_DONE)
        failed = sum(1 for r in self.records.values() if r.get("status") == STATUS_FAILED)
        skipped = sum(1 for r in self.records.values() if r.get("status") == STATUS_SKIPPED)
        template = S.SUMMARY_STOPPED if stopped else S.SUMMARY
        text = template.format(done=done, failed=failed)
        if skipped == 1:
            text += S.SUMMARY_SKIPPED_ONE
        elif skipped:
            text += S.SUMMARY_SKIPPED.format(skipped=skipped)
        self.summary_label.setText(text)
        self.summary_label.setVisible(True)

        # The run is over, so the estimate goes away rather than going stale.
        self.overall_label.setText(S.OVERALL_FINISHED.format(
            done=self._finished_count(), total=len(self.rows)))
        self.open_output_button.setVisible(True)

        self.start_button.setText(S.START)
        self.start_button.setEnabled(True)
        self._set_controls_enabled(True)
        self.runner = None
        self._update_start_enabled()

    def _open_output(self) -> None:
        if self.output_dir and self.output_dir.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_dir)))

    def _show_about(self) -> None:
        QMessageBox.information(self, S.ABOUT_TITLE, S.ABOUT_TEXT)

    def closeEvent(self, event) -> None:
        if self._running():
            box = QMessageBox(self)
            box.setWindowTitle(S.QUIT_CONFIRM_TITLE)
            box.setText(S.QUIT_CONFIRM)
            close = box.addButton(S.QUIT_CONFIRM_YES, QMessageBox.DestructiveRole)
            keep = box.addButton(S.QUIT_CONFIRM_NO, QMessageBox.RejectRole)
            box.setDefaultButton(keep)
            box.exec()
            if box.clickedButton() is not close:
                event.ignore()
                return
            self.runner.stop()
            self.runner.wait(15000)
        self._remember()
        event.accept()


def main() -> int:
    multiprocessing.freeze_support()
    QGuiApplication.setDesktopFileName(S.APP_NAME)
    app = QApplication(sys.argv)
    app.setApplicationName(S.APP_NAME)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
