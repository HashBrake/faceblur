"""Every string the user sees.

Written to the wording rules in Section 7 of the build plan:

- Active voice. The app does the thing.
- One instruction per sentence. At most 20 words per sentence.
- One word for one thing. The input is a video or a folder. The result is a
  blurred copy. The destination is the output folder.
- No phrasal verbs where one verb exists. Choose, not pick out. Start, not kick off.
- No marketing words.
- No em dashes and no en dashes.
- No emoji.
- Error messages say what happened, then what to do next.
- Never upgrade a hedge to a fact. Detection may miss a face, so the text says may.

The asd-ste100 skill is not installed in this environment, so these rules were
applied by hand.
"""

# Window
APP_NAME = "FaceBlur"
WINDOW_TITLE = "FaceBlur"
ABOUT_TITLE = "About FaceBlur"
ABOUT_TEXT = (
    "FaceBlur blurs faces in video. It runs on this PC. Your video never leaves "
    "the machine.\n\n"
    "FaceBlur does not change audio.\n\n"
    "FaceBlur may miss a face. Check the blurred copies before you share them."
)
MENU_ABOUT = "About FaceBlur"

# Step 1, choose the input
DROP_ZONE_EMPTY = "Drop a video or a folder here"
DROP_ZONE_HINT = "The app reads every video in a folder."
CHOOSE_FILES = "Choose files"
CHOOSE_FOLDER = "Choose a folder"
CHOOSE_FILES_TITLE = "Choose videos"
CHOOSE_FOLDER_TITLE = "Choose a folder of videos"
VIDEO_FILTER = "Videos (*.mp4 *.mov *.avi *.mkv *.m4v *.webm *.mpg *.mpeg *.wmv)"
INPUT_SUMMARY_ONE = "1 video, {size}"
INPUT_SUMMARY_MANY = "{count} videos, {size}"
INPUT_NONE_FOUND = "That folder holds no videos. Choose a folder that holds videos."

# Step 2, choose the output folder
OUTPUT_LABEL = "Output folder"
CHOOSE_OUTPUT = "Choose output folder"
CHOOSE_OUTPUT_TITLE = "Choose the output folder"
OUTPUT_EMPTY = "No folder chosen"
OUTPUT_HELP = "The app writes the blurred copies to this folder."

# Step 3, advanced settings
ADVANCED = "Advanced settings"
ENGINE_LABEL = "Detector"
ENGINE_STANDARD = "Standard"
ENGINE_STANDARD_HELP = "One detector finds faces, a second confirms them. May miss small or turned faces."
ENGINE_MAX = "Maximum coverage"
ENGINE_MAX_HELP = "Two detectors, no confirmation. Finds more faces. May also blur hands and objects."
STRIDE_LABEL = "Detect on every Nth frame"
STRIDE_HELP = "Detect faces on every Nth frame. 1 is safest. Higher is faster."
MODE_LABEL = "Redaction style"
MODE_BLUR = "Blur"
MODE_BLUR_HELP = "The app shrinks the face area, then blurs it. The face cannot come back."
MODE_PIXELATE = "Pixelate"
MODE_PIXELATE_HELP = "The app shrinks the face area, then shows it as large blocks."
MODE_SOLID = "Black box"
MODE_SOLID_HELP = "The app paints the face area black."
WORKERS_LABEL = "Videos at the same time"
WORKERS_HELP = "Process this many videos at once. Higher uses more of the CPU."
REPLACE_EXISTING = "Replace existing files"
REPLACE_EXISTING_HELP = "Blur a video again even when its blurred copy exists."

# Step 4, run
START = "Blur faces"
STOP = "Stop"
START_HELP = "Choose videos and an output folder to start."

# Progress
STATUS_WAITING = "Waiting"
STATUS_DETECTING = "Detecting"
STATUS_WRITING = "Writing"
STATUS_CHECKING = "Checking"
STATUS_DONE = "Done"
STATUS_FAILED = "Failed"
STATUS_STOPPED = "Stopped"
STATUS_SKIPPED = "Already done"
PROGRESS_HEADING = "Progress"
OVERALL_IDLE = "No videos yet."
OVERALL_RUNNING = "{done} of {total} done. {estimate}"
OVERALL_FINISHED = "{done} of {total} done."
ESTIMATING = "Estimating time left."
TIME_LEFT_MINUTES = "About {minutes} minutes left."
TIME_LEFT_ONE_MINUTE = "About 1 minute left."
TIME_LEFT_ONE_HOUR = "About 1 hour left."
TIME_LEFT_ONE_HOUR_MINUTES = "About 1 hour {minutes} minutes left."
TIME_LEFT_HOURS_ONLY = "About {hours} hours left."
TIME_LEFT_HOURS = "About {hours} hours {minutes} minutes left."
TIME_LEFT_SECONDS = "Less than a minute left."

# Finish
SUMMARY = "{done} videos done. {failed} failed."
SUMMARY_STOPPED = "You stopped the run. {done} videos done. {failed} failed."
# Added when a run skipped copies that already existed. Without it a second run
# over the same output folder reports zero videos done, which reads as a failure.
SUMMARY_SKIPPED = " {skipped} were already done."
SUMMARY_SKIPPED_ONE = " 1 was already done."
OPEN_OUTPUT = "Open output folder"

# Confirmations
STOP_CONFIRM_TITLE = "Stop"
STOP_CONFIRM = "Stop now? The app deletes unfinished files."
STOP_CONFIRM_YES = "Stop"
STOP_CONFIRM_NO = "Keep going"
QUIT_CONFIRM_TITLE = "Close FaceBlur"
QUIT_CONFIRM = "The app is still working. Close now? The app deletes unfinished files."
QUIT_CONFIRM_YES = "Close"
QUIT_CONFIRM_NO = "Keep going"

# Errors
ERR_UNREADABLE = "Could not read this video. Check that the file is not open in another program."
ERR_NO_SPACE = "Not enough disk space in the output folder."
ERR_FFMPEG = "Could not write the blurred copy. See the log file for details."
ERR_OUTPUT_TITLE = "Output folder"
ERR_OUTPUT_UNWRITABLE = "Could not write to this folder. Choose a folder you can write to."
ERR_INPUT_TITLE = "Videos"
ERR_MODELS_TITLE = "Missing model file"
ERR_MODELS = "Could not find a model file. Install FaceBlur again."

# Accessible names for controls that show an icon or a bare value
ACC_OVERALL_BAR = "Overall progress"
ACC_FILE_BAR = "Progress for {name}"
ACC_ADVANCED_TOGGLE = "Show or hide advanced settings"
ACC_DROP_ZONE = "Drop zone for videos and folders"

# Icons drawn as text next to a status word, so colour never carries the meaning
# on its own.
STATUS_MARK = {
    STATUS_WAITING: "·",    # middle dot
    STATUS_DETECTING: "▶",  # right pointing triangle
    STATUS_WRITING: "▶",
    STATUS_CHECKING: "▶",
    STATUS_DONE: "✓",       # check mark
    STATUS_FAILED: "✕",     # multiplication x
    STATUS_STOPPED: "■",    # black square
    STATUS_SKIPPED: "✓",
}


def human_size(num_bytes: int) -> str:
    """Bytes as a short string. Used in the input summary."""
    step = 1024.0
    value = float(num_bytes)
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if value < step or unit == "TB":
            if unit == "bytes":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} TB"


def time_left(seconds: float) -> str:
    """Seconds as a sentence. Only called once a file has finished."""
    if seconds < 60:
        return TIME_LEFT_SECONDS
    minutes = int(round(seconds / 60))
    if minutes < 60:
        if minutes == 1:
            return TIME_LEFT_ONE_MINUTE
        return TIME_LEFT_MINUTES.format(minutes=minutes)
    hours, minutes = divmod(minutes, 60)
    if hours == 1:
        if minutes == 0:
            return TIME_LEFT_ONE_HOUR
        return TIME_LEFT_ONE_HOUR_MINUTES.format(minutes=minutes)
    if minutes == 0:
        return TIME_LEFT_HOURS_ONLY.format(hours=hours)
    return TIME_LEFT_HOURS.format(hours=hours, minutes=minutes)


def input_summary(count: int, total_bytes: int) -> str:
    size = human_size(total_bytes)
    if count == 1:
        return INPUT_SUMMARY_ONE.format(size=size)
    return INPUT_SUMMARY_MANY.format(count=count, size=size)
