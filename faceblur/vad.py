"""Find the speech in a recording's audio, for the audit record.

The release carries no audio at all (decision D-B), so nothing here changes a
sound. What the language team needs instead is to know what was taken away:
every stretch of the input audio in which somebody spoke, as start and end
seconds. Silero VAD finds them. It is a 2 MB model that reads 32 ms of 16 kHz
mono at a time, carries a recurrent state from one window to the next and says
how likely that window is to be speech.

It runs on the CPU. The model is tiny, an hour of audio is about a hundred
thousand windows of microseconds each, and DirectML would only add a copy to
the card and back for every one of them.
"""
from __future__ import annotations

from pathlib import Path

import av
import h5py
import numpy as np
import onnxruntime as ort

from . import df2
from .detect import MODEL_DIR, model_sha256 as _sha256
from .settings import Settings

VAD_MODEL = MODEL_DIR / "silero_vad.onnx"

RATE = 16000
# The window the model is trained on at 16 kHz, and the tail of the previous
# window the reference code puts in front of each one. Without that context
# the model sees every window start cold and its probabilities fall apart.
WINDOW = 512
CONTEXT = 64


def model_sha256() -> str:
    """The hash of the VAD model, for the audit record."""
    return _sha256(Path(VAD_MODEL))


def read_pcm(f: h5py.File) -> tuple[np.ndarray, int] | None:
    """The audio of a DF2 file as int16 [samples, channels] and its rate, or None."""
    path = f"{df2.AUDIO_GROUP}/{df2.AUDIO_PCM}"
    if path not in f:
        return None
    dataset = f[path]
    return dataset[()], int(dataset.attrs[df2.ATTR_SAMPLE_RATE])


def _mono_16k(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    """int16 [samples, channels] or [samples] to float32 mono at 16 kHz in [-1, 1]."""
    pcm = np.asarray(pcm)
    if pcm.ndim == 2:
        mono = pcm.astype(np.float32).mean(axis=1)
    else:
        mono = pcm.astype(np.float32)
    mono /= 32768.0
    if sample_rate == RATE or mono.size == 0:
        return mono
    frame = av.AudioFrame.from_ndarray(mono[np.newaxis, :], format="fltp", layout="mono")
    frame.sample_rate = sample_rate
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=RATE)
    # The resampler holds some samples back until it is flushed with None.
    out = [o.to_ndarray()[0] for o in resampler.resample(frame)]
    out += [o.to_ndarray()[0] for o in resampler.resample(None)]
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def _probabilities(audio: np.ndarray) -> np.ndarray:
    """Speech probability of each 512-sample window, the state carried across windows."""
    session = ort.InferenceSession(str(VAD_MODEL), providers=["CPUExecutionProvider"])
    count = -(-audio.size // WINDOW)
    # The last window is padded with silence rather than dropped, so speech
    # that runs to the very end of a recording is still found.
    padded = np.zeros(count * WINDOW, np.float32)
    padded[:audio.size] = audio
    state = np.zeros((2, 1, 128), np.float32)
    context = np.zeros(CONTEXT, np.float32)
    sr = np.array(RATE, dtype=np.int64)
    probs = np.empty(count, np.float32)
    for i in range(count):
        chunk = padded[i * WINDOW:(i + 1) * WINDOW]
        x = np.concatenate([context, chunk])[np.newaxis, :]
        out, state = session.run(None, {"input": x, "state": state, "sr": sr})
        probs[i] = out[0, 0]
        context = chunk[-CONTEXT:]
    return probs


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) window indices of each run of True."""
    edges = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def speech_segments(pcm: np.ndarray, sample_rate: int,
                    settings: Settings) -> list[tuple[float, float]]:
    """Every stretch of speech in `pcm` as (start_s, end_s), padded and merged."""
    audio = _mono_16k(pcm, sample_rate)
    if audio.size == 0:
        return []
    duration = audio.size / RATE
    window_s = WINDOW / RATE
    runs = _runs(_probabilities(audio) >= settings.vad_threshold)
    # A breath between two words is not the end of the speech: join runs
    # whose gap is shorter than the shortest silence that counts.
    joined: list[list[int]] = []
    for start, end in runs:
        if joined and (start - joined[-1][1]) * window_s * 1000 < settings.vad_min_silence_ms:
            joined[-1][1] = end
        else:
            joined.append([start, end])
    pad = settings.vad_pad_ms / 1000
    segments: list[list[float]] = []
    for start, end in joined:
        if (end - start) * window_s * 1000 < settings.vad_min_speech_ms:
            continue  # a click or a cough, not speech
        lo = max(0.0, start * window_s - pad)
        hi = min(duration, end * window_s + pad)
        # Padding can make neighbours touch; the record lists each stretch once.
        if segments and lo <= segments[-1][1]:
            segments[-1][1] = max(segments[-1][1], hi)
        else:
            segments.append([lo, hi])
    return [(round(float(lo), 3), round(float(hi), 3)) for lo, hi in segments]
