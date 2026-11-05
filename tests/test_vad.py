"""Silero VAD: the speech stretches the audit lists as removed audio.

tests/data/speech_16k.wav is synthesized speech followed by digital silence,
16 kHz mono pcm_s16le, 4.7 s, 75200 samples. Windows SAPI (Microsoft David
Desktop) said "The quick brown fox jumps over the lazy dog." into a 16 kHz
mono wav; SAPI's own trailing quiet was cut at 2.7 s and 2.0 s of zeros
appended. The voice runs from about 0.11 s to 2.59 s. Made with:

    Add-Type -AssemblyName System.Speech
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $s.SetOutputToWaveFile("speech_raw.wav", (New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,
        [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)))
    $s.Speak("The quick brown fox jumps over the lazy dog.")
    $FF -y -i speech_raw.wav -af "atrim=end=2.7,apad=pad_dur=2" -ar 16000 -ac 1 \
        -c:a pcm_s16le -map_metadata -1 -fflags +bitexact tests/data/speech_16k.wav
"""
from __future__ import annotations

import wave

import h5py
import numpy as np
import onnxruntime as ort
import pytest

from faceblur import vad
from faceblur.settings import Settings, SettingsError

from conftest import DATA

SPEECH = DATA / "speech_16k.wav"
# Where the spoken part ends and the appended silence begins.
SPOKEN_END = 2.7


def load_speech() -> np.ndarray:
    with wave.open(str(SPEECH)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def first_speech_s(pcm: np.ndarray) -> float:
    """The start of the first 10 ms block that is clearly not quiet."""
    blocks = np.abs(pcm[:pcm.size // 160 * 160].reshape(-1, 160)).max(axis=1)
    return float(np.flatnonzero(blocks > 300)[0]) / 100


def total(segments) -> float:
    return sum(hi - lo for lo, hi in segments)


@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    return load_speech()


@pytest.fixture(scope="module")
def segments(speech):
    return vad.speech_segments(speech, 16000, Settings())


def test_the_sample_is_speech_then_silence(speech):
    assert speech.size == 75200
    assert SPEECH.stat().st_size < 200_000
    assert np.abs(speech[int(SPOKEN_END * 16000):]).max() == 0
    assert np.abs(speech[:int(SPOKEN_END * 16000)]).max() > 1000


def test_speech_is_found_and_covered(speech, segments):
    assert segments
    assert all(isinstance(t, float) for seg in segments for t in seg)
    covered = sum(max(0.0, min(hi, SPOKEN_END) - lo) for lo, hi in segments)
    assert covered >= 0.6 * SPOKEN_END


def test_no_segment_runs_far_into_the_silence(segments):
    limit = SPOKEN_END + (Settings().vad_pad_ms + 100) / 1000
    assert all(hi <= limit for _, hi in segments)


def test_the_first_segment_starts_near_the_first_speech(speech, segments):
    earliest = max(0.0, first_speech_s(speech) - Settings().vad_pad_ms / 1000)
    assert segments[0][0] >= earliest - 1e-3
    assert segments[0][0] <= first_speech_s(speech) + 0.1


def test_padding_adds_the_pad_on_both_sides(speech):
    # A second of silence in front, so the padding is not clipped at 0.
    shifted = np.concatenate([np.zeros(16000, np.int16), speech])
    padded = vad.speech_segments(shifted, 16000, Settings())
    bare = vad.speech_segments(shifted, 16000, Settings().with_changes(vad_pad_ms=0))
    assert len(padded) == len(bare) == 1
    assert abs(total(padded) - total(bare) - 0.6) <= 0.05


def test_padding_is_clipped_at_zero(speech, segments):
    bare = vad.speech_segments(speech, 16000, Settings().with_changes(vad_pad_ms=0))
    # The speech starts within 300 ms of the start, so the pad would reach before 0.
    assert 0 < bare[0][0] < 0.3
    assert segments[0][0] == 0.0


def test_a_tone_is_not_speech():
    t = np.arange(3 * 16000) / 16000
    tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    assert vad.speech_segments(tone, 16000, Settings()) == []


def test_silence_is_not_speech():
    assert vad.speech_segments(np.zeros((3 * 16000, 1), np.int16), 16000, Settings()) == []


def test_48k_stereo_gives_the_same_segments(speech, segments):
    stereo = np.repeat(speech, 3)[:, np.newaxis].repeat(2, axis=1)
    other = vad.speech_segments(stereo, 48000, Settings())
    assert len(other) == len(segments)
    for (a0, a1), (b0, b1) in zip(segments, other):
        assert abs(a0 - b0) <= 0.1 and abs(a1 - b1) <= 0.1


def test_read_pcm_reads_the_fixture_audio(speech, tmp_path):
    with h5py.File(DATA / "df2_small.h5", "r") as f:
        pcm, rate = vad.read_pcm(f)
    assert rate == 16000
    assert pcm.dtype == np.int16 and pcm.shape == (speech.size, 1)
    assert np.array_equal(pcm[:, 0], speech)
    empty = tmp_path / "no_audio.h5"
    with h5py.File(empty, "w") as f:
        f.create_group("dataset/meta")
    with h5py.File(empty, "r") as f:
        assert vad.read_pcm(f) is None


def test_vad_settings_are_recorded_and_checked():
    d = Settings().to_dict()
    assert d["vad_threshold"] == 0.5
    assert d["vad_pad_ms"] == 300
    assert d["vad_min_speech_ms"] == 250
    assert d["vad_min_silence_ms"] == 100
    with pytest.raises(SettingsError):
        Settings(vad_threshold=1.5)
    with pytest.raises(SettingsError):
        Settings(vad_pad_ms=-1)


def test_the_model_is_pinned_and_loads_on_the_cpu():
    from test_models import EXPECTED

    assert vad.model_sha256() == EXPECTED["silero_vad.onnx"][0]
    session = ort.InferenceSession(str(vad.VAD_MODEL), providers=["CPUExecutionProvider"])
    assert {i.name for i in session.get_inputs()} == {"input", "state", "sr"}
