import numpy as np

from audio_monitor.classifier import BackgroundClassifier


def test_silence_classification():
    classifier = BackgroundClassifier()
    samples = np.zeros(16000)
    result = classifier.classify(samples, sample_rate=16000)
    assert result.label == "ambient/silence"


def test_voice_like_signal():
    classifier = BackgroundClassifier()
    t = np.linspace(0, 1, 16000)
    samples = 0.1 * np.sin(2 * np.pi * 300 * t)
    result = classifier.classify(samples, sample_rate=16000)
    assert result.label == "voice/call"
    assert result.stress_level is not None
    assert 0.0 <= result.stress_score <= 1.0


def test_music_like_signal():
    classifier = BackgroundClassifier()
    t = np.linspace(0, 1, 16000)
    samples = 0.4 * np.sin(2 * np.pi * 4000 * t)
    result = classifier.classify(samples, sample_rate=16000)
    assert result.label == "music/high-activity"
