import numpy as np

from tts.base import AudioData
from tts import playback


class FakeSD:
    class Default:
        device = [0, 0]

    default = Default()

    def __init__(self):
        self.calls = []

    def query_devices(self, device=None, kind=None):
        return {"default_samplerate": 44100.0}

    def play(self, samples, samplerate):
        self.calls.append((np.asarray(samples, dtype=np.float32), int(samplerate)))

    def wait(self):
        pass


def test_play_resamples_to_detected_output_device_rate(monkeypatch):
    fake_sd = FakeSD()
    monkeypatch.setattr(playback, "sd", fake_sd)

    audio = AudioData(samples=np.linspace(-1.0, 1.0, 24000, dtype=np.float32), sample_rate=24000)

    playback.play(audio)

    assert len(fake_sd.calls) == 1
    samples_out, samplerate_out = fake_sd.calls[0]
    assert samplerate_out == 44100
    assert samples_out.shape[0] > 24000
