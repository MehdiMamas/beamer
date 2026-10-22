"""Echo cancellation for a machine that plays the other computer's sound and sends its own microphone.

The microphone hears what the speakers play. `EchoCanceller` learns that path (delay, room, gain) with
a partitioned frequency-domain block NLMS filter and subtracts the estimate. The far-end reference is
what this machine actually plays: `far()` takes each played frame, `process()` takes each microphone
frame and returns it with the echo taken out. Both run on one 20 ms frame (960 samples at 48 kHz).

It covers up to `partitions` frames (120 ms by default) of delay between writing a frame to the output
and hearing it in the microphone. It does not detect double talk, so it adapts with a moderate step
and stops adapting while nothing is played, so that silence cannot make it diverge."""

import threading

from . import audio_io

BLOCK = audio_io.FRAME_SAMPLES


class EchoCanceller:
    def __init__(self, partitions: int = 6, step: float = 0.5, block: int = BLOCK) -> None:
        import numpy
        self._np = numpy
        self.block = block
        self.partitions = partitions
        self.step = step
        bins = block + 1
        self._weights = numpy.zeros((partitions, bins), dtype=numpy.complex128)
        self._history = numpy.zeros((partitions, bins), dtype=numpy.complex128)
        self._far_tail = numpy.zeros(block, dtype=numpy.float64)       # the previous far block
        self._far_now = numpy.zeros(block, dtype=numpy.float64)        # the newest far block
        self._power = numpy.full(bins, 1e-3)
        self._lock = threading.Lock()

    def far(self, pcm: bytes, channels: int = 2) -> None:
        """A frame of s16le that is being played (stereo is mixed down to one channel)."""
        np = self._np
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if len(samples) != self.block:
            return
        with self._lock:
            self._far_tail, self._far_now = self._far_now, samples
            spectrum = np.fft.rfft(np.concatenate((self._far_tail, self._far_now)))
            self._history = np.roll(self._history, 1, axis=0)
            self._history[0] = spectrum

    def process(self, pcm: bytes) -> bytes:
        """A frame of microphone s16le (mono) in, the same frame without the echo out."""
        np = self._np
        near = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
        if len(near) != self.block:
            return pcm
        with self._lock:
            estimate = np.fft.irfft((self._weights * self._history).sum(axis=0), n=2 * self.block)[self.block:]
            error = near - estimate
            self._power = 0.9 * self._power + 0.1 * (np.abs(self._history) ** 2).sum(axis=0)
            if float((self._far_now ** 2).mean()) > 1e-8:
                spectrum = np.fft.rfft(np.concatenate((np.zeros(self.block), error)))
                scaled = spectrum / (self._power + 1e-6)
                for index in range(self.partitions):
                    gradient = np.fft.irfft(np.conj(self._history[index]) * scaled, n=2 * self.block)
                    gradient[self.block:] = 0.0
                    self._weights[index] += self.step * np.fft.rfft(gradient)
        return (np.clip(error, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class EchoSource:
    """A microphone source whose frames come out echo-cancelled."""

    def __init__(self, source, canceller: EchoCanceller) -> None:
        self._source, self._canceller = source, canceller
        self.channels = getattr(source, "channels", 1)

    def read(self) -> bytes:
        return self._canceller.process(self._source.read())

    def close(self) -> None:
        self._source.close()


class EchoSink:
    """A sound sink that shows the canceller every frame it plays."""

    def __init__(self, sink, canceller: EchoCanceller, channels: int = 2) -> None:
        self._sink, self._canceller, self._channels = sink, canceller, channels

    def write(self, pcm: bytes) -> None:
        self._canceller.far(pcm, self._channels)
        self._sink.write(pcm)

    def close(self) -> None:
        self._sink.close()
