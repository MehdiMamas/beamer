"""The receiving end of an audio stream: a jitter buffer and the pump that feeds a device from it.

Pure Python, no audio library: frames come from the link (so they arrive late, out of order or
not at all) and leave one per `frame_ms`, in order, with silence where a frame is missing."""

import logging
import threading
from typing import Optional

LOGGER = logging.getLogger(__name__)

TARGET_MS = 60
MAX_MS = 400


class JitterBuffer:
    """Reorders frames by `seq`, holds about `TARGET_MS` before it starts, drops late frames, hands
    out silence on an underrun and, past `max_ms`, drops the oldest.

    `push` and `pull` may be called from different threads."""

    def __init__(self, frame_bytes: int, frame_ms: int = 20, target_ms: int = TARGET_MS, max_ms: int = MAX_MS) -> None:
        if frame_bytes <= 0 or frame_ms <= 0:
            raise ValueError("a frame has a size")
        self.frame_bytes = frame_bytes
        self.target = max(1, target_ms // frame_ms)
        self.limit = max(self.target, max_ms // frame_ms)
        self._silence = bytes(frame_bytes)
        self._frames = {}
        self._next: Optional[int] = None
        self._playing = False
        self._lock = threading.Lock()
        self.late = 0
        self.dropped = 0
        self.underruns = 0
        self.lost = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def push(self, seq: int, pcm: bytes) -> bool:
        """True when the frame was kept. Frames of the wrong size, repeats and frames the player
        has already passed are refused."""
        if len(pcm) != self.frame_bytes:
            return False
        with self._lock:
            if self._next is not None and self._playing and seq < self._next:
                self.late += 1
                return False
            if seq in self._frames:
                return False
            if self._frames and seq > max(self._frames) + self.limit:
                # The sender jumped ahead (or restarted): what is held can never be played in order.
                self.dropped += len(self._frames)
                self._frames.clear()
                self._playing = False
            self._frames[seq] = pcm
            while len(self._frames) > self.limit:
                oldest = min(self._frames)
                del self._frames[oldest]
                self.dropped += 1
                if self._next is not None and self._next <= oldest:
                    self._next = oldest + 1
            return True

    def pull(self) -> bytes:
        """The next frame to play: always `frame_bytes` long."""
        with self._lock:
            if not self._playing:
                if len(self._frames) < self.target:
                    return self._silence
                self._playing = True
                self._next = min(self._frames)
            frame = self._frames.pop(self._next, None)
            if frame is not None:
                self._next += 1
                return frame
            if not self._frames:
                # Nothing left: wait for the buffer to fill again before playing on.
                self.underruns += 1
                self._playing = False
                return self._silence
            # A frame went missing while later ones wait: skip it rather than stall behind it.
            self.lost += 1
            self._next += 1
            return self._silence


class Playback:
    """Feeds `sink` (anything with `write(pcm)` and `close()`) from a `JitterBuffer`. The device's
    own pacing sets the speed, so the loop is `write(pull())`; `step()` is one turn of it."""

    def __init__(self, sink, buffer: JitterBuffer, name: str = "audio", decoder=None) -> None:
        self.sink = sink
        self.buffer = buffer
        self.name = name
        self.decoder = decoder      # turns an Opus packet into a frame; None when the stream is s16le
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def push(self, seq: int, pcm: bytes, format: str = "s16le") -> bool:
        """Queue one frame. An Opus packet is decoded first; one that does not decode is dropped."""
        if format == "opus":
            if self.decoder is None:
                return False
            try:
                pcm = self.decoder.decode(pcm)
            except ValueError:
                return False
        elif self.decoder is not None:
            return False
        return self.buffer.push(seq, pcm)

    def step(self) -> None:
        self.sink.write(self.buffer.pull())

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self.step()
        except Exception:
            LOGGER.exception("%s stopped: the output failed", self.name)

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        try:
            self.sink.close()
        except Exception:
            LOGGER.exception("%s could not close its output", self.name)
