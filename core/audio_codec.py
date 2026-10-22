"""Opus for audio frames (optional).

Opus is an optional dependency: the `opuslib` package plus the libopus library. Nothing here imports
it until a codec is asked for, and a missing package or library raises `AudioUnavailable`, which the
caller answers by staying on raw s16le. `loader` lets a test hand in a fake module with the opuslib
interface (`Encoder(rate, channels, application)`, `Decoder(rate, channels)`)."""

import importlib.util
import logging

from . import audio_io, protocol

LOGGER = logging.getLogger(__name__)

BITRATE = {1: 64000, 2: 96000}   # bits per second: speech-grade microphone, music-grade stereo sound


def _opuslib():
    try:
        import opuslib
    except Exception as error:  # ImportError, or OSError when libopus is missing
        raise audio_io.AudioUnavailable("Opus needs the opuslib package and libopus (pip install opuslib): %s" % error) from error
    return opuslib


def available() -> bool:
    """True when the package can be found. The library itself is only loaded by `Encoder`/`Decoder`,
    so a missing libopus still shows up there, as `AudioUnavailable`, and the stream stays s16le."""
    try:
        return importlib.util.find_spec("opuslib") is not None
    except (ImportError, ValueError):
        return False


class Encoder:
    """One s16le frame (20 ms) in, one Opus packet out."""

    def __init__(self, channels: int, loader=_opuslib) -> None:
        lib = loader()
        self.channels = channels
        self.samples = audio_io.FRAME_SAMPLES
        try:
            self._encoder = lib.Encoder(audio_io.RATE, channels, "audio")
        except Exception as error:
            raise audio_io.AudioUnavailable("Opus encoder: %s" % error) from error
        try:
            self._encoder.bitrate = BITRATE.get(channels, 96000)
        except Exception:
            LOGGER.debug("this opuslib cannot set the bit rate; using its default")

    def encode(self, pcm: bytes) -> bytes:
        packet = self._encoder.encode(pcm, self.samples)
        if not packet or len(packet) > protocol.AUDIO_MAX_OPUS_BYTES:
            raise ValueError("Opus packet of %d bytes" % len(packet or b""))
        return bytes(packet)


class Decoder:
    """One Opus packet in, one s16le frame out, or ValueError when the packet is bad."""

    def __init__(self, channels: int, loader=_opuslib) -> None:
        lib = loader()
        self.channels = channels
        self.samples = audio_io.FRAME_SAMPLES
        try:
            self._decoder = lib.Decoder(audio_io.RATE, channels)
        except Exception as error:
            raise audio_io.AudioUnavailable("Opus decoder: %s" % error) from error

    def decode(self, packet: bytes) -> bytes:
        try:
            pcm = bytes(self._decoder.decode(packet, self.samples))
        except Exception as error:
            raise ValueError("bad Opus packet: %s" % error) from error
        if len(pcm) != self.samples * self.channels * 2:
            raise ValueError("Opus packet decoded to %d bytes" % len(pcm))
        return pcm
