"""Sound devices behind a small interface, so the rest of audio needs no sound hardware.

A backend offers `open_source(kind)` (kind "loopback": what this machine plays; "mic": the default
microphone) and `open_sink(kind)` (kind "speaker": the default output; "virtual_mic": a virtual
cable the other computer's microphone is played into). A source has `read()` returning one frame of
s16le, and `close()`; a sink has `write(pcm)` and `close()`. Anything that cannot be opened raises
`AudioUnavailable` with a sentence for the log.

The real backend uses the optional `soundcard` package, imported only when a device is opened."""

import importlib.util
import logging
import re
import subprocess
import sys

LOGGER = logging.getLogger(__name__)

RATE = 48000
FRAME_MS = 20
FRAME_SAMPLES = RATE * FRAME_MS // 1000
# Names of virtual cables that an input can be played into, by platform. The device is the cable's
# playback side; other programs see its recording side as a microphone.
VIRTUAL_CABLES = {"win32": ("CABLE Input",), "darwin": ("BlackHole",)}
LOOPBACK_DEVICES = {"darwin": ("BlackHole",)}
LINUX_SINK = "beamer_mic"
APPS_SINK = "beamer_apps"       # Linux: the null sink the chosen programs are moved to
MAX_APPS = 32
ROUTE_EVERY = 100               # frames between looks for new streams of the chosen programs (2 s)


class AudioUnavailable(Exception):
    """A device or the audio library is missing; the text says what and why."""


def available() -> bool:
    """True when the optional audio library can be imported (nothing is imported to find out)."""
    try:
        return all(importlib.util.find_spec(name) is not None for name in ("soundcard", "numpy"))
    except (ImportError, ValueError):
        return False


def _load():
    try:
        import numpy
        import soundcard
    except Exception as error:
        raise AudioUnavailable("audio needs the soundcard package (pip install soundcard): %s" % error) from error
    return soundcard, numpy


def _named(devices, names):
    for device in devices:
        label = getattr(device, "name", "") or ""
        if any(name.lower() in label.lower() for name in names):
            return device
    return None


class _Source:
    def __init__(self, recorder, numpy, channels: int) -> None:
        self._recorder, self._np = recorder, numpy
        self.channels = channels

    def read(self) -> bytes:
        data = self._recorder.record(numframes=FRAME_SAMPLES)
        return (self._np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    def close(self) -> None:
        self._recorder.__exit__(None, None, None)


class _Sink:
    def __init__(self, player, numpy, channels: int, on_close=None) -> None:
        self._player, self._np, self.channels, self._on_close = player, numpy, channels, on_close

    def write(self, pcm: bytes) -> None:
        samples = self._np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        self._player.play(samples.reshape(-1, self.channels))

    def close(self) -> None:
        try:
            self._player.__exit__(None, None, None)
        finally:
            if self._on_close is not None:
                self._on_close()


def _pactl(args, run=subprocess.run) -> str:
    try:
        done = run(["pactl", *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise AudioUnavailable("pactl is not available: %s" % error) from error
    if done.returncode != 0:
        raise AudioUnavailable("pactl %s failed: %s" % (args[0], (done.stderr or "").strip()))
    return (done.stdout or "").strip()


class LinuxVirtualMic:
    """A null sink with a source that reads from it, which other programs list as a microphone.
    `unload()` removes both."""

    def __init__(self, run=subprocess.run) -> None:
        self._run = run
        self._modules = []

    def load(self) -> str:
        try:
            self._modules.append(_pactl(["load-module", "module-null-sink", "sink_name=" + LINUX_SINK,
                                         "sink_properties=device.description=Beamer_Microphone"], self._run))
            self._modules.append(_pactl(["load-module", "module-remap-source", "master=%s.monitor" % LINUX_SINK,
                                         "source_name=%s_source" % LINUX_SINK,
                                         "source_properties=device.description=Beamer_Microphone"], self._run))
        except AudioUnavailable:
            self.unload()
            raise
        return LINUX_SINK

    def unload(self) -> None:
        while self._modules:
            module = self._modules.pop()
            if not re.fullmatch(r"\d+", module):
                continue
            try:
                _pactl(["unload-module", module], self._run)
            except AudioUnavailable:
                LOGGER.warning("could not unload audio module %s", module)


def parse_sink_inputs(text: str) -> list:
    """The output of `pactl list sink-inputs` as [(index, {property: value})]."""
    streams, current = [], None
    for line in (text or "").splitlines():
        match = re.match(r"Sink Input #(\d+)", line)
        if match:
            current = (int(match.group(1)), {})
            streams.append(current)
        elif current is not None:
            pair = re.match(r'\s+([\w.\-]+) = "(.*)"\s*$', line)
            if pair:
                current[1][pair.group(1)] = pair.group(2)
    return streams


def matches_app(properties: dict, names) -> bool:
    """True when a stream's application name, program binary or media name holds one of `names`."""
    wanted = [name.strip().lower() for name in names if name and name.strip()]
    if not wanted:
        return False
    fields = [properties.get(key, "").lower() for key in
              ("application.name", "application.process.binary", "application.id")]
    return any(name in field for name in wanted for field in fields if field)


class LinuxAppRouter:
    """Sends the sound of chosen programs to a null sink of its own, so capturing that sink's monitor
    gives only those programs. `route()` moves the matching streams (and is called again from time
    to time, because a program opens a new stream for each sound); `unload()` puts them back."""

    def __init__(self, names, run=subprocess.run) -> None:
        self._names = [name for name in list(names)[:MAX_APPS] if isinstance(name, str)]
        self._run = run
        self._module = None
        self._moved = set()

    def load(self) -> str:
        """Create the sink and return the name of its monitor."""
        if not self._names:
            raise AudioUnavailable("no programs are chosen for sound sharing")
        module = _pactl(["load-module", "module-null-sink", "sink_name=" + APPS_SINK,
                         "sink_properties=device.description=Beamer_Programs"], self._run)
        if not re.fullmatch(r"\d+", module):
            raise AudioUnavailable("pactl gave an unexpected module id")
        self._module = module
        self.route()
        return APPS_SINK + ".monitor"

    def route(self) -> int:
        """Move every stream of a chosen program that is not yet on the sink. Returns how many moved."""
        moved = 0
        try:
            streams = parse_sink_inputs(_pactl(["list", "sink-inputs"], self._run))
        except AudioUnavailable:
            return 0
        for index, properties in streams:
            if index in self._moved or not matches_app(properties, self._names):
                continue
            try:
                _pactl(["move-sink-input", str(index), APPS_SINK], self._run)
            except AudioUnavailable:
                continue
            self._moved.add(index)
            moved += 1
        return moved

    def unload(self) -> None:
        for index in sorted(self._moved):
            try:
                _pactl(["move-sink-input", str(index), "@DEFAULT_SINK@"], self._run)
            except AudioUnavailable:
                pass     # the stream is gone already
        self._moved.clear()
        module, self._module = self._module, None
        if module:
            try:
                _pactl(["unload-module", module], self._run)
            except AudioUnavailable:
                LOGGER.warning("could not unload audio module %s", module)


class _RoutedSource(_Source):
    """A loopback source of the router's sink that keeps looking for new streams of the chosen programs."""

    def __init__(self, recorder, numpy, channels: int, router: LinuxAppRouter) -> None:
        super().__init__(recorder, numpy, channels)
        self._router, self._reads = router, 0

    def read(self) -> bytes:
        self._reads += 1
        if self._reads % ROUTE_EVERY == 0:
            self._router.route()
        return super().read()

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._router.unload()


class SoundcardBackend:
    """Real devices through `soundcard`: WASAPI loopback on Windows, a PulseAudio or PipeWire monitor
    on Linux. macOS cannot capture what it plays, so "loopback" there needs a loopback device
    (BlackHole) chosen as the output the system plays to."""

    def __init__(self, platform: str = None, run=subprocess.run, apps=None) -> None:
        self.platform = platform or sys.platform
        self._run = run
        self._apps = apps       # a list of program names, or a function returning it; empty means everything

    def app_names(self) -> list:
        names = self._apps() if callable(self._apps) else self._apps
        return [name for name in (names or []) if isinstance(name, str) and name.strip()]

    def _key(self) -> str:
        return "darwin" if self.platform == "darwin" else "win32" if self.platform.startswith("win") else "linux"

    def open_source(self, kind: str):
        if kind == "loopback" and self.app_names() and self._key() != "linux":
            raise AudioUnavailable("sharing the sound of chosen programs works on Linux only; "
                                   "clear the program list to share all of this computer's sound")
        sc, np = _load()
        try:
            if kind == "mic":
                device, channels = sc.default_microphone(), 1
                if device is None:
                    raise AudioUnavailable("no microphone found")
            elif kind == "loopback" and self.app_names():
                channels = 2
                router = LinuxAppRouter(self.app_names(), self._run)
                try:
                    monitor = router.load()
                    device = sc.get_microphone(monitor, include_loopback=True)
                    recorder = device.recorder(samplerate=RATE, channels=channels, blocksize=FRAME_SAMPLES)
                    recorder.__enter__()
                except Exception:
                    router.unload()
                    raise
                return _RoutedSource(recorder, np, channels, router)
            elif kind == "loopback":
                channels = 2
                if self._key() == "darwin":
                    device = _named(sc.all_microphones(), LOOPBACK_DEVICES["darwin"])
                    if device is None:
                        raise AudioUnavailable("macOS cannot capture its own sound; install a loopback "
                                               "device such as BlackHole and play to it")
                else:
                    speaker = sc.default_speaker()
                    device = sc.get_microphone(speaker.name, include_loopback=True) if speaker else None
                    if device is None:
                        raise AudioUnavailable("no output device to capture the sound of")
            else:
                raise AudioUnavailable("unknown source %r" % kind)
            recorder = device.recorder(samplerate=RATE, channels=channels, blocksize=FRAME_SAMPLES)
            recorder.__enter__()
        except AudioUnavailable:
            raise
        except Exception as error:
            raise AudioUnavailable("could not open the %s: %s" % (kind, error)) from error
        return _Source(recorder, np, channels)

    def open_sink(self, kind: str):
        sc, np = _load()
        on_close, channels = None, 2
        try:
            if kind == "speaker":
                device = sc.default_speaker()
                if device is None:
                    raise AudioUnavailable("no output device found")
            elif kind == "virtual_mic":
                channels = 1
                if self._key() == "linux":
                    cable = LinuxVirtualMic(self._run)
                    name = cable.load()
                    on_close = cable.unload
                    device = sc.get_speaker(name)
                else:
                    names = VIRTUAL_CABLES.get(self._key(), ())
                    device = _named(sc.all_speakers(), names)
                    if device is None:
                        raise AudioUnavailable("no virtual audio cable found (install %s) to play the "
                                               "microphone into" % " or ".join(names or ("one",)))
            else:
                raise AudioUnavailable("unknown sink %r" % kind)
            player = device.player(samplerate=RATE, channels=channels, blocksize=FRAME_SAMPLES)
            player.__enter__()
        except AudioUnavailable:
            if on_close is not None:
                on_close()
            raise
        except Exception as error:
            if on_close is not None:
                on_close()
            raise AudioUnavailable("could not open the %s: %s" % (kind, error)) from error
        return _Sink(player, np, channels, on_close)
