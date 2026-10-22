"""This computer's own system sounds and output volume, for the audio alert and volume-sync messages.

Every call here is best effort: a missing tool, library or device gives False or None, never an
exception. Each function takes the platform and the command runner so a test can fake both.

  alert sound   Windows winsound.MessageBeep; Mac afplay of a system sound (else osascript beep);
                Linux canberra-gtk-play (else paplay of a freedesktop sound).
  volume        Windows IAudioEndpointVolume through pycaw (optional; absent means unavailable);
                Mac osascript; Linux pactl on the default sink."""

import logging
import re
import subprocess
import sys
import threading
from typing import Callable, Optional, Tuple

from . import protocol

LOGGER = logging.getLogger(__name__)

POLL_SECONDS = 0.5
CHANGE = 0.01       # a level that moved less than this is the same level
TIMEOUT = 3

MAC_SOUNDS = {"default": "Tink", "error": "Basso", "notify": "Glass"}
LINUX_EVENTS = {"default": "bell", "error": "dialog-error", "notify": "message"}
LINUX_FILES = {"default": "bell.oga", "error": "dialog-error.oga", "notify": "message.oga"}
WIN_BEEPS = {"default": 0, "error": 0x10, "notify": 0x40}     # MB_OK, MB_ICONHAND, MB_ICONASTERISK


def _key(platform: Optional[str]) -> str:
    text = platform or sys.platform
    return "win32" if text.startswith("win") else "darwin" if text == "darwin" else "linux"


def _run(run, command) -> Optional[str]:
    """The output of a command, or None when it is missing or fails."""
    try:
        done = run(command, capture_output=True, text=True, timeout=TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if getattr(done, "returncode", 1) == 0 else None


def play_alert(kind: str, platform: str = None, run=subprocess.run, winsound=None) -> bool:
    """Play this computer's own sound for `kind`. Returns True when something was started."""
    if kind not in protocol.AUDIO_ALERT_KINDS:
        return False
    system = _key(platform)
    if system == "win32":
        try:
            if winsound is None:
                import winsound
            winsound.MessageBeep(WIN_BEEPS[kind])
            return True
        except Exception:
            return False
    if system == "darwin":
        path = "/System/Library/Sounds/%s.aiff" % MAC_SOUNDS[kind]
        if _run(run, ["afplay", path]) is not None:
            return True
        return _run(run, ["osascript", "-e", "beep"]) is not None
    if _run(run, ["canberra-gtk-play", "-i", LINUX_EVENTS[kind]]) is not None:
        return True
    return _run(run, ["paplay", "/usr/share/sounds/freedesktop/stereo/" + LINUX_FILES[kind]]) is not None


def _pycaw_endpoint():
    """The default speakers' IAudioEndpointVolume, through pycaw. Raises when pycaw is not installed."""
    from pycaw.pycaw import AudioUtilities
    speakers = AudioUtilities.GetSpeakers()
    endpoint = getattr(speakers, "EndpointVolume", None)
    if endpoint is not None:
        return endpoint
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import IAudioEndpointVolume
    return speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None).QueryInterface(IAudioEndpointVolume)


class VolumeControl:
    """Reads and sets the default output's volume (0..1) and mute."""

    def __init__(self, platform: str = None, run=subprocess.run, windows_loader: Callable = _pycaw_endpoint) -> None:
        self._system = _key(platform)
        self._run = run
        self._windows_loader = windows_loader

    def get(self) -> Optional[Tuple[float, bool]]:
        try:
            if self._system == "win32":
                endpoint = self._windows_loader()
                return float(endpoint.GetMasterVolumeLevelScalar()), bool(endpoint.GetMute())
            if self._system == "darwin":
                text = _run(self._run, ["osascript", "-e", "output volume of (get volume settings) & \",\" & "
                                        "output muted of (get volume settings)"])
                if text is None:
                    return None
                level, muted = text.strip().split(",")
                return max(0.0, min(1.0, int(level) / 100.0)), muted.strip() == "true"
            volume = _run(self._run, ["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
            mute = _run(self._run, ["pactl", "get-sink-mute", "@DEFAULT_SINK@"])
            found = re.search(r"(\d+)%", volume or "")
            if not found or mute is None:
                return None
            return max(0.0, min(1.0, int(found.group(1)) / 100.0)), "yes" in mute.lower()
        except Exception:
            return None

    def set(self, level: float, muted: bool) -> bool:
        if not isinstance(level, (int, float)) or isinstance(level, bool) or not 0.0 <= level <= 1.0:
            return False
        try:
            if self._system == "win32":
                endpoint = self._windows_loader()
                endpoint.SetMasterVolumeLevelScalar(float(level), None)
                endpoint.SetMute(1 if muted else 0, None)
                return True
            if self._system == "darwin":
                script = "set volume output volume %d output muted %s" % (round(level * 100), "true" if muted else "false")
                return _run(self._run, ["osascript", "-e", script]) is not None
            done = _run(self._run, ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "%d%%" % round(level * 100)])
            mute = _run(self._run, ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1" if muted else "0"])
            return done is not None and mute is not None
        except Exception:
            return False


class VolumePoller:
    """Calls `on_change(level, muted)` when the local volume or mute changes.

    A value the peer sent is passed to `applied()` before it is set locally, which makes it the new
    baseline: the next poll sees no change, so a volume that came from the peer is never sent back."""

    def __init__(self, control: VolumeControl, on_change: Callable[[float, bool], None], interval: float = POLL_SECONDS) -> None:
        self._control, self._on_change, self._interval = control, on_change, interval
        self._last: Optional[Tuple[float, bool]] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def applied(self, level: float, muted: bool) -> None:
        with self._lock:
            self._last = (float(level), bool(muted))

    def poll(self) -> bool:
        """One look at the volume. Returns True when a change was reported."""
        now = self._control.get()
        if now is None:
            return False
        with self._lock:
            last, self._last = self._last, now
        if last is None:        # the first look is the baseline, not a change
            return False
        if abs(now[0] - last[0]) < CHANGE and now[1] == last[1]:
            self._last = last
            return False
        try:
            self._on_change(now[0], now[1])
        except Exception:
            LOGGER.exception("audio: could not send the volume")
        return True

    def apply(self, level: float, muted: bool) -> bool:
        """Set the local volume from the peer's, without it echoing back."""
        self.applied(level, muted)
        return self._control.set(level, muted)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="audio-volume", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.poll()

    def close(self) -> None:
        self._stop.set()
