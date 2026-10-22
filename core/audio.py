"""Audio between computers (WIRE.md section 3, "Audio").

One computer is the host: it plays the other's sound and shares its microphone, which the other
plays into a virtual microphone. In "follow" mode the host is the computer that owns the input, and
only while the input is on the other one. In "fixed" mode one chosen computer is the host all the
time, whatever owns the input. `elect` decides, per link, who is who from both sides' settings.

Nothing here touches a socket or a sound card: messages go out through a `send` callable and devices
come from a backend (`audio_io`), so it is all testable with fakes. Whatever fails (no library, no
device, no cable) is logged and that direction stays off; the link and the app carry on."""

import logging
import threading
from typing import Callable, Optional

from . import audio_aec, audio_codec, audio_io, audio_stream, protocol
from .i18n import tr

LOGGER = logging.getLogger(__name__)

SOUND_CHANNELS = 2
MIC_CHANNELS = 1


def frame_bytes(channels: int) -> int:
    return protocol.audio_frame_bytes(audio_io.RATE, channels, audio_io.FRAME_MS)


class Streamer:
    """Reads frames from `source` and sends each as an `audio_data`. `step()` is one frame;
    the thread repeats it until stopped or the source fails (`on_fail(text)` then)."""

    def __init__(self, source, direction: str, send: Callable[[dict], bool], on_fail=None, encoder=None) -> None:
        self.source = source
        self.encoder = encoder      # an Opus encoder, or None to send s16le
        self.direction = direction
        self._send = send
        self._on_fail = on_fail
        self.seq = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def step(self) -> None:
        pcm = self.source.read()
        format = "s16le"
        if self.encoder is not None:
            pcm, format = self.encoder.encode(pcm), "opus"
        # A full queue drops this frame; the sequence still moves, so the player hears a gap, not a shift.
        self._send(protocol.audio_data_msg(self.direction, self.seq, pcm, format))
        self.seq = (self.seq + 1) & protocol.AUDIO_MAX_SEQ

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="audio-" + self.direction, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self.step()
        except Exception as error:
            if not self._stop.is_set():
                LOGGER.exception("audio %s capture stopped", self.direction)
                if self._on_fail is not None:
                    self._on_fail(tr("capture failed: %s") % error)

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        try:
            self.source.close()
        except Exception:
            LOGGER.exception("audio %s source did not close", self.direction)


def elect(local, remote, me: str, peer: str):
    """The role of this machine on its link to `peer`, from the audio settings of both sides.

    `local` and `remote` are (mode, host) with host a machine id text; `remote` is None until the
    peer has announced its setting. Returns (role, note): role is "follow" (the owner of the input
    is the host), "host", "guest" or "none"; note is a sentence when this side gave way or the link
    cannot be served, else None. A side that fixed a host wins over one that follows; two sides that
    fixed different hosts are settled by the lower machine id, so both reach the same answer."""
    local_fixed = local is not None and local[0] == "fixed" and bool(local[1])
    remote_fixed = remote is not None and remote[0] == "fixed" and bool(remote[1])
    note = None
    if not local_fixed and not remote_fixed:
        return "follow", None
    if local_fixed and remote_fixed and local[1] != remote[1]:
        if me < peer:
            host = local[1]
        else:
            host = remote[1]
            note = "both computers fixed an audio host; the one with the lower id wins, so the other's choice is used"
    elif local_fixed:
        host = local[1]
    else:
        host = remote[1]
        note = "the other computer fixed the audio host; following its choice"
    if host == me:
        return "host", note
    if host == peer:
        return "guest", note
    return "none", note or "the audio host is a third computer; no audio on this link"


class _HostSide:
    """What this machine runs for one peer whose sound it plays and to which it sends its mic."""

    def __init__(self) -> None:
        self.sound_wanted = False
        self.sound_play: Optional[audio_stream.Playback] = None
        self.mic_send: Optional[Streamer] = None
        self.echo = None          # the EchoCanceller shared by the sound played and the mic sent


class _GuestSide:
    """What this machine runs for the host: its sound out, the host's microphone in."""

    def __init__(self) -> None:
        self.sound_send: Optional[Streamer] = None
        self.mic_play: Optional[audio_stream.Playback] = None


class AudioController:
    """Both roles of one machine, kept per peer. `input_moved` (this machine owns the input and it
    is now on a peer, or None) and `owner_changed` (a peer owns this machine's input, or None) drive
    "follow" mode; `link_up`/`link_down` and `refresh` (a setting changed) drive "fixed" mode.
    `receive` takes what a link delivered, already checked by `protocol.read_audio`.

    Peers may be bytes (Windows) or id text (macOS); all are kept as id text inside.
    `send(peer, message)` returns False when the message could not be queued; `caps_of(peer)` is
    the capability set of the link; `config()` is (mode, host) from the settings, host "" meaning
    this machine; `me` is this machine's id text (or a function returning it)."""

    def __init__(
        self,
        send: Callable[[object, dict], bool],
        caps_of: Callable[[object], Optional[frozenset]],
        me: str,
        backend=None,
        want_sound: Callable[[], bool] = lambda: False,
        want_mic: Callable[[], bool] = lambda: False,
        config: Callable[[], tuple] = lambda: ("follow", ""),
        status: Callable[[str], None] = None,
        want_opus: Callable[[], bool] = lambda: False,
        want_echo_cancel: Callable[[], bool] = lambda: False,
        codec_loader=None,
        apps: Callable[[], list] = lambda: [],
        want_alerts: Callable[[], bool] = lambda: False,
        play_alert: Callable[[str], None] = None,
        want_volume: Callable[[], bool] = lambda: False,
        apply_volume: Callable[[float, bool], None] = None,
    ) -> None:
        self._send_fn = send
        self._caps_of = caps_of
        self._me_source = me
        self._backend = backend
        self._want_sound = want_sound
        self._want_mic = want_mic
        self._config = config
        self._status = status
        self._want_opus = want_opus
        self._want_echo = want_echo_cancel
        self._codec_loader = codec_loader
        self._apps = apps
        self._want_alerts = want_alerts
        self._play_alert = play_alert
        self._want_volume = want_volume
        self._apply_volume = apply_volume
        self._lock = threading.RLock()
        self._raw = {}            # id text -> the peer as the app holds it
        self._up = set()
        self._remote = {}         # id text -> (mode, host text or None) as that peer announced
        self._driven = None       # follow mode: the peer the input is on
        self._owner = None        # follow mode: the peer that owns this machine's input
        self._hosts = {}
        self._guests = {}
        self._notes = {}

    @property
    def _me(self) -> str:
        value = self._me_source() if callable(self._me_source) else self._me_source
        return value or ""

    # -- helpers ------------------------------------------------------------

    def _key(self, peer) -> str:
        key = protocol.peer_id_text(peer)
        self._raw[key] = peer
        return key

    def _send(self, key: str, message: dict) -> bool:
        return bool(self._send_fn(self._raw[key], message))

    def _devices(self):
        if self._backend is None:
            self._backend = audio_io.SoundcardBackend(apps=self._apps)
        return self._backend

    def usable(self) -> bool:
        return self._backend is not None or audio_io.available()

    def _say(self, text: str) -> None:
        LOGGER.warning("audio: %s", text)
        if self._status is not None:
            try:
                self._status(text)
            except Exception:
                LOGGER.exception("the audio status callback failed")

    def _local(self):
        mode, host = self._config()
        if mode != "fixed":
            return ("follow", None)
        return ("fixed", host or self._me)

    def _playback(self, sink_kind: str, channels: int, name: str, decoder=None, wrap=None) -> audio_stream.Playback:
        sink = self._devices().open_sink(sink_kind)
        if wrap is not None:
            sink = wrap(sink)
        buffer = audio_stream.JitterBuffer(frame_bytes(channels), audio_io.FRAME_MS)
        playback = audio_stream.Playback(sink, buffer, name, decoder)
        playback.start()
        return playback

    # -- Opus and echo cancellation -----------------------------------------

    def _codec_kwargs(self) -> dict:
        return {} if self._codec_loader is None else {"loader": self._codec_loader}

    def _opus_ok(self) -> bool:
        """True when the setting is on and the codec can be found."""
        try:
            wanted = bool(self._want_opus())
        except Exception:
            return False
        return wanted and (self._codec_loader is not None or audio_codec.available())

    def _peer_opus(self, key: str) -> bool:
        return "audio_opus" in (self._caps_of(self._raw[key]) or ())

    def _encoder(self, channels: int):
        """An Opus encoder, or None (noted in the log) when it cannot be had."""
        try:
            return audio_codec.Encoder(channels, **self._codec_kwargs())
        except audio_io.AudioUnavailable as error:
            LOGGER.warning("audio: Opus is not usable, sending raw sound: %s", error)
            return None

    def _decoder(self, channels: int):
        try:
            return audio_codec.Decoder(channels, **self._codec_kwargs())
        except audio_io.AudioUnavailable as error:
            LOGGER.warning("audio: Opus is not usable here: %s", error)
            return None

    def _echo_for(self, side: "_HostSide"):
        """The canceller of this host side when it plays the other's sound and sends its mic."""
        if side.echo is None and self._want_echo() and self._want_sound() and self._want_mic():
            try:
                side.echo = audio_aec.EchoCanceller()
            except Exception as error:   # numpy missing: no cancellation, everything else carries on
                LOGGER.warning("audio: echo cancellation is off: %s", error)
        return side.echo

    @staticmethod
    def _stop_all(*parts) -> None:
        for part in parts:
            if part is not None:
                part.stop()

    # -- roles --------------------------------------------------------------

    def _role(self, key: str) -> str:
        if key not in self._up or "audio" not in (self._caps_of(self._raw[key]) or ()):
            return "none"
        role, note = elect(self._local(), self._remote.get(key), self._me, key)
        if note != self._notes.get(key):
            self._notes[key] = note
            if note:
                LOGGER.warning("audio with %s: %s", key[:8], note)
        if role == "follow":
            return "host" if key == self._driven else "guest" if key == self._owner else "none"
        return role

    def _reconcile(self) -> None:
        roles = {key: self._role(key) for key in self._up}
        for key in [k for k in self._hosts if roles.get(k) != "host"]:
            self._end_host(key, send=True)
        for key in [k for k in self._guests if roles.get(k) != "guest"]:
            self._end_guest(key)
        for key, role in roles.items():
            if role == "host":
                if key not in self._hosts:
                    self._hosts[key] = _HostSide()
                self._apply_host(key)
            elif role == "guest" and key not in self._guests:
                self._guests[key] = _GuestSide()

    def _announce(self, key: str) -> None:
        if "audio" not in (self._caps_of(self._raw[key]) or ()):
            return
        mode, host = self._local()
        self._send(key, protocol.audio_config_msg(mode, protocol.read_id(host) if host else None))

    def _ensure(self, peer) -> str:
        key = self._key(peer)
        if key not in self._up:
            self._up.add(key)
            self._announce(key)
        return key

    # -- events from the app ------------------------------------------------

    def input_moved(self, peer) -> None:
        """The input is now on `peer`, or (None) back on this machine."""
        with self._lock:
            self._driven = None if peer is None else self._ensure(peer)
            self._reconcile()

    def owner_changed(self, peer) -> None:
        """`peer` took the input of this machine, or (None) let it go."""
        with self._lock:
            self._owner = None if peer is None else self._ensure(peer)
            self._reconcile()

    def link_up(self, peer) -> None:
        with self._lock:
            self._ensure(peer)
            self._reconcile()

    def link_down(self, peer) -> None:
        with self._lock:
            key = protocol.peer_id_text(peer)
            if key not in self._raw:
                return
            self._up.discard(key)
            self._remote.pop(key, None)
            self._notes.pop(key, None)
            if key == self._driven:
                self._driven = None
            if key == self._owner:
                self._owner = None
            self._end_host(key, send=False)
            self._end_guest(key)

    def refresh(self) -> None:
        """A setting changed: tell every peer, and start or stop what the settings now ask for."""
        with self._lock:
            for key in self._up:
                self._announce(key)
            self._reconcile()

    def close(self) -> None:
        with self._lock:
            for key in list(self._hosts):
                self._end_host(key, send=False)
            for key in list(self._guests):
                self._end_guest(key)
            self._driven = self._owner = None
            self._up.clear()

    # -- the host side ------------------------------------------------------

    def _apply_host(self, key: str) -> None:
        side = self._hosts[key]
        ok = self.usable()
        want_sound = ok and bool(self._want_sound())
        want_mic = ok and bool(self._want_mic())
        if want_sound and not side.sound_wanted:
            side.sound_wanted = True
            fmt = "opus" if self._opus_ok() and self._peer_opus(key) and self._decoder(SOUND_CHANNELS) else "s16le"
            self._send(key, protocol.audio_want_msg("out", audio_io.RATE, SOUND_CHANNELS, fmt, audio_io.FRAME_MS))
        elif not want_sound and side.sound_wanted:
            self._stop_sound_play(side)
            self._send(key, protocol.audio_stop_msg("out"))
        if want_mic and side.mic_send is None:
            try:
                source = self._devices().open_source("mic")
            except audio_io.AudioUnavailable as error:
                self._say(tr("microphone sharing is off: %s") % error)
            else:
                echo = self._echo_for(side)
                if echo is not None:
                    source = audio_aec.EchoSource(source, echo)
                encoder = self._encoder(MIC_CHANNELS) if self._opus_ok() and self._peer_opus(key) else None
                self._send(key, protocol.audio_start_msg("mic", audio_io.RATE, MIC_CHANNELS,
                                                         "opus" if encoder else "s16le", audio_io.FRAME_MS))
                side.mic_send = Streamer(source, "mic", lambda message: self._send(key, message),
                                         lambda text: self._mic_failed(key, text), encoder)
                side.mic_send.start()
        elif not want_mic and side.mic_send is not None:
            self._stop_mic_send(side)
            self._send(key, protocol.audio_stop_msg("mic"))

    def _mic_failed(self, key: str, text: str) -> None:
        with self._lock:
            side = self._hosts.get(key)
            if side is not None and side.mic_send is not None:
                side.mic_send = None
                self._send(key, protocol.audio_stop_msg("mic", "failed"))
        self._say(tr("microphone sharing stopped: %s") % text)

    def _stop_sound_play(self, side: _HostSide) -> None:
        play, side.sound_play, side.sound_wanted = side.sound_play, None, False
        self._stop_all(play)

    def _stop_mic_send(self, side: _HostSide) -> None:
        mic, side.mic_send = side.mic_send, None
        self._stop_all(mic)

    def _end_host(self, key: str, send: bool) -> None:
        side = self._hosts.pop(key, None)
        if side is None:
            return
        had_sound, had_mic = side.sound_wanted, side.mic_send is not None
        self._stop_sound_play(side)
        self._stop_mic_send(side)
        if send and key in self._up:
            if had_sound:
                self._send(key, protocol.audio_stop_msg("out"))
            if had_mic:
                self._send(key, protocol.audio_stop_msg("mic"))

    def _from_guest(self, key: str, side: _HostSide, kind: str, fields: dict) -> None:
        if kind == protocol.MSG_AUDIO_START and fields["dir"] == "out":
            if not side.sound_wanted or side.sound_play is not None:
                return
            if fields["channels"] != SOUND_CHANNELS:
                self._send(key, protocol.audio_stop_msg("out", "format"))
                return
            decoder = None
            if fields.get("format", "s16le") == "opus":
                decoder = self._decoder(SOUND_CHANNELS) if self._opus_ok() else None
                if decoder is None:
                    self._send(key, protocol.audio_stop_msg("out", "format"))
                    return
            echo = self._echo_for(side)
            wrap = None if echo is None else (lambda sink: audio_aec.EchoSink(sink, echo, SOUND_CHANNELS))
            try:
                side.sound_play = self._playback("speaker", SOUND_CHANNELS, "audio-sound", decoder, wrap)
            except audio_io.AudioUnavailable as error:
                self._say(tr("the other computer's sound is off: %s") % error)
                side.sound_wanted = False
                self._send(key, protocol.audio_stop_msg("out", "unavailable"))
        elif kind == protocol.MSG_AUDIO_DATA and fields["dir"] == "out":
            if side.sound_play is not None:
                side.sound_play.push(fields["seq"], fields["pcm"], fields.get("format", "s16le"))
        elif kind == protocol.MSG_AUDIO_STOP:
            if fields["dir"] == "out":
                self._stop_sound_play(side)
            else:
                self._stop_mic_send(side)

    # -- the guest side -----------------------------------------------------

    def _end_guest(self, key: str) -> None:
        side = self._guests.pop(key, None)
        if side is not None:
            self._stop_all(side.sound_send, side.mic_play)

    def _from_host(self, key: str, side: _GuestSide, kind: str, fields: dict) -> None:
        direction = fields["dir"]
        if kind == protocol.MSG_AUDIO_WANT and direction == "out":
            self._start_sound(key, side, fields)
        elif kind == protocol.MSG_AUDIO_START and direction == "mic":
            self._start_mic(key, side, fields)
        elif kind == protocol.MSG_AUDIO_DATA and direction == "mic":
            if side.mic_play is not None:
                side.mic_play.push(fields["seq"], fields["pcm"], fields.get("format", "s16le"))
        elif kind == protocol.MSG_AUDIO_STOP:
            if direction == "out":
                sound, side.sound_send = side.sound_send, None
                self._stop_all(sound)
            else:
                play, side.mic_play = side.mic_play, None
                self._stop_all(play)

    def _start_sound(self, key: str, side: _GuestSide, fields: dict) -> None:
        sound, side.sound_send = side.sound_send, None
        self._stop_all(sound)
        if fields["channels"] != SOUND_CHANNELS or not self.usable():
            self._send(key, protocol.audio_stop_msg("out", "unavailable"))
            return
        try:
            source = self._devices().open_source("loopback")
        except audio_io.AudioUnavailable as error:
            self._say(tr("cannot send this computer's sound: %s") % error)
            self._send(key, protocol.audio_stop_msg("out", "unavailable"))
            return
        encoder = self._encoder(SOUND_CHANNELS) if fields.get("format", "s16le") == "opus" and self._opus_ok() else None
        self._send(key, protocol.audio_start_msg("out", audio_io.RATE, SOUND_CHANNELS,
                                                 "opus" if encoder else "s16le", audio_io.FRAME_MS))
        side.sound_send = Streamer(source, "out", lambda message: self._send(key, message),
                                   lambda text: self._say(tr("sending this computer's sound stopped: %s") % text), encoder)
        side.sound_send.start()

    def _start_mic(self, key: str, side: _GuestSide, fields: dict) -> None:
        play, side.mic_play = side.mic_play, None
        self._stop_all(play)
        if fields["channels"] != MIC_CHANNELS or not self.usable():
            self._send(key, protocol.audio_stop_msg("mic", "unavailable"))
            return
        decoder = None
        if fields.get("format", "s16le") == "opus":
            decoder = self._decoder(MIC_CHANNELS) if self._opus_ok() else None
            if decoder is None:
                self._send(key, protocol.audio_stop_msg("mic", "format"))
                return
        try:
            side.mic_play = self._playback("virtual_mic", MIC_CHANNELS, "audio-mic", decoder)
        except audio_io.AudioUnavailable as error:
            self._say(tr("cannot play the other computer's microphone: %s") % error)
            self._send(key, protocol.audio_stop_msg("mic", "unavailable"))

    # -- alert sounds and volume (no stream, any linked peer) ---------------

    def _audio_peers(self) -> list:
        return [key for key in self._up if "audio" in (self._caps_of(self._raw[key]) or ())]

    def send_alert(self, kind: str = "default") -> int:
        """Ask every linked peer to play its own system sound `kind` (when the setting is on).
        Returns how many were asked."""
        if kind not in protocol.AUDIO_ALERT_KINDS or not self._want_alerts():
            return 0
        with self._lock:
            return sum(1 for key in self._audio_peers() if self._send(key, protocol.audio_alert_msg(kind)))

    def _alert_in(self, key: str, fields: dict) -> None:
        if key not in self._up or not self._want_alerts() or self._play_alert is None:
            return
        try:
            self._play_alert(fields["kind"])
        except Exception:
            LOGGER.exception("audio: could not play the alert sound")

    def send_volume(self, level: float, muted: bool) -> int:
        """Tell every linked peer this machine's output volume and mute (when the setting is on)."""
        if not self._want_volume():
            return 0
        with self._lock:
            return sum(1 for key in self._audio_peers() if self._send(key, protocol.audio_volume_msg(level, muted)))

    def _volume_in(self, key: str, fields: dict) -> None:
        if key not in self._up or not self._want_volume() or self._apply_volume is None:
            return
        try:
            self._apply_volume(fields["level"], fields["muted"])
        except Exception:
            LOGGER.exception("audio: could not apply the other computer's volume")

    # -- messages -----------------------------------------------------------

    def receive(self, peer, kind: str, fields: dict) -> None:
        """A checked audio message from `peer`. A stream message counts only from a peer this
        machine is serving or hosting right now, so nothing else can start or feed a stream."""
        with self._lock:
            key = self._key(peer)
            if kind == protocol.MSG_AUDIO_CONFIG:
                host = fields["host"]
                self._remote[key] = (fields["mode"], protocol.id_text(host) if host else None)
                self._reconcile()
            elif kind == protocol.MSG_AUDIO_ALERT:
                self._alert_in(key, fields)
            elif kind == protocol.MSG_AUDIO_VOLUME:
                self._volume_in(key, fields)
            elif key in self._hosts:
                self._from_guest(key, self._hosts[key], kind, fields)
            elif key in self._guests:
                self._from_host(key, self._guests[key], kind, fields)

    from_driven = from_owner = receive
