import math
import unittest

from core import audio, audio_aec, audio_codec, audio_io, link, protocol
from core.tests.test_audio import FRAME, ME, MIC_FRAME, PEER, FakeBackend, wait_for


class FakeOpusEncoder:
    def __init__(self, rate, channels, application):
        self.bitrate = 0
        self.channels = channels

    def encode(self, pcm, samples):
        return b"OP" + pcm[: len(pcm) // 8]


class FakeOpusDecoder:
    def __init__(self, rate, channels):
        self.channels = channels

    def decode(self, packet, samples):
        if not packet.startswith(b"OP"):
            raise ValueError("corrupted")
        return bytes(samples * 2 * self.channels)


class FakeOpusLib:
    Encoder = FakeOpusEncoder
    Decoder = FakeOpusDecoder


def fake_lib():
    return FakeOpusLib


def missing_lib():
    raise audio_io.AudioUnavailable("no opus")


class ProtocolPass2Tests(unittest.TestCase):
    def test_opus_data_round_trips_and_keeps_its_format(self):
        got = protocol.read_audio(protocol.audio_data_msg("out", 3, b"x" * 120, "opus"))
        self.assertEqual(got[1]["format"], "opus")
        self.assertEqual(got[1]["pcm"], b"x" * 120)

    def test_s16le_data_has_no_format_field_and_reads_as_s16le(self):
        msg = protocol.audio_data_msg("out", 3, bytes(FRAME))
        self.assertNotIn("format", msg["data"])
        self.assertEqual(protocol.read_audio(msg)[1]["format"], "s16le")

    def test_opus_limits(self):
        for size in (0, protocol.AUDIO_MAX_OPUS_BYTES + 1):
            msg = protocol.audio_data_msg("out", 1, b"x" * size, "opus")
            self.assertIsNone(protocol.read_audio(msg))
        msg = protocol.audio_data_msg("out", 1, b"x" * protocol.AUDIO_MAX_OPUS_BYTES, "opus")
        self.assertIsNotNone(protocol.read_audio(msg))

    def test_unknown_format_is_refused(self):
        msg = protocol.audio_data_msg("out", 1, bytes(FRAME))
        msg["data"]["format"] = "flac"
        self.assertIsNone(protocol.read_audio(msg))

    def test_alert_allow_list(self):
        for kind in protocol.AUDIO_ALERT_KINDS:
            self.assertEqual(protocol.read_audio(protocol.audio_alert_msg(kind))[1]["kind"], kind)
        for bad in ("../x", "", None, 3, "DEFAULT"):
            self.assertIsNone(protocol.read_audio({"type": protocol.MSG_AUDIO_ALERT, "data": {"kind": bad}}))
        self.assertIsNone(protocol.read_audio({"type": protocol.MSG_AUDIO_ALERT, "data": {}}))

    def test_volume_validation(self):
        got = protocol.read_audio(protocol.audio_volume_msg(0.25, True))
        self.assertEqual(got[1], {"level": 0.25, "muted": True})
        for level in (-0.1, 1.5, float("nan"), "loud", None, True):
            msg = {"type": protocol.MSG_AUDIO_VOLUME, "data": {"level": level, "muted": False}}
            self.assertIsNone(protocol.read_audio(msg), level)
        msg = {"type": protocol.MSG_AUDIO_VOLUME, "data": {"level": 0.5, "muted": "no"}}
        self.assertIsNone(protocol.read_audio(msg))

    def test_new_messages_need_the_audio_capability(self):
        self.assertEqual(link._CAP_FOR[protocol.MSG_AUDIO_ALERT], "audio")
        self.assertEqual(link._CAP_FOR[protocol.MSG_AUDIO_VOLUME], "audio")


class CodecTests(unittest.TestCase):
    def test_round_trip_with_a_fake_library(self):
        encoder = audio_codec.Encoder(2, loader=fake_lib)
        packet = encoder.encode(bytes(FRAME))
        self.assertTrue(packet.startswith(b"OP"))
        self.assertEqual(len(audio_codec.Decoder(2, loader=fake_lib).decode(packet)), FRAME)

    def test_bad_packet_raises_value_error(self):
        with self.assertRaises(ValueError):
            audio_codec.Decoder(2, loader=fake_lib).decode(b"junk")

    def test_missing_library_is_audio_unavailable(self):
        with self.assertRaises(audio_io.AudioUnavailable):
            audio_codec.Encoder(2, loader=missing_lib)
        with self.assertRaises(audio_io.AudioUnavailable):
            audio_codec.Decoder(2, loader=missing_lib)


class OpusRig:
    def __init__(self, opus=True, loader=fake_lib, caps=frozenset({"audio", "audio_opus"}), mic=False):
        self.sent = []
        self.backend = FakeBackend()
        self.controller = audio.AudioController(
            lambda peer, msg: self.sent.append((peer, msg)) or True,
            lambda peer: caps,
            protocol.id_text(ME),
            backend=self.backend,
            want_sound=lambda: True,
            want_mic=lambda: mic,
            config=lambda: ("follow", ""),
            want_opus=lambda: opus,
            codec_loader=loader,
        )

    def of(self, kind):
        return [m["data"] for p, m in self.sent if m["type"] == kind]


class OpusNegotiationTests(unittest.TestCase):
    def host_asks(self, **kw):
        r = OpusRig(**kw)
        self.addCleanup(r.controller.close)
        r.controller.input_moved(PEER)
        return r

    def test_host_asks_for_opus_when_both_have_it(self):
        self.assertEqual(self.host_asks().of(protocol.MSG_AUDIO_WANT)[0]["format"], "opus")

    def test_peer_without_the_capability_gets_s16le(self):
        r = self.host_asks(caps=frozenset({"audio"}))
        self.assertEqual(r.of(protocol.MSG_AUDIO_WANT)[0]["format"], "s16le")

    def test_setting_off_gets_s16le(self):
        self.assertEqual(self.host_asks(opus=False).of(protocol.MSG_AUDIO_WANT)[0]["format"], "s16le")

    def test_no_codec_gets_s16le(self):
        self.assertEqual(self.host_asks(loader=missing_lib).of(protocol.MSG_AUDIO_WANT)[0]["format"], "s16le")

    def test_guest_sends_opus_frames_when_asked(self):
        r = OpusRig()
        self.addCleanup(r.controller.close)
        r.controller.owner_changed(PEER)
        r.controller.receive(PEER, protocol.MSG_AUDIO_WANT,
                             {"dir": "out", "rate": 48000, "channels": 2, "format": "opus", "frame_ms": 20})
        self.assertEqual(r.of(protocol.MSG_AUDIO_START)[0]["format"], "opus")
        self.assertTrue(wait_for(lambda: any(d.get("format") == "opus" for d in r.of(protocol.MSG_AUDIO_DATA))))

    def test_guest_falls_back_to_s16le_when_it_has_no_codec(self):
        r = OpusRig(loader=missing_lib)
        self.addCleanup(r.controller.close)
        r.controller.owner_changed(PEER)
        r.controller.receive(PEER, protocol.MSG_AUDIO_WANT,
                             {"dir": "out", "rate": 48000, "channels": 2, "format": "opus", "frame_ms": 20})
        self.assertEqual(r.of(protocol.MSG_AUDIO_START)[0]["format"], "s16le")

    def test_host_refuses_an_opus_stream_it_cannot_decode(self):
        r = OpusRig(loader=missing_lib)
        self.addCleanup(r.controller.close)
        r.controller.input_moved(PEER)
        r.controller.receive(PEER, protocol.MSG_AUDIO_START,
                             {"dir": "out", "rate": 48000, "channels": 2, "format": "opus", "frame_ms": 20})
        self.assertEqual(r.backend.sinks, [])

    def test_host_plays_opus_and_survives_a_corrupt_packet(self):
        r = self.host_asks()
        r.controller.receive(PEER, protocol.MSG_AUDIO_START,
                             {"dir": "out", "rate": 48000, "channels": 2, "format": "opus", "frame_ms": 20})
        for n in range(8):
            r.controller.receive(PEER, protocol.MSG_AUDIO_DATA,
                                 {"dir": "out", "seq": n, "pcm": b"OP" + bytes(40), "format": "opus"})
        r.controller.receive(PEER, protocol.MSG_AUDIO_DATA,
                             {"dir": "out", "seq": 9, "pcm": b"junk", "format": "opus"})
        sink = r.backend.sinks[0][1]
        self.assertTrue(wait_for(lambda: len(sink.frames) > 0))


class EchoTests(unittest.TestCase):
    def erle(self, delay, gain=0.6, frames=300):
        import numpy as np
        rng = np.random.default_rng(1)
        b = audio_aec.BLOCK
        canceller = audio_aec.EchoCanceller()
        far = np.clip(np.convolve(rng.normal(0, 3000, frames * b), [1, 0.5, 0.3], "same"), -30000, 30000)
        echo = np.zeros_like(far)
        echo[delay:] = gain * far[:-delay]
        before = after = 0.0
        for t in range(frames):
            f = far[t * b:(t + 1) * b].astype("<i2")
            m = echo[t * b:(t + 1) * b].astype("<i2")
            canceller.far(np.repeat(f, 2).tobytes(), 2)
            out = np.frombuffer(canceller.process(m.tobytes()), dtype="<i2").astype(float)
            if t >= frames - 50:
                before += float((m.astype(float) ** 2).sum())
                after += float((out ** 2).sum())
        return 10 * math.log10(before / max(after, 1e-9))

    def test_echo_is_cancelled_by_at_least_15_db(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy missing")
        for delay in (960, 1100, 2500):
            self.assertGreaterEqual(self.erle(delay), 15, delay)

    def test_silence_leaves_the_frame_alone(self):
        canceller = audio_aec.EchoCanceller()
        frame = bytes(MIC_FRAME)
        canceller.far(bytes(FRAME), 2)
        self.assertEqual(canceller.process(frame), frame)

    def test_wrong_sized_frames_pass_through(self):
        canceller = audio_aec.EchoCanceller()
        self.assertEqual(canceller.process(b"\x01\x00" * 10), b"\x01\x00" * 10)


PACTL_LIST = """Sink Input #12
\tDriver: protocol-native.c
\tProperties:
\t\tapplication.name = "Firefox"
\t\tapplication.process.binary = "firefox"
Sink Input #15
\tProperties:
\t\tapplication.name = "Spotify"
\t\tapplication.process.binary = "spotify"
"""


class FakePactl:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        out = ""
        if cmd[1:3] == ["list", "sink-inputs"]:
            out = PACTL_LIST
        elif cmd[1] == "load-module":
            out = "42\n"

        class Result:
            returncode = 0
            stdout = out
            stderr = ""
        return Result()


class AppRoutingTests(unittest.TestCase):
    def test_parse_sink_inputs(self):
        streams = audio_io.parse_sink_inputs(PACTL_LIST)
        self.assertEqual([i for i, _ in streams], [12, 15])
        self.assertEqual(streams[1][1]["application.name"], "Spotify")

    def test_matching_is_case_insensitive_and_uses_the_binary(self):
        props = audio_io.parse_sink_inputs(PACTL_LIST)[0][1]
        self.assertTrue(audio_io.matches_app(props, ["FIRE"]))
        self.assertFalse(audio_io.matches_app(props, ["spotify"]))
        self.assertFalse(audio_io.matches_app(props, []))
        self.assertFalse(audio_io.matches_app(props, ["  "]))

    def test_router_moves_only_matching_streams_once_and_puts_them_back(self):
        fake = FakePactl()
        router = audio_io.LinuxAppRouter(["spotify"], fake)
        self.assertEqual(router.load(), audio_io.APPS_SINK + ".monitor")
        moves = [c for c in fake.calls if "move-sink-input" in c]
        self.assertEqual(moves, [[moves[0][0], "move-sink-input", "15", audio_io.APPS_SINK]])
        self.assertEqual(router.route(), 0)
        router.unload()
        text = [" ".join(c[1:]) for c in fake.calls]
        self.assertIn("move-sink-input 15 @DEFAULT_SINK@", text)
        self.assertIn("unload-module 42", text)

    def test_router_with_no_names_is_unavailable(self):
        with self.assertRaises(audio_io.AudioUnavailable):
            audio_io.LinuxAppRouter([], FakePactl()).load()

    def test_other_systems_say_linux_only(self):
        for platform in ("win32", "darwin"):
            backend = audio_io.SoundcardBackend(platform=platform, run=FakePactl(), apps=["spotify"])
            with self.assertRaises(audio_io.AudioUnavailable) as ctx:
                backend.open_source("loopback")
            self.assertIn("Linux", str(ctx.exception))


class AlertVolumeTests(unittest.TestCase):
    def make(self, alerts=True, volume=True, caps=frozenset({"audio"})):
        self.sent, self.played, self.applied = [], [], []
        c = audio.AudioController(
            lambda peer, msg: self.sent.append((peer, msg)) or True,
            lambda peer: caps,
            protocol.id_text(ME),
            backend=FakeBackend(),
            want_sound=lambda: False,
            want_mic=lambda: False,
            config=lambda: ("follow", ""),
            want_alerts=lambda: alerts,
            play_alert=self.played.append,
            want_volume=lambda: volume,
            apply_volume=lambda level, muted: self.applied.append((level, muted)),
        )
        self.addCleanup(c.close)
        c.link_up(PEER)
        self.sent.clear()
        return c

    def test_alert_is_sent_and_played(self):
        c = self.make()
        self.assertEqual(c.send_alert("error"), 1)
        c.receive(PEER, protocol.MSG_AUDIO_ALERT, {"kind": "error"})
        self.assertEqual(self.played, ["error"])

    def test_alert_off_or_unknown_kind_does_nothing(self):
        c = self.make(alerts=False)
        self.assertEqual(c.send_alert("default"), 0)
        c.receive(PEER, protocol.MSG_AUDIO_ALERT, {"kind": "default"})
        self.assertEqual(self.played, [])
        c = self.make()
        self.assertEqual(c.send_alert("rm -rf"), 0)

    def test_alert_from_a_peer_that_is_not_linked_is_ignored(self):
        c = self.make()
        c.receive(bytes([9]) * 16, protocol.MSG_AUDIO_ALERT, {"kind": "default"})
        self.assertEqual(self.played, [])

    def test_a_failing_player_does_not_raise(self):
        c = self.make()
        c._play_alert = lambda kind: 1 / 0
        c.receive(PEER, protocol.MSG_AUDIO_ALERT, {"kind": "default"})

    def test_volume_sent_and_applied(self):
        c = self.make()
        self.assertEqual(c.send_volume(0.4, False), 1)
        c.receive(PEER, protocol.MSG_AUDIO_VOLUME, {"level": 0.4, "muted": True})
        self.assertEqual(self.applied, [(0.4, True)])

    def test_volume_off_does_nothing(self):
        c = self.make(volume=False)
        self.assertEqual(c.send_volume(0.4, False), 0)
        c.receive(PEER, protocol.MSG_AUDIO_VOLUME, {"level": 0.4, "muted": True})
        self.assertEqual(self.applied, [])

    def test_peer_without_audio_capability_is_not_asked(self):
        c = self.make(caps=frozenset())
        self.assertEqual(c.send_alert("default"), 0)
        self.assertEqual(c.send_volume(0.4, False), 0)


class MediaKeyTests(unittest.TestCase):
    def test_windows_capture_knows_every_media_key(self):
        try:
            from win_app import capture_win
        except Exception as error:
            self.skipTest("capture_win not importable here: %s" % error)
        self.assertTrue(link.MEDIA_KEYS <= set(capture_win.VK_TO_NAME.values()))

    def test_x11_capture_knows_every_media_key(self):
        try:
            from win_app import capture_x11
        except Exception as error:
            self.skipTest("capture_x11 not importable here: %s" % error)
        names = getattr(capture_x11, "MEDIA_KEYSYMS", None)
        if names is None:
            self.skipTest("no media table")
        self.assertTrue(link.MEDIA_KEYS <= set(names))

    def test_mac_media_keys_cover_all_but_stop(self):
        from mac_app import media_keys
        self.assertEqual(link.MEDIA_KEYS - set(media_keys.NX_KEY_NAMES.values()), {"media_stop"})

    def test_keytable_has_every_media_key(self):
        from core import keytable
        names = set(keytable.__dict__.get("EVDEV_TO_NAME", {}).values()) or set()
        source = open(keytable.__file__, encoding="utf-8").read()
        for name in link.MEDIA_KEYS:
            self.assertIn('"%s"' % name, source)
        del names


if __name__ == "__main__":
    unittest.main()
