import base64
import threading
import time
import unittest

from core import audio, audio_io, audio_stream, protocol

ME = bytes([1]) * 16
PEER = bytes([2]) * 16
OTHER = bytes([3]) * 16
FRAME = audio.frame_bytes(2)
MIC_FRAME = audio.frame_bytes(1)
FMT = {"rate": 48000, "format": "s16le", "frame_ms": 20}


def message(kind, **data):
    return {"type": kind, "data": data}


def wait_for(condition, seconds=2.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


class FakeSource:
    def __init__(self, size):
        self.size = size
        self.closed = threading.Event()

    def read(self):
        if self.closed.wait(0.002):
            raise OSError("closed")
        return bytes(self.size)

    def close(self):
        self.closed.set()


class FakeSink:
    def __init__(self):
        self.frames = []
        self.closed = False

    def write(self, pcm):
        self.frames.append(pcm)
        time.sleep(0.002)

    def close(self):
        self.closed = True


class FakeBackend:
    def __init__(self):
        self.sources, self.sinks = [], []

    def open_source(self, kind):
        source = FakeSource(FRAME if kind == "loopback" else MIC_FRAME)
        self.sources.append((kind, source))
        return source

    def open_sink(self, kind):
        sink = FakeSink()
        self.sinks.append((kind, sink))
        return sink

    def open_kinds(self):
        return sorted([k for k, s in self.sources if not s.closed.is_set()]
                      + [k for k, s in self.sinks if not s.closed])


class Rig:
    def __init__(self, mode="follow", host="", sound=True, mic=True, caps=frozenset({"audio"})):
        self.sent = []
        self.backend = FakeBackend()
        self.setting = [mode, host]
        self.controller = audio.AudioController(
            lambda peer, msg: self.sent.append((peer, msg)) or True,
            lambda peer: caps,
            protocol.id_text(ME),
            backend=self.backend,
            want_sound=lambda: sound,
            want_mic=lambda: mic,
            config=lambda: tuple(self.setting),
        )

    def kinds(self, peer=PEER):
        return [(m["type"], m["data"].get("dir")) for p, m in self.sent
                if p == peer and m["type"] != protocol.MSG_AUDIO_DATA]

    def stream_kinds(self, peer=PEER):
        return [k for k in self.kinds(peer) if k[0] != protocol.MSG_AUDIO_CONFIG]

    def close(self):
        self.controller.close()


def rig(test, **kw):
    r = Rig(**kw)
    test.addCleanup(r.close)
    return r


class ProtocolTests(unittest.TestCase):
    def test_every_message_round_trips_through_read_audio(self):
        pcm = bytes(FRAME)
        for msg, kind in [
            (protocol.audio_want_msg("out"), protocol.MSG_AUDIO_WANT),
            (protocol.audio_start_msg("mic", 48000, 1), protocol.MSG_AUDIO_START),
            (protocol.audio_data_msg("out", 7, pcm), protocol.MSG_AUDIO_DATA),
            (protocol.audio_stop_msg("mic", "failed"), protocol.MSG_AUDIO_STOP),
            (protocol.audio_config_msg("fixed", ME), protocol.MSG_AUDIO_CONFIG),
            (protocol.audio_config_msg("follow"), protocol.MSG_AUDIO_CONFIG),
        ]:
            got = protocol.read_audio(msg)
            self.assertIsNotNone(got, msg)
            self.assertEqual(got[0], kind)
        self.assertEqual(protocol.read_audio(protocol.audio_data_msg("out", 7, pcm))[1]["pcm"], pcm)
        self.assertEqual(protocol.read_audio(protocol.audio_config_msg("fixed", ME))[1],
                         {"mode": "fixed", "host": ME})

    def test_formats_outside_the_allow_lists_are_refused(self):
        for bad in [
            protocol.audio_want_msg("speaker"), protocol.audio_want_msg("out", rate=44100),
            protocol.audio_want_msg("out", channels=6), protocol.audio_want_msg("out", format="f32le"),
            protocol.audio_want_msg("out", frame_ms=5), protocol.audio_start_msg("out", channels=0),
        ]:
            self.assertIsNone(protocol.read_audio(bad), bad)

    def test_data_limits(self):
        enc = lambda raw: base64.b64encode(raw).decode()
        self.assertIsNotNone(protocol.read_audio(message(protocol.MSG_AUDIO_DATA, dir="out", seq=0, pcm=enc(bytes(8)))))
        for data in [
            dict(dir="out", seq=0, pcm=enc(bytes(protocol.AUDIO_MAX_FRAME_BYTES + 4))),
            dict(dir="out", seq=0, pcm=enc(bytes(6))),
            dict(dir="out", seq=0, pcm=""),
            dict(dir="out", seq=-1, pcm=enc(bytes(8))),
            dict(dir="out", seq=protocol.AUDIO_MAX_SEQ + 1, pcm=enc(bytes(8))),
            dict(dir="side", seq=0, pcm=enc(bytes(8))),
            dict(dir="out", seq=0, pcm="not base64!"),
            dict(dir="out", seq=0, pcm=5),
        ]:
            self.assertIsNone(protocol.read_audio(message(protocol.MSG_AUDIO_DATA, **data)), data)

    def test_config_needs_a_real_host_when_fixed(self):
        for data in [dict(mode="fixed"), dict(mode="fixed", host="zz"),
                     dict(mode="fixed", host=protocol.id_text(bytes(16))), dict(mode="loud"), dict()]:
            self.assertIsNone(protocol.read_audio(message(protocol.MSG_AUDIO_CONFIG, **data)), data)

    def test_other_messages_are_not_audio(self):
        self.assertIsNone(protocol.read_audio({"type": "key", "data": {}}))
        self.assertIn(protocol.MSG_AUDIO_DATA, protocol.AUDIO_MESSAGES)


class JitterBufferTests(unittest.TestCase):
    def frame(self, n):
        return bytes([n]) * FRAME

    def buffer(self, **kw):
        return audio_stream.JitterBuffer(FRAME, 20, **kw)

    def test_waits_for_the_target_then_plays_in_order(self):
        buf = self.buffer(target_ms=60)
        for n in (2, 0, 1):
            buf.push(n, self.frame(n))
        self.assertEqual([buf.pull()[0] for _ in range(3)], [0, 1, 2])

    def test_silence_before_the_target_and_after_the_end(self):
        buf = self.buffer(target_ms=60)
        buf.push(0, self.frame(1))
        self.assertEqual(buf.pull(), bytes(FRAME))
        for n in range(1, 3):
            buf.push(n, self.frame(1))
        for _ in range(3):
            buf.pull()
        self.assertEqual(buf.pull(), bytes(FRAME))
        self.assertEqual(buf.underruns, 1)

    def test_late_repeated_and_wrong_sized_frames_are_refused(self):
        buf = self.buffer(target_ms=20)
        self.assertTrue(buf.push(5, self.frame(5)))
        self.assertFalse(buf.push(5, self.frame(5)))
        self.assertFalse(buf.push(6, b"short"))
        buf.pull()
        self.assertFalse(buf.push(4, self.frame(4)))
        self.assertEqual(buf.late, 1)

    def test_a_missing_frame_is_skipped_not_waited_for(self):
        buf = self.buffer(target_ms=20)
        buf.push(0, self.frame(1))
        buf.push(2, self.frame(3))
        self.assertEqual(buf.pull()[0], 1)
        self.assertEqual(buf.pull(), bytes(FRAME))
        self.assertEqual(buf.pull()[0], 3)
        self.assertEqual(buf.lost, 1)

    def test_it_never_holds_more_than_its_limit(self):
        buf = self.buffer(target_ms=20, max_ms=100)
        for n in range(50):
            buf.push(n, self.frame(1))
        self.assertLessEqual(len(buf), buf.limit)


class ElectionTests(unittest.TestCase):
    FOLLOW = ("follow", None)

    def test_neither_side_fixed_follows_the_input(self):
        self.assertEqual(audio.elect(self.FOLLOW, None, "a", "b"), ("follow", None))
        self.assertEqual(audio.elect(self.FOLLOW, self.FOLLOW, "a", "b"), ("follow", None))

    def test_a_fixed_side_wins_over_a_following_one(self):
        self.assertEqual(audio.elect(("fixed", "a"), self.FOLLOW, "a", "b")[0], "host")
        self.assertEqual(audio.elect(("fixed", "b"), self.FOLLOW, "a", "b")[0], "guest")
        role, note = audio.elect(self.FOLLOW, ("fixed", "a"), "a", "b")
        self.assertEqual(role, "host")
        self.assertTrue(note)
        self.assertEqual(audio.elect(self.FOLLOW, ("fixed", "b"), "a", "b")[0], "guest")

    def test_two_different_fixed_hosts_are_settled_the_same_way_on_both_sides(self):
        for x, y in [("a", "b"), ("b", "a")]:
            one = audio.elect(("fixed", x), ("fixed", y), "a", "b")
            two = audio.elect(("fixed", y), ("fixed", x), "b", "a")
            self.assertEqual({one[0], two[0]}, {"host", "guest"})

    def test_a_third_computer_as_host_means_no_audio_on_this_link(self):
        role, note = audio.elect(("fixed", "c"), self.FOLLOW, "a", "b")
        self.assertEqual(role, "none")
        self.assertTrue(note)

    def test_the_same_fixed_host_on_both_sides(self):
        self.assertEqual(audio.elect(("fixed", "a"), ("fixed", "a"), "a", "b")[0], "host")
        self.assertEqual(audio.elect(("fixed", "b"), ("fixed", "b"), "a", "b")[0], "guest")


class FollowModeTests(unittest.TestCase):
    def test_sound_streams_only_while_the_input_is_on_the_peer(self):
        r = rig(self)
        c = r.controller
        c.link_up(PEER)
        self.assertEqual(r.kinds(), [(protocol.MSG_AUDIO_CONFIG, None)])
        c.input_moved(PEER)
        self.assertIn((protocol.MSG_AUDIO_WANT, "out"), r.kinds())
        self.assertIn((protocol.MSG_AUDIO_START, "mic"), r.kinds())
        self.assertTrue(wait_for(lambda: r.backend.open_kinds() == ["mic"]))
        c.input_moved(None)
        self.assertIn((protocol.MSG_AUDIO_STOP, "out"), r.kinds())
        self.assertIn((protocol.MSG_AUDIO_STOP, "mic"), r.kinds())
        self.assertEqual(r.backend.open_kinds(), [])

    def test_the_config_goes_out_before_any_stream_message(self):
        r = rig(self)
        r.controller.input_moved(PEER)
        self.assertEqual(r.kinds()[0], (protocol.MSG_AUDIO_CONFIG, None))

    def test_the_guest_sends_its_sound_when_asked_and_stops_when_told(self):
        r = rig(self)
        c = r.controller
        c.owner_changed(PEER)
        c.receive(PEER, protocol.MSG_AUDIO_WANT, dict(FMT, dir="out", channels=2))
        self.assertIn((protocol.MSG_AUDIO_START, "out"), r.kinds())
        self.assertTrue(wait_for(lambda: any(m["type"] == protocol.MSG_AUDIO_DATA for p, m in r.sent)))
        c.receive(PEER, protocol.MSG_AUDIO_STOP, {"dir": "out"})
        self.assertEqual(r.backend.open_kinds(), [])

    def test_the_guest_plays_the_hosts_microphone_into_the_virtual_input(self):
        r = rig(self)
        c = r.controller
        c.owner_changed(PEER)
        c.receive(PEER, protocol.MSG_AUDIO_START, dict(FMT, dir="mic", channels=1))
        self.assertEqual(r.backend.open_kinds(), ["virtual_mic"])
        sink = r.backend.sinks[0][1]
        for n in range(10):
            c.receive(PEER, protocol.MSG_AUDIO_DATA, {"dir": "mic", "seq": n, "pcm": bytes([1]) * MIC_FRAME})
        self.assertTrue(wait_for(lambda: any(f[0] == 1 for f in sink.frames)))

    def test_a_peer_that_is_not_in_a_role_is_ignored(self):
        r = rig(self)
        r.controller.link_up(PEER)
        r.controller.receive(PEER, protocol.MSG_AUDIO_WANT, dict(FMT, dir="out", channels=2))
        r.controller.receive(PEER, protocol.MSG_AUDIO_START, dict(FMT, dir="mic", channels=1))
        self.assertEqual(r.backend.sources + r.backend.sinks, [])

    def test_nothing_is_sent_to_a_peer_without_the_capability(self):
        r = rig(self, caps=frozenset())
        r.controller.link_up(PEER)
        r.controller.input_moved(PEER)
        self.assertEqual(r.sent, [])

    def test_a_dropped_link_ends_the_streams_without_sending(self):
        r = rig(self)
        r.controller.input_moved(PEER)
        r.sent.clear()
        r.controller.link_down(PEER)
        self.assertEqual(r.stream_kinds(), [])
        self.assertEqual(r.backend.open_kinds(), [])

    def test_the_switches_decide_what_streams(self):
        r = rig(self, sound=False, mic=False)
        r.controller.input_moved(PEER)
        self.assertEqual(r.stream_kinds(), [])


class FixedModeTests(unittest.TestCase):
    def test_a_fixed_host_serves_every_peer_whatever_owns_the_input(self):
        r = rig(self, mode="fixed", host="")
        c = r.controller
        c.link_up(PEER)
        c.link_up(OTHER)
        for peer in (PEER, OTHER):
            self.assertIn((protocol.MSG_AUDIO_WANT, "out"), r.kinds(peer))
            self.assertIn((protocol.MSG_AUDIO_START, "mic"), r.kinds(peer))
        c.input_moved(PEER)
        c.input_moved(None)
        c.owner_changed(OTHER)
        c.owner_changed(None)
        self.assertEqual([m for p, m in r.sent if m["type"] == protocol.MSG_AUDIO_STOP], [])
        self.assertEqual(r.backend.open_kinds(), ["mic", "mic"])

    def test_the_config_names_this_machine_as_the_host(self):
        r = rig(self, mode="fixed", host="")
        r.controller.link_up(PEER)
        cfg = [m for p, m in r.sent if m["type"] == protocol.MSG_AUDIO_CONFIG][0]
        self.assertEqual(protocol.read_audio(cfg)[1], {"mode": "fixed", "host": ME})

    def test_a_peer_chosen_as_host_makes_this_machine_a_guest_whatever_owns_the_input(self):
        r = rig(self, mode="fixed", host=protocol.id_text(PEER))
        c = r.controller
        c.link_up(PEER)
        c.input_moved(PEER)
        c.input_moved(None)
        self.assertEqual(r.stream_kinds(), [])
        c.receive(PEER, protocol.MSG_AUDIO_WANT, dict(FMT, dir="out", channels=2))
        self.assertEqual(r.backend.open_kinds(), ["loopback"])
        c.input_moved(PEER)
        self.assertEqual(r.backend.open_kinds(), ["loopback"])

    def test_a_fixed_peer_that_names_this_machine_is_served_even_when_this_one_follows(self):
        r = rig(self, mode="follow")
        c = r.controller
        c.link_up(PEER)
        self.assertNotIn((protocol.MSG_AUDIO_WANT, "out"), r.kinds())
        c.receive(PEER, protocol.MSG_AUDIO_CONFIG, {"mode": "fixed", "host": ME})
        self.assertIn((protocol.MSG_AUDIO_WANT, "out"), r.kinds())

    def test_a_peer_that_fixes_itself_as_host_turns_this_following_machine_into_a_guest(self):
        r = rig(self, mode="follow")
        c = r.controller
        c.input_moved(PEER)
        self.assertIn((protocol.MSG_AUDIO_WANT, "out"), r.kinds())
        r.sent.clear()
        c.receive(PEER, protocol.MSG_AUDIO_CONFIG, {"mode": "fixed", "host": PEER})
        self.assertIn((protocol.MSG_AUDIO_STOP, "out"), r.kinds())
        c.receive(PEER, protocol.MSG_AUDIO_WANT, dict(FMT, dir="out", channels=2))
        self.assertEqual(r.backend.open_kinds(), ["loopback"])

    def test_changing_the_setting_re_announces_and_switches_the_streams(self):
        r = rig(self, mode="follow")
        c = r.controller
        c.input_moved(PEER)
        r.sent.clear()
        r.setting[:] = ["fixed", protocol.id_text(PEER)]
        c.refresh()
        kinds = r.kinds()
        self.assertEqual(kinds[0], (protocol.MSG_AUDIO_CONFIG, None))
        self.assertIn((protocol.MSG_AUDIO_STOP, "out"), kinds)
        self.assertEqual(r.backend.open_kinds(), [])

    def test_a_third_computer_as_host_serves_nobody_on_this_link(self):
        r = rig(self, mode="fixed", host=protocol.id_text(OTHER))
        r.controller.link_up(PEER)
        r.controller.input_moved(PEER)
        self.assertEqual(r.stream_kinds(), [])


class OptionalDependencyTests(unittest.TestCase):
    def test_the_linux_virtual_microphone_loads_and_unloads_its_modules(self):
        calls = []

        class Done:
            returncode, stderr = 0, ""

            def __init__(self, out):
                self.stdout = out

        def run(argv, **kw):
            calls.append(argv)
            return Done(str(40 + len(calls)))

        cable = audio_io.LinuxVirtualMic(run)
        self.assertEqual(cable.load(), audio_io.LINUX_SINK)
        cable.unload()
        self.assertEqual([c[1] for c in calls], ["load-module", "load-module", "unload-module", "unload-module"])
        self.assertEqual(calls[2][2], "42")

    def test_a_failing_pactl_raises_and_unloads_what_was_loaded(self):
        calls = []

        class Done:
            def __init__(self, code):
                self.returncode, self.stdout, self.stderr = code, "7", "boom"

        def run(argv, **kw):
            calls.append(argv)
            return Done(1 if len(calls) == 2 else 0)

        with self.assertRaises(audio_io.AudioUnavailable):
            audio_io.LinuxVirtualMic(run).load()
        self.assertEqual(calls[-1][1], "unload-module")

    def test_without_the_library_a_device_cannot_be_opened_but_the_import_works(self):
        import importlib.util
        if importlib.util.find_spec("soundcard") is not None:
            self.skipTest("soundcard is installed")
        self.assertFalse(audio_io.available())
        with self.assertRaises(audio_io.AudioUnavailable):
            audio_io.SoundcardBackend("win32").open_source("mic")


if __name__ == "__main__":
    unittest.main()
