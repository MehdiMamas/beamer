import unittest
from types import SimpleNamespace

from core import audio_system


class FakeRun:
    def __init__(self, answers=None, missing=()):
        self.calls, self.answers, self.missing = [], answers or {}, missing

    def __call__(self, command, **kwargs):
        self.calls.append(list(command))
        if command[0] in self.missing:
            raise FileNotFoundError(command[0])
        out = self.answers.get(" ".join(command), "")
        return SimpleNamespace(returncode=0, stdout=out)


class FakeEndpoint:
    def __init__(self):
        self.level, self.muted = 0.4, False

    def GetMasterVolumeLevelScalar(self):
        return self.level

    def GetMute(self):
        return self.muted

    def SetMasterVolumeLevelScalar(self, level, ctx):
        self.level = level

    def SetMute(self, muted, ctx):
        self.muted = bool(muted)


class AlertTests(unittest.TestCase):
    def test_windows_uses_winsound(self):
        beeps = []
        fake = SimpleNamespace(MessageBeep=beeps.append)
        self.assertTrue(audio_system.play_alert("error", "win32", winsound=fake))
        self.assertEqual(beeps, [0x10])

    def test_unknown_kind_refused(self):
        self.assertFalse(audio_system.play_alert("rm -rf", "linux", FakeRun()))

    def test_linux_falls_back_to_paplay(self):
        run = FakeRun(missing=("canberra-gtk-play",))
        self.assertTrue(audio_system.play_alert("notify", "linux", run))
        self.assertEqual(run.calls[-1][0], "paplay")

    def test_linux_nothing_installed(self):
        self.assertFalse(audio_system.play_alert("default", "linux", FakeRun(missing=("canberra-gtk-play", "paplay"))))

    def test_mac_afplay(self):
        run = FakeRun()
        self.assertTrue(audio_system.play_alert("default", "darwin", run))
        self.assertEqual(run.calls[0][0], "afplay")


class VolumeTests(unittest.TestCase):
    def test_linux_get_and_set(self):
        run = FakeRun({"pactl get-sink-volume @DEFAULT_SINK@": "Volume: front-left: 32768 /  50% / -18 dB",
                       "pactl get-sink-mute @DEFAULT_SINK@": "Mute: yes"})
        control = audio_system.VolumeControl("linux", run)
        self.assertEqual(control.get(), (0.5, True))
        self.assertTrue(control.set(0.25, False))
        self.assertIn(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "25%"], run.calls)
        self.assertIn(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"], run.calls)

    def test_mac_get(self):
        run = FakeRun()
        run.answers = {}
        control = audio_system.VolumeControl("darwin", lambda c, **k: SimpleNamespace(returncode=0, stdout="30,false\n"))
        self.assertEqual(control.get(), (0.3, False))

    def test_windows_through_fake_endpoint(self):
        endpoint = FakeEndpoint()
        control = audio_system.VolumeControl("win32", windows_loader=lambda: endpoint)
        self.assertTrue(control.set(0.8, True))
        self.assertEqual(control.get(), (0.8, True))

    def test_windows_without_pycaw_is_unavailable(self):
        def missing():
            raise ImportError("pycaw")
        control = audio_system.VolumeControl("win32", windows_loader=missing)
        self.assertIsNone(control.get())
        self.assertFalse(control.set(0.5, False))

    def test_bad_level_refused(self):
        control = audio_system.VolumeControl("linux", FakeRun())
        for bad in (-0.1, 1.5, True, "1", float("nan")):
            self.assertFalse(control.set(bad, False))

    def test_missing_tool_gives_none(self):
        self.assertIsNone(audio_system.VolumeControl("linux", FakeRun(missing=("pactl",))).get())


class PollerTests(unittest.TestCase):
    def setUp(self):
        self.endpoint = FakeEndpoint()
        self.control = audio_system.VolumeControl("win32", windows_loader=lambda: self.endpoint)
        self.sent = []
        self.poller = audio_system.VolumePoller(self.control, lambda l, m: self.sent.append((l, m)))

    def test_first_look_is_baseline_then_change_reported(self):
        self.assertFalse(self.poller.poll())
        self.assertFalse(self.poller.poll())
        self.endpoint.level = 0.7
        self.assertTrue(self.poller.poll())
        self.assertEqual(self.sent, [(0.7, False)])
        self.assertFalse(self.poller.poll())

    def test_mute_change_reported(self):
        self.poller.poll()
        self.endpoint.muted = True
        self.assertTrue(self.poller.poll())

    def test_applied_value_is_not_sent_back(self):
        self.poller.poll()
        self.assertTrue(self.poller.apply(0.9, True))
        self.assertFalse(self.poller.poll())
        self.assertEqual(self.sent, [])

    def test_failing_callback_does_not_stop_polling(self):
        poller = audio_system.VolumePoller(self.control, lambda l, m: 1 / 0)
        poller.poll()
        self.endpoint.level = 0.1
        self.assertTrue(poller.poll())


if __name__ == "__main__":
    unittest.main()
