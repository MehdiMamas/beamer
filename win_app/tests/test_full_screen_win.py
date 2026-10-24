import ctypes
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import desktop_win


class FullScreenAppTest(unittest.TestCase):
    def test_beamer_is_not_a_full_screen_app_to_itself(self):
        state = ctypes.c_int(3)

        def query_user_notification_state(value):
            value._obj.value = state.value
            return 0

        process = ctypes.c_ulong(os.getpid())
        foreground = SimpleNamespace(
            GetForegroundWindow=lambda: 42,
            GetWindowThreadProcessId=lambda _hwnd, value: setattr(value._obj, "value", process.value),
        )
        shell = SimpleNamespace(SHQueryUserNotificationState=query_user_notification_state)
        with mock.patch.object(desktop_win, "_require"), \
                mock.patch.object(desktop_win, "user32", foreground), \
                mock.patch.object(desktop_win.ctypes, "windll", SimpleNamespace(shell32=shell), create=True), \
                mock.patch.object(desktop_win, "_foreground_covers_its_monitor") as covers, \
                mock.patch.object(desktop_win, "_foreground_app_name") as name:
            self.assertIsNone(desktop_win.full_screen_app())
        covers.assert_not_called()
        name.assert_not_called()

    def _foreground_app(self, state_value, covers_monitor):
        state = ctypes.c_int(state_value)

        def query_user_notification_state(value):
            value._obj.value = state.value
            return 0

        other_process = ctypes.c_ulong(os.getpid() + 1)
        foreground = SimpleNamespace(
            GetForegroundWindow=lambda: 42,
            GetWindowThreadProcessId=lambda _hwnd, value: setattr(value._obj, "value", other_process.value),
        )
        shell = SimpleNamespace(SHQueryUserNotificationState=query_user_notification_state)
        with mock.patch.object(desktop_win, "_require"), \
                mock.patch.object(desktop_win, "user32", foreground), \
                mock.patch.object(desktop_win.ctypes, "windll", SimpleNamespace(shell32=shell), create=True), \
                mock.patch.object(desktop_win, "_foreground_covers_its_monitor", return_value=covers_monitor), \
                mock.patch.object(desktop_win, "_foreground_app_name", return_value="Rocket League"):
            return desktop_win.full_screen_app()

    def test_exclusive_full_screen_counts_without_a_monitor_check(self):
        self.assertEqual(self._foreground_app(3, covers_monitor=False), "Rocket League")
        self.assertEqual(self._foreground_app(4, covers_monitor=False), "Rocket League")

    def test_borderless_window_covering_its_monitor_counts_in_any_state(self):
        for state in (0, 1, 2, 5, 6):
            self.assertEqual(self._foreground_app(state, covers_monitor=True), "Rocket League", state)

    def test_overlay_raising_busy_without_covering_the_monitor_does_not_count(self):
        self.assertIsNone(self._foreground_app(2, covers_monitor=False))

    def test_maximised_window_not_covering_the_monitor_does_not_count(self):
        for state in (0, 1, 2):
            self.assertIsNone(self._foreground_app(state, covers_monitor=False), state)


if __name__ == "__main__":
    unittest.main()
