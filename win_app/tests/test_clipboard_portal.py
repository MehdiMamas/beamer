"""set_contents on the link's reader thread: it never waits for the RemoteDesktop session. The
desktop's consent dialog can hold the session up for longer than the sender waits for an
acknowledgement, so a clipboard arriving before the session is up is dropped at once, not waited on
(Fedora, 10-10-2026: a 2 s wait stalled input and the Mac ended the link)."""

import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import clipboard_portal
import inject_portal

# Well under core.link.ACK_STALL_SECONDS (1.5), which core.link needs PyNaCl to import.
QUICK_SECONDS = 0.2
STUB_WAIT_SECONDS = 0.6


class StubRemote:
    """Stands in for inject_portal.Remote. `ask`, like the real one, starts the session and waits
    for it (here STUB_WAIT_SECONDS) unless it is ready, then runs the work on a stub session."""

    def __init__(self, ready):
        self.ready = threading.Event()
        if ready:
            self.ready.set()
        self.asked = 0
        self.session = mock.Mock(changes=5)
        self.session.set_selection.return_value = True

    def ask(self, work, default=None, **_kwargs):
        self.asked += 1
        if not self.ready.is_set():
            time.sleep(STUB_WAIT_SECONDS)
            return default
        return work(self.session)


class SetContentsTest(unittest.TestCase):
    def setUp(self):
        clipboard_portal.forget_sync()
        self.saved = inject_portal._remote

    def tearDown(self):
        inject_portal._remote = self.saved

    def timed(self):
        started = time.monotonic()
        result = clipboard_portal.set_contents("hello", None)
        return result, time.monotonic() - started

    def test_a_session_not_started_is_not_waited_on_and_not_started(self):
        stub = StubRemote(ready=False)
        inject_portal._remote = None
        with mock.patch.object(inject_portal, "remote", return_value=stub):
            result, elapsed = self.timed()
        self.assertFalse(result)
        self.assertLess(elapsed, QUICK_SECONDS)
        self.assertEqual(stub.asked, 0)

    def test_a_session_started_but_not_ready_is_not_waited_on(self):
        stub = StubRemote(ready=False)
        inject_portal._remote = stub
        with mock.patch.object(inject_portal, "remote", return_value=stub):
            result, elapsed = self.timed()
        self.assertFalse(result)
        self.assertLess(elapsed, QUICK_SECONDS)
        self.assertEqual(stub.asked, 0)

    def test_a_ready_session_still_takes_the_clipboard(self):
        stub = StubRemote(ready=True)
        inject_portal._remote = stub
        with mock.patch.object(inject_portal, "remote", return_value=stub):
            self.assertTrue(clipboard_portal.set_contents("hello\r\nthere", None))
        stub.session.set_selection.assert_called_once()
        offer = stub.session.set_selection.call_args[0][0]
        self.assertEqual(offer["text/plain"], b"hello\nthere")
        self.assertEqual(clipboard_portal._synced, 5)


class ReadContentsTest(unittest.TestCase):
    def test_strict_read_distinguishes_an_unavailable_session_from_empty(self):
        stub = StubRemote(ready=True)
        stub.session.offered = []
        stub.session.ours = False
        with mock.patch.object(inject_portal, "remote", return_value=stub):
            self.assertEqual(clipboard_portal.read_contents(), (None, None))
        with mock.patch.object(inject_portal, "remote") as remote:
            remote.return_value.ask.side_effect = lambda work, default=None, **kwargs: default
            with self.assertRaises(OSError):
                clipboard_portal.read_contents()
            self.assertEqual(clipboard_portal.get_contents(), (None, None))

    def test_strict_read_distinguishes_a_failed_transfer_from_empty_text(self):
        with mock.patch.object(clipboard_portal, "_offered", return_value=(["text/plain"], False, 5)), \
                mock.patch.object(inject_portal, "remote") as remote:
            remote.return_value.ask.return_value = None
            with self.assertRaises(OSError):
                clipboard_portal.read_contents()
            self.assertEqual(clipboard_portal.get_contents(), (None, None))
        with mock.patch.object(clipboard_portal, "_offered", return_value=(["text/plain"], False, 5)), \
                mock.patch.object(clipboard_portal, "_read", return_value=b""):
            self.assertEqual(clipboard_portal.read_contents(), (None, None))


class FilesTest(unittest.TestCase):
    def setUp(self):
        clipboard_portal.forget_sync()
        self.saved = inject_portal._remote

    def tearDown(self):
        inject_portal._remote = self.saved

    def test_set_files_offers_a_uri_list_and_a_gnome_copy(self):
        stub = StubRemote(ready=True)
        inject_portal._remote = stub
        with mock.patch.object(inject_portal, "remote", return_value=stub):
            self.assertTrue(clipboard_portal.set_files(["/tmp/a b.txt", "/tmp/dir"]))
        offer = stub.session.set_selection.call_args[0][0]
        self.assertEqual(set(offer), {"text/uri-list", "x-special/gnome-copied-files"})
        self.assertEqual(offer["text/uri-list"].split(b"\r\n")[0].split(b"a%20b.txt")[1:], [b""])
        self.assertTrue(offer["text/uri-list"].startswith(b"file://"))
        self.assertTrue(offer["x-special/gnome-copied-files"].startswith(b"copy\nfile://"))
        self.assertEqual(clipboard_portal._synced, 5)

    def test_set_files_without_a_session_or_paths_is_false_and_never_waits(self):
        stub = StubRemote(ready=False)
        inject_portal._remote = stub
        with mock.patch.object(inject_portal, "remote", return_value=stub):
            self.assertFalse(clipboard_portal.set_files(["/tmp/a"]))
        self.assertEqual(stub.asked, 0)
        ready = StubRemote(ready=True)
        inject_portal._remote = ready
        with mock.patch.object(inject_portal, "remote", return_value=ready):
            self.assertFalse(clipboard_portal.set_files([]))
        ready.session.set_selection.assert_not_called()

    def test_read_files_keeps_existing_local_files_only(self):
        # POSIX-style uris, so the test means the same on every OS that runs it
        uris = (b"# comment" + b"\r\n" + b"file:///tmp/x%20y.txt" + b"\r\n" + b"file:///tmp/gone" + b"\r\n"
                + b"https://example.com/a" + b"\r\n" + b"file://other/host/z")
        with mock.patch.object(clipboard_portal, "_offered", return_value=(["text/uri-list"], False, 5)), \
                mock.patch.object(clipboard_portal, "_read", return_value=uris), \
                mock.patch.object(clipboard_portal.os.path, "exists", side_effect=lambda p: p == "/tmp/x y.txt"):
            self.assertEqual(clipboard_portal.read_files(), ["/tmp/x y.txt"])

    def test_read_files_is_empty_for_our_own_copy_text_and_failures(self):
        with mock.patch.object(clipboard_portal, "_offered", return_value=(["text/uri-list"], True, 5)):
            self.assertEqual(clipboard_portal.read_files(), [])
        with mock.patch.object(clipboard_portal, "_offered", return_value=(["text/plain"], False, 5)):
            self.assertEqual(clipboard_portal.read_files(), [])
        with mock.patch.object(clipboard_portal, "_offered", return_value=None):
            self.assertEqual(clipboard_portal.read_files(), [])
        with mock.patch.object(clipboard_portal, "_offered", return_value=(["text/uri-list"], False, 5)), \
                mock.patch.object(clipboard_portal, "_read", side_effect=OSError("no")):
            self.assertEqual(clipboard_portal.read_files(), [])


if __name__ == "__main__":
    unittest.main()
