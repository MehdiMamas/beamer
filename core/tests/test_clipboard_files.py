"""Files and folders on the clipboard: the pure encode/parse/write module, the message, and the
receiver's use of a platform's `set_files` and `read_files`."""

import base64
import os
import shutil
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from core import clipboard_files as cf
from core import protocol, receiver
from core.tests.responder_harness import FakeClipboard, FakeInjector


def entry(path, data=b"x"):
    return {"path": path, "data": base64.b64encode(data).decode("ascii")}


class Scratch(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def make(self, *parts, data=b"x"):
        path = os.path.join(self.dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(data)
        return path


class ValidPaths(unittest.TestCase):
    def test_plain_paths_pass(self):
        for path in ("a.txt", "dir/a.txt", "dir/sub/b c.bin", "\u00e9/\u00fc.txt", "CONSOLE.txt", "a.b.c"):
            with self.subTest(path=path):
                self.assertTrue(cf.valid_path(path))
        self.assertTrue(cf.valid_path("dir/", True))

    def test_unsafe_paths_are_refused(self):
        bad = ["", "/abs", "C:/x", "C:x", "a\\b", "..", "../x", "a/../b", "a/./b", ".", "a//b", "a/", "a\x00b", "a\nb",
               "a\x7fb", "CON", "con.txt", "a/NUL", "aux.tar.gz", "COM1", "lpt9.x", "a<b", "a>b", 'a"b', "a|b", "a?b",
               "a*b", "a:b", "trail.", "trail ", "d/trail./x", "x" * 256, "a/" * 400 + "b", "\ud800", None, 5]
        for path in bad:
            with self.subTest(path=path):
                self.assertFalse(cf.valid_path(path))
        self.assertFalse(cf.valid_path("dir", True))
        self.assertFalse(cf.valid_path("dir/", False))
        self.assertFalse(cf.valid_path("../", True))

    def test_depth_limit(self):
        self.assertTrue(cf.valid_path("/".join("d" * 32)))
        self.assertFalse(cf.valid_path("/".join("d" * 33)))

    def test_path_byte_limit(self):
        self.assertTrue(cf.valid_path("/".join(["a" * 200] * 5)))
        self.assertFalse(cf.valid_path("/".join(["a" * 200] * 6)))


class ParseEntries(unittest.TestCase):
    def test_round_trip_in_order(self):
        wire = [{"path": "top/", "dir": True}, entry("top/a.txt", b"hello"), {"path": "top/empty/", "dir": True}, entry("b.bin", bytes(range(256)))]
        self.assertEqual(cf.parse_entries(wire), [("top/", None), ("top/a.txt", b"hello"), ("top/empty/", None), ("b.bin", bytes(range(256)))])

    def test_refusals_drop_the_whole_list(self):
        good = entry("ok.txt")
        cases = {
            "not a list": "x", "empty": [], "not dicts": [5], "dir false": [{"path": "d/", "dir": False}],
            "traversal": [good, entry("../x")], "dup": [good, entry("ok.txt")], "case dup": [good, entry("OK.TXT")],
            "dir and file": [entry("a"), {"path": "a/", "dir": True}], "file as parent": [entry("a"), entry("a/b")],
            "no data": [{"path": "a"}], "data not str": [{"path": "a", "data": 5}],
            "bad base64": [{"path": "a", "data": "!!!!"}], "padding": [{"path": "a", "data": "QQ"}],
            "urlsafe": [{"path": "a", "data": "_-8="}], "newline": [{"path": "a", "data": "QUJD\nREVG"}],
        }
        for name, wire in cases.items():
            with self.subTest(name):
                self.assertIsNone(cf.parse_entries(wire))

    def test_entry_limit(self):
        self.assertIsNotNone(cf.parse_entries([entry("f%d" % n, b"") for n in range(cf.MAX_ENTRIES)]))
        self.assertIsNone(cf.parse_entries([entry("f%d" % n, b"") for n in range(cf.MAX_ENTRIES + 1)]))

    def test_byte_limit_counts_every_file(self):
        half = b"\0" * (cf.MAX_BYTES // 2)
        self.assertIsNotNone(cf.parse_entries([entry("a", half), entry("b", half)]))
        self.assertIsNone(cf.parse_entries([entry("a", half), entry("b", half), entry("c", b"1")]))
        self.assertIsNone(cf.parse_entries([entry("a", b"\0" * (cf.MAX_BYTES + 1))]))


class EncodePaths(Scratch):
    def test_a_nested_folder_with_an_empty_one_round_trips(self):
        self.make("top", "a.txt", data=b"aa")
        self.make("top", "sub", "b.bin", data=b"\x00\x01")
        os.makedirs(os.path.join(self.dir, "top", "empty"))
        loose = self.make("loose.txt", data=b"L")
        wire = cf.encode_paths([os.path.join(self.dir, "top"), loose])
        parsed = cf.parse_entries(wire)
        self.assertEqual(sorted(parsed), sorted([("top/", None), ("top/a.txt", b"aa"), ("top/empty/", None), ("top/sub/", None),
                                                 ("top/sub/b.bin", b"\x00\x01"), ("loose.txt", b"L")]))
        tops = cf.write_transfer(parsed, root=os.path.join(self.dir, "out"))
        self.assertEqual(sorted(os.path.basename(t) for t in tops), ["loose.txt", "top"])
        folder = os.path.dirname(tops[0])
        with open(os.path.join(folder, "top", "sub", "b.bin"), "rb") as handle:
            self.assertEqual(handle.read(), b"\x00\x01")
        self.assertTrue(os.path.isdir(os.path.join(folder, "top", "empty")))

    def test_symlinks_are_skipped(self):
        real = self.make("real.txt")
        link = os.path.join(self.dir, "link.txt")
        try:
            os.symlink(real, link)
        except (OSError, NotImplementedError):
            self.skipTest("no symlinks here")
        self.assertIsNone(cf.encode_paths([link]))
        self.assertEqual([e["path"] for e in cf.encode_paths([link, real])], ["real.txt"])

    def test_over_the_byte_limit_sends_nothing(self):
        big = self.make("big.bin", data=b"\0" * (cf.MAX_BYTES + 1))
        self.assertIsNone(cf.encode_paths([big]))
        exact = self.make("exact.bin", data=b"\0" * cf.MAX_BYTES)
        self.assertEqual(len(cf.encode_paths([exact])), 1)

    def test_over_the_entry_limit_sends_nothing(self):
        for n in range(cf.MAX_ENTRIES + 1):
            self.make("many", "f%d" % n, data=b"")
        self.assertIsNone(cf.encode_paths([os.path.join(self.dir, "many")]))

    def test_nothing_and_missing(self):
        self.assertIsNone(cf.encode_paths([]))
        self.assertIsNone(cf.encode_paths([os.path.join(self.dir, "gone")]))


class WriteTransfer(Scratch):
    def test_keeps_only_the_last_five_transfers(self):
        root = os.path.join(self.dir, "cache")
        folders = []
        for n in range(8):
            tops = cf.write_transfer([("f%d.txt" % n, b"x")], root=root)
            folders.append(os.path.dirname(tops[0]))
        left = sorted(os.listdir(root))
        self.assertEqual(len(left), 5)
        self.assertEqual([os.path.join(root, name) for name in left], sorted(folders)[-5:])

    def test_a_path_leaving_the_folder_is_refused(self):
        root = os.path.join(self.dir, "cache")
        with self.assertRaises(OSError):
            cf.write_transfer([("../escape.txt", b"x")], root=root)
        self.assertFalse(os.path.exists(os.path.join(root, "escape.txt")))

    def test_cache_roots(self):
        self.assertEqual(cf.cache_root("darwin", {"HOME": "/Users/m"}), os.path.join("/Users/m", "Library", "Caches", "Beamer", "Clipboard"))
        self.assertEqual(cf.cache_root("win32", {"LOCALAPPDATA": "L"}), os.path.join("L", "Beamer", "Clipboard"))
        self.assertEqual(cf.cache_root("linux", {"XDG_CACHE_HOME": "/c", "HOME": "/h"}), os.path.join("/c", "beamer", "clipboard"))
        self.assertEqual(cf.cache_root("linux", {"HOME": "/h"}), os.path.join("/h", ".cache", "beamer", "clipboard"))


class Message(unittest.TestCase):
    def test_files_ride_the_message_and_come_back_decoded(self):
        wire = [entry("a.txt", b"hi"), {"path": "d/", "dir": True}]
        read = protocol.read_clipboard(protocol.clipboard_msg("t", None, wire))
        self.assertEqual(read, {"text": "t", "image": None, "files": [("a.txt", b"hi"), ("d/", None)]})

    def test_no_files_leaves_the_read_unchanged(self):
        self.assertEqual(protocol.read_clipboard(protocol.clipboard_msg("t")), {"text": "t", "image": None})

    def test_bad_files_drop_only_the_files(self):
        for wire in ([entry("../x")], [entry("a")] * 2, [entry("f%d" % n, b"") for n in range(1025)],
                     [entry("big", b"\0" * (cf.MAX_BYTES + 1))], "x"):
            with self.subTest(wire=str(wire)[:30]):
                message = {"type": "clipboard", "data": {"text": "keep", "files": wire}}
                self.assertEqual(protocol.read_clipboard(message), {"text": "keep", "image": None})

    def test_constants_match(self):
        self.assertEqual(protocol.CLIPBOARD_FILES_MAX_BYTES, 8 * 1024 * 1024)
        self.assertEqual(protocol.CLIPBOARD_FILES_MAX_ENTRIES, 1024)


class FileClipboard(FakeClipboard):
    def __init__(self, text="here"):
        super().__init__(text)
        self.files = None
        self.set_files_calls = []
        self.fail_files = False

    def set_files(self, paths):
        if self.fail_files:
            return False
        self.set_files_calls.append(list(paths))
        self.files = list(paths)
        self.stamp += 1
        return True

    def read_files(self):
        return self.files


class ReceiverFiles(Scratch):
    def setUp(self):
        super().setUp()
        patch = mock.patch.object(cf, "cache_root", return_value=os.path.join(self.dir, "cache"))
        patch.start()
        self.addCleanup(patch.stop)
        self.clipboard = FileClipboard()
        self.sent = []

    def build(self, caps):
        link = SimpleNamespace(name="Peer", closed=threading.Event(), caps=set(caps), own_caps=set(caps), held={}, post=self.sent.append)
        machine = receiver.LinkResponder(None, lambda: {}, status_callback=lambda *_: None, clipboard=self.clipboard, injector=FakeInjector())
        machine._owner = link
        machine._visit = 1
        machine._clipboard_mark = self.clipboard.stamp
        return machine, link

    def receive(self, machine, link, message):
        machine._on_clipboard(link, message)
        deadline = time.monotonic() + 5
        while machine._clipboard_running and time.monotonic() < deadline:
            time.sleep(0.01)

    def let_go(self, machine, link):
        machine._clipboard_running = True
        with machine._lock:
            after = machine._end_locked(link, let_go=True)
        machine._run(after)
        machine._clipboard_worker()

    def test_received_files_are_written_and_offered_as_files(self):
        machine, link = self.build({"clipboard", "clipboard_files"})
        wire = [{"path": "d/", "dir": True}, entry("d/a.txt", b"hello"), entry("b.txt", b"b")]
        self.receive(machine, link, protocol.clipboard_msg(None, None, wire))
        self.assertEqual(len(self.clipboard.set_files_calls), 1)
        tops = self.clipboard.set_files_calls[0]
        self.assertEqual(sorted(os.path.basename(t) for t in tops), ["b.txt", "d"])
        with open(os.path.join(os.path.dirname(tops[0]), "d", "a.txt"), "rb") as handle:
            self.assertEqual(handle.read(), b"hello")
        self.assertEqual(self.clipboard.set_calls, [])

    def test_text_goes_when_the_platform_cannot_set_files(self):
        machine, link = self.build({"clipboard", "clipboard_files"})
        self.clipboard.fail_files = True
        self.receive(machine, link, protocol.clipboard_msg("words", None, [entry("a")]))
        self.assertEqual(self.clipboard.set_calls, [("words", None)])

    def test_a_platform_without_set_files_gets_the_text(self):
        machine, link = self.build({"clipboard", "clipboard_files"})
        plain = FakeClipboard("here")
        machine._clipboard = plain
        self.receive(machine, link, protocol.clipboard_msg("words", None, [entry("a")]))
        self.assertEqual(plain.set_calls, [("words", None)])

    def test_files_are_ignored_without_our_capability(self):
        machine, link = self.build({"clipboard"})
        self.receive(machine, link, protocol.clipboard_msg("words", None, [entry("a")]))
        self.assertEqual(self.clipboard.set_files_calls, [])
        self.assertEqual(self.clipboard.set_calls, [("words", None)])

    def test_a_file_list_set_from_a_message_is_not_sent_back(self):
        machine, link = self.build({"clipboard", "clipboard_files"})
        self.receive(machine, link, protocol.clipboard_msg(None, None, [entry("a.txt")]))
        self.let_go(machine, link)
        self.assertEqual(self.sent, [])

    def test_a_local_file_copy_goes_back_with_the_files_and_without_the_image(self):
        machine, link = self.build({"clipboard", "clipboard_files", "clipboard_image"})
        self.clipboard.files = [self.make("note.txt", data=b"abc")]
        self.clipboard.image = b"png"
        self.clipboard.copy("here")
        with mock.patch.object(protocol, "png_fits", return_value=True):
            self.let_go(machine, link)
        self.assertEqual(len(self.sent), 1)
        data = self.sent[0]["data"]
        self.assertEqual([e["path"] for e in data["files"]], ["note.txt"])
        self.assertNotIn("image", data)

    def test_a_peer_without_the_capability_gets_no_files(self):
        machine, link = self.build({"clipboard"})
        self.clipboard.files = [self.make("note.txt")]
        self.clipboard.copy("here")
        self.let_go(machine, link)
        self.assertEqual(len(self.sent), 1)
        self.assertNotIn("files", self.sent[0]["data"])

    def test_files_over_the_limit_leave_the_text(self):
        machine, link = self.build({"clipboard", "clipboard_files"})
        self.clipboard.files = [self.make("big.bin", data=b"\0" * (cf.MAX_BYTES + 1))]
        self.clipboard.copy("here")
        self.let_go(machine, link)
        self.assertEqual(self.sent[0]["data"], {"text": "here"})


if __name__ == "__main__":
    unittest.main()
