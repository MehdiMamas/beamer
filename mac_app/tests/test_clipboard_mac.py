"""clipboard_mac's pasteboard logic against a fake pasteboard, plus the one
real AppKit conversion (TIFF to PNG) run on data that never touches the
general pasteboard, so the suite leaves the machine's clipboard alone."""

import struct
import threading
import time
import unittest
from unittest import mock
import zlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import clipboard_mac
from core import protocol


def tiny_png(width=2, height=2):
    """A valid 8-bit RGBA PNG with no filtering, built from the standard
    library so the tests need no imaging dependency."""
    rgba = bytes(range(width * height * 4))[: width * height * 4]
    raw = b"".join(b"\x00" + rgba[y * width * 4:(y + 1) * width * 4] for y in range(height))

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return protocol.PNG_SIGNATURE + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


class FakePasteboard:
    def __init__(self, data=None, string=None):
        self.data = data or {}
        self.string = string
        self.cleared = 0
        self.written = []
        self.count = 1

    def changeCount(self):
        return self.count

    def stringForType_(self, kind):
        return self.string

    def dataForType_(self, kind):
        return self.data.get(kind)

    def types(self):
        return list(self.data) + ([clipboard_mac.AppKit.NSPasteboardTypeString] if self.string is not None else [])

    def clearContents(self):
        self.cleared += 1
        self.count += 1
        return self.count

    def setString_forType_(self, text, kind):
        self.written.append((kind, text))
        return True

    def setData_forType_(self, data, kind):
        self.written.append((kind, bytes(data)))
        return True


@unittest.skipIf(clipboard_mac.AppKit is None, "AppKit is only available on macOS")
class PasteboardImageTests(unittest.TestCase):
    def test_png_on_the_pasteboard_is_returned_as_is(self):
        png = tiny_png()
        board = FakePasteboard({clipboard_mac.AppKit.NSPasteboardTypePNG: png})
        self.assertEqual(clipboard_mac._png_from_pasteboard(board), png)

    def test_tiff_only_is_normalised_to_png(self):
        AppKit = clipboard_mac.AppKit
        png = tiny_png()
        rep = AppKit.NSBitmapImageRep.imageRepWithData_(AppKit.NSData.dataWithBytes_length_(png, len(png)))
        board = FakePasteboard({AppKit.NSPasteboardTypeTIFF: rep.TIFFRepresentation()})
        out = clipboard_mac._png_from_pasteboard(board)
        self.assertTrue(out.startswith(protocol.PNG_SIGNATURE))
        back = AppKit.NSBitmapImageRep.imageRepWithData_(AppKit.NSData.dataWithBytes_length_(out, len(out)))
        self.assertEqual((back.pixelsWide(), back.pixelsHigh()), (2, 2))

    def test_junk_tiff_gives_no_image(self):
        board = FakePasteboard({clipboard_mac.AppKit.NSPasteboardTypeTIFF: b"not a tiff"})
        self.assertIsNone(clipboard_mac._png_from_pasteboard(board))

    def test_empty_pasteboard_gives_no_image(self):
        self.assertIsNone(clipboard_mac._png_from_pasteboard(FakePasteboard()))

    def test_set_contents_writes_text_and_png_after_one_clear(self):
        AppKit = clipboard_mac.AppKit
        board = FakePasteboard()
        fake_appkit = type("FakeAppKit", (), {})()
        fake_appkit.NSPasteboard = type("NSPasteboard", (), {"generalPasteboard": staticmethod(lambda: board)})
        fake_appkit.NSData = AppKit.NSData
        fake_appkit.NSPasteboardTypeString = AppKit.NSPasteboardTypeString
        fake_appkit.NSPasteboardTypePNG = AppKit.NSPasteboardTypePNG
        png = tiny_png()
        real = clipboard_mac.AppKit
        clipboard_mac.AppKit = fake_appkit
        try:
            self.assertTrue(clipboard_mac.set_contents("hello", png))
        finally:
            clipboard_mac.AppKit = real
        self.assertEqual(board.cleared, 1)
        self.assertEqual(board.written, [(AppKit.NSPasteboardTypeString, "hello"), (AppKit.NSPasteboardTypePNG, png)])

    def test_set_contents_with_nothing_writes_nothing(self):
        self.assertFalse(clipboard_mac.set_contents(None, None))



@unittest.skipIf(clipboard_mac.AppKit is None, "AppKit is only available on macOS")
class ChangedContentsTests(unittest.TestCase):
    def setUp(self):
        AppKit = clipboard_mac.AppKit
        self.board = FakePasteboard(string="copied")
        fake_appkit = type("FakeAppKit", (), {})()
        fake_appkit.NSPasteboard = type("NSPasteboard", (), {"generalPasteboard": staticmethod(lambda: self.board)})
        fake_appkit.NSData = AppKit.NSData
        for name in ("NSPasteboardTypeString", "NSPasteboardTypePNG", "NSPasteboardTypeTIFF"):
            setattr(fake_appkit, name, getattr(AppKit, name))
        real = clipboard_mac.AppKit
        clipboard_mac.AppKit = fake_appkit
        clipboard_mac.forget_sync()
        self.addCleanup(setattr, clipboard_mac, "AppKit", real)
        self.addCleanup(clipboard_mac.forget_sync)

    def test_an_unchanged_clipboard_is_sent_once(self):
        self.assertEqual(clipboard_mac.changed_contents()[0], "copied")
        self.assertEqual(clipboard_mac.changed_contents(), (None, None))

    def test_a_new_copy_is_sent_again(self):
        clipboard_mac.changed_contents()
        self.board.count += 1
        self.assertEqual(clipboard_mac.changed_contents()[0], "copied")

    def test_what_the_peer_sent_is_not_sent_back(self):
        clipboard_mac.set_contents("from the pc", None)
        self.assertEqual(clipboard_mac.changed_contents(), (None, None))

    def test_a_failed_read_does_not_hold_the_clipboard_back(self):
        real = clipboard_mac.get_contents
        clipboard_mac.get_contents = lambda: (None, None)
        try:
            self.assertEqual(clipboard_mac.changed_contents(), (None, None))
        finally:
            clipboard_mac.get_contents = real
        self.assertEqual(clipboard_mac.changed_contents()[0], "copied")

    def test_image_failure_keeps_the_text_representation(self):
        def failed_image(kind):
            raise RuntimeError("image provider failed")
        self.board.dataForType_ = failed_image
        self.assertEqual(clipboard_mac.get_contents(), ("copied", None))

    def test_strict_read_distinguishes_a_provider_failure_from_empty(self):
        self.board.string = None
        self.assertEqual(clipboard_mac.read_contents(), (None, None))
        with mock.patch.object(self.board, "stringForType_", side_effect=OSError("provider busy")):
            with self.assertRaises(OSError):
                clipboard_mac.read_contents()
            self.assertEqual(clipboard_mac.get_contents(), (None, None))
        self.board.string = "copied"
        self.assertEqual(clipboard_mac.read_contents(), ("copied", None))

    def test_strict_read_preserves_valid_text_when_the_image_provider_fails(self):
        with mock.patch.object(self.board, "dataForType_", side_effect=OSError("image busy")):
            self.assertEqual(clipboard_mac.read_contents(), ("copied", None))

    def test_strict_read_retries_advertised_text_returning_nil_at_a_stable_count(self):
        count = self.board.count
        with mock.patch.object(self.board, "stringForType_", side_effect=[None, "copied"]):
            with self.assertRaises(OSError):
                clipboard_mac.read_contents()
            self.assertEqual(self.board.count, count)
            self.assertEqual(clipboard_mac.read_contents(), ("copied", None))

    def test_strict_read_retries_advertised_png_returning_nil(self):
        AppKit = clipboard_mac.AppKit
        self.board.string = None
        self.board.data = {AppKit.NSPasteboardTypePNG: tiny_png()}
        original = self.board.dataForType_
        with mock.patch.object(self.board, "dataForType_", return_value=None):
            with self.assertRaises(OSError):
                clipboard_mac.read_contents()
            self.assertEqual(clipboard_mac.get_contents(), (None, None))
        self.assertEqual(clipboard_mac.read_contents(), (None, original(AppKit.NSPasteboardTypePNG)))

    def test_strict_read_keeps_real_empty_text_and_an_unoffered_clipboard_terminal(self):
        self.board.string = ""
        self.assertEqual(clipboard_mac.read_contents(), (None, None))
        self.board.string = None
        self.assertEqual(clipboard_mac.read_contents(), (None, None))

    def test_strict_read_retries_advertised_tiff_returning_nil(self):
        AppKit = clipboard_mac.AppKit
        self.board.string = None
        self.board.data = {AppKit.NSPasteboardTypeTIFF: b"promised TIFF"}
        with mock.patch.object(self.board, "dataForType_", return_value=None):
            with self.assertRaises(OSError):
                clipboard_mac.read_contents()
            self.assertEqual(clipboard_mac.get_contents(), (None, None))

    def test_strict_read_keeps_an_unsupported_tiff_terminal(self):
        AppKit = clipboard_mac.AppKit
        self.board.string = None
        self.board.data = {AppKit.NSPasteboardTypeTIFF: b"not a TIFF"}
        with mock.patch.object(AppKit, "NSBitmapImageRep", create=True) as codec, \
                mock.patch.object(AppKit, "NSBitmapImageFileTypePNG", 0, create=True):
            codec.imageRepWithData_.return_value = None
            self.assertEqual(clipboard_mac.read_contents(), (None, None))
            codec.imageRepWithData_.return_value = mock.Mock()
            codec.imageRepWithData_.return_value.representationUsingType_properties_.return_value = None
            self.assertEqual(clipboard_mac.read_contents(), (None, None))

    def test_strict_read_rejects_a_pasteboard_that_never_stabilises(self):
        def read(kind):
            self.board.count += 1
            return self.board.string
        with mock.patch.object(self.board, "stringForType_", side_effect=read):
            with self.assertRaises(OSError):
                clipboard_mac.read_contents()
            self.assertEqual(clipboard_mac.get_contents(), (None, None))

    def test_copy_during_read_retries_the_newest_stable_snapshot(self):
        original = self.board.stringForType_
        def read(kind):
            old = original(kind)
            self.board.count += 1
            self.board.string = "latest"
            self.board.stringForType_ = original
            return old
        self.board.stringForType_ = read
        self.assertEqual(clipboard_mac.get_contents(), ("latest", None))

    def test_local_copy_during_peer_write_is_not_marked_synchronised(self):
        original = self.board.setString_forType_
        def write(text, kind):
            wrote = original(text, kind)
            self.board.count += 1
            self.board.string = "local copy"
            return wrote
        self.board.setString_forType_ = write
        self.assertTrue(clipboard_mac.set_contents("peer", None))
        self.assertNotEqual(clipboard_mac.write_stamp(), self.board.count)
        self.assertEqual(clipboard_mac.changed_contents(), ("local copy", None))

    def test_a_new_link_sends_it_again(self):
        clipboard_mac.changed_contents()
        clipboard_mac.forget_sync()
        self.assertEqual(clipboard_mac.changed_contents()[0], "copied")

    def test_the_change_stamp_is_the_change_count(self):
        # Wire version 6's responder compares it across a visit (core.receiver.LinkResponder).
        before = clipboard_mac.change_stamp()
        self.assertEqual(before, self.board.count)
        clipboard_mac.set_contents("from a peer", None)
        self.assertNotEqual(clipboard_mac.change_stamp(), before)


class _SlowBoard(FakePasteboard):
    """Counts how many threads are inside it at once, each read and write taking a moment."""

    def __init__(self):
        super().__init__(string="copied")
        self.inside = 0
        self.most = 0
        self.lock = threading.Lock()

    def _visit(self):
        with self.lock:
            self.inside += 1
            self.most = max(self.most, self.inside)
        time.sleep(0.02)
        with self.lock:
            self.inside -= 1

    def stringForType_(self, kind):
        self._visit()
        return self.string

    def clearContents(self):
        self._visit()
        super().clearContents()


class OneAtATimeTests(unittest.TestCase):
    """NSPasteboard is not documented as thread-safe, and reading it from two threads at once has
    raised from inside AppKit: the controller's clipboard reads, its writes and the Mac's responder
    each run on threads of their own, so every read and write takes its turn."""

    def setUp(self):
        AppKit = clipboard_mac.AppKit
        self.board = _SlowBoard()
        fake_appkit = type("FakeAppKit", (), {})()
        fake_appkit.NSPasteboard = type("NSPasteboard", (), {"generalPasteboard": staticmethod(lambda: self.board)})
        fake_appkit.NSData = AppKit.NSData
        for name in ("NSPasteboardTypeString", "NSPasteboardTypePNG", "NSPasteboardTypeTIFF"):
            setattr(fake_appkit, name, getattr(AppKit, name))
        real = clipboard_mac.AppKit
        clipboard_mac.AppKit = fake_appkit
        self.addCleanup(setattr, clipboard_mac, "AppKit", real)

    def test_reads_and_writes_from_several_threads_never_overlap(self):
        work = [clipboard_mac.get_contents, lambda: clipboard_mac.set_contents("x", None)] * 3
        threads = [threading.Thread(target=job) for job in work]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        self.assertEqual(self.board.most, 1)


class _FakeURL:
    def __init__(self, path, file_url=True):
        self._path, self._file = path, file_url

    def path(self):
        return self._path

    def isFileURL(self):
        return self._file


class _FileBoard:
    def __init__(self, urls=()):
        self.urls, self.written, self.cleared, self.options = list(urls), None, 0, None

    def readObjectsForClasses_options_(self, classes, options):
        self.options = options
        return self.urls

    def clearContents(self):
        self.cleared += 1
        return 40 + self.cleared

    def writeObjects_(self, objects):
        self.written = list(objects)
        return True


def _file_appkit(board):
    return mock.Mock(
        NSPasteboard=mock.Mock(generalPasteboard=lambda: board),
        NSURL=mock.Mock(fileURLWithPath_=lambda path: _FakeURL(path)),
        NSPasteboardURLReadingFileURLsOnlyKey="file-only")


class FileListTests(unittest.TestCase):
    def test_read_files_asks_for_file_urls_only_and_skips_other_urls(self):
        board = _FileBoard([_FakeURL("/Users/m/a.txt"), _FakeURL("http://x", file_url=False), _FakeURL("/Users/m/dir")])
        with mock.patch.object(clipboard_mac, "AppKit", _file_appkit(board)):
            self.assertEqual(clipboard_mac.read_files(), ["/Users/m/a.txt", "/Users/m/dir"])
        self.assertEqual(board.options, {"file-only": True})

    def test_read_files_is_empty_without_appkit_or_files_or_on_error(self):
        with mock.patch.object(clipboard_mac, "AppKit", None):
            self.assertEqual(clipboard_mac.read_files(), [])
        with mock.patch.object(clipboard_mac, "AppKit", _file_appkit(_FileBoard())):
            self.assertEqual(clipboard_mac.read_files(), [])
        appkit = _file_appkit(_FileBoard())
        appkit.NSPasteboard.generalPasteboard = mock.Mock(side_effect=RuntimeError("boom"))
        with mock.patch.object(clipboard_mac, "AppKit", appkit):
            self.assertEqual(clipboard_mac.read_files(), [])

    def test_set_files_writes_file_urls_and_remembers_the_write(self):
        board = _FileBoard()
        with mock.patch.object(clipboard_mac, "AppKit", _file_appkit(board)):
            self.assertTrue(clipboard_mac.set_files(["/c/t/a.txt", "/c/t/dir"]))
            self.assertEqual([url.path() for url in board.written], ["/c/t/a.txt", "/c/t/dir"])
            self.assertEqual(clipboard_mac.write_stamp(), 41)
            self.assertFalse(clipboard_mac.set_files([]))
        with mock.patch.object(clipboard_mac, "AppKit", None):
            self.assertFalse(clipboard_mac.set_files(["/x"]))


OneAtATimeTests = unittest.skipIf(clipboard_mac.AppKit is None, "AppKit is only available on macOS")(OneAtATimeTests)


if __name__ == "__main__":
    unittest.main()
