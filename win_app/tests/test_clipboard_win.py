"""The clipboard image path that runs on any platform: the pure DIB codec,
the clipboard message's image payload, and the read/write orchestration
over fake user32/kernel32 backed by real ctypes buffers. The Qt PNG codec is
replaced with a marker encoding so nothing here needs PySide6 or Windows."""

import base64
import ctypes
import struct
import sys
import unittest
from unittest import mock
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import clipboard_win
from core import protocol
from core.tests import pngs
from clipboard_win import BI_BITFIELDS, BI_RGB, CF_DIB, CF_UNICODETEXT, bgra_to_dib, dib_to_bgra


def header(width, height, bpp, compression=BI_RGB, size=40):
    fields = struct.pack("<IiiHHIIiiII", size, width, height, 1, bpp, compression, 0, 0, 0, 0, 0)
    return fields + b"\x00" * (size - 40)


# 2x2 image, top-down, BGRA: blue, green / red, half-transparent white.
BGRA = bytes([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 128])
PNG = pngs.png()


class DibCodecTests(unittest.TestCase):
    def test_32_bit_round_trip(self):
        dib = bgra_to_dib(2, 2, BGRA)
        self.assertEqual(len(dib), 40 + 16)
        self.assertEqual(dib_to_bgra(dib), (2, 2, BGRA))

    def test_written_dib_is_bottom_up(self):
        dib = bgra_to_dib(2, 2, BGRA)
        self.assertEqual(dib[40:48], BGRA[8:16])

    def test_24_bit_rows_are_padded_and_alpha_is_opaque(self):
        # Width 3 at 24 bits is 9 bytes a row, padded to 12; one row, so
        # bottom-up and top-down coincide.
        row = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9]) + b"\x00\x00\x00"
        self.assertEqual(
            dib_to_bgra(header(3, 1, 24) + row),
            (3, 1, bytes([1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255])),
        )

    def test_negative_height_is_top_down(self):
        self.assertEqual(dib_to_bgra(header(2, -2, 32) + BGRA), (2, 2, BGRA))

    def test_bi_rgb_all_zero_alpha_reads_as_opaque(self):
        pixels = bytes([10, 20, 30, 0] * 4)
        self.assertEqual(dib_to_bgra(header(2, -2, 32) + pixels)[2], bytes([10, 20, 30, 255] * 4))

    def test_bi_rgb_real_alpha_is_kept(self):
        pixels = bytes([10, 20, 30, 0, 10, 20, 30, 200] * 2)
        self.assertEqual(dib_to_bgra(header(2, -2, 32) + pixels)[2], pixels)

    def test_bitfields_with_v5_alpha_mask_keeps_zero_alpha(self):
        masks = struct.pack("<IIII", 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
        head = header(2, -2, 32, BI_BITFIELDS, size=124)
        head = head[:40] + masks + head[56:]
        pixels = bytes([10, 20, 30, 0] * 4)
        self.assertEqual(dib_to_bgra(head + pixels)[2], pixels)

    def test_bitfields_with_40_byte_header_reads_masks_after_it(self):
        masks = struct.pack("<III", 0x00FF0000, 0x0000FF00, 0x000000FF)
        self.assertEqual(dib_to_bgra(header(2, -2, 32, BI_BITFIELDS) + masks + BGRA), (2, 2, BGRA))

    def test_v2_header_mask_overread_does_not_produce_transparent_output(self):
        # clipboard-injection-2: a 52-byte BITMAPV2INFOHEADER carries only 3
        # BI_BITFIELDS masks (ending at offset 52), but dib_to_bgra always
        # unpacks 4 (through offset 56) whenever size > BITMAPINFOHEADER_SIZE,
        # so masks[3] is actually the DIB's first 4 pixel bytes. The only way
        # those bytes can equal _ALPHA_MASK (0xFF000000, i.e. le bytes
        # 00 00 00 FF) is for that same pixel's own real alpha byte to be
        # 0xFF -- which makes it impossible for every real alpha byte in the
        # image to be zero, so the all-zero-alpha-means-opaque fallback this
        # would otherwise suppress was never going to fire anyway. The
        # over-read is real but inert: this pixel, and every value across the
        # small range that could trigger it, decode identically to a version
        # that reads the correct 3 masks for a size-52 header.
        masks = struct.pack("<III", 0x00FF0000, 0x0000FF00, 0x000000FF)
        head = header(1, -1, 32, BI_BITFIELDS, size=52)
        head = head[:40] + masks
        pixel = bytes([0x00, 0x00, 0x00, 0xFF])  # B, G, R, A=0xFF
        self.assertEqual(dib_to_bgra(head + pixel), (1, 1, pixel))

    def test_unusual_masks_are_refused(self):
        masks = struct.pack("<III", 0x000000FF, 0x0000FF00, 0x00FF0000)
        self.assertIsNone(dib_to_bgra(header(2, -2, 32, BI_BITFIELDS) + masks + BGRA))

    def test_junk_is_refused_without_raising(self):
        for junk in (
            b"",
            b"short",
            header(2, 2, 8) + b"\x00" * 64,
            header(2, 2, 32, compression=1) + BGRA,
            header(2, 2, 32) + BGRA[:-1],
            header(0, 2, 32) + BGRA,
            header(2, 0, 32) + BGRA,
            header(2, 2, 32, size=48) + BGRA,
            header(1 << 30, 1 << 30, 32) + BGRA,
        ):
            with self.subTest(junk=junk[:48]):
                self.assertIsNone(dib_to_bgra(junk))


class ClipboardMessageTests(unittest.TestCase):
    def test_text_only_message_is_unchanged(self):
        self.assertEqual(protocol.clipboard_msg("hello"), {"type": "clipboard", "data": {"text": "hello"}})
        self.assertIsNone(protocol.clipboard_image(protocol.clipboard_msg("hello")["data"]))

    def test_image_round_trips_through_the_sealed_wire(self):
        token = base64.urlsafe_b64encode(bytes(range(32))).decode("ascii").rstrip("=")
        sender = protocol.LinkSession(token, protocol.ROLE_INITIATOR)
        receiver = protocol.LinkSession(token, protocol.ROLE_RESPONDER)
        receiver.accept_preamble(sender.preamble())
        sender.accept_preamble(receiver.preamble())
        frame = sender.seal(protocol.clipboard_msg("shot.png", PNG))
        message = receiver.open(frame[protocol.HEADER_SIZE:])
        self.assertEqual(message["data"]["text"], "shot.png")
        self.assertEqual(message["data"]["image_format"], "png")
        self.assertEqual(protocol.clipboard_image(message["data"]), PNG)

    def test_image_only_message_carries_no_text_key(self):
        self.assertNotIn("text", protocol.clipboard_msg(None, PNG)["data"])

    def test_oversized_image_is_refused(self):
        big = protocol.PNG_SIGNATURE + b"\x00" * protocol.CLIPBOARD_IMAGE_MAX_BYTES
        self.assertIsNone(protocol.clipboard_image(protocol.clipboard_msg(None, big)["data"]))

    def test_malformed_payloads_are_refused_without_raising(self):
        good = protocol.clipboard_msg(None, PNG)["data"]
        for data in (
            None,
            {},
            {"image": good["image"]},
            {"image": good["image"], "image_format": "jpeg"},
            {"image": "not base64!", "image_format": "png"},
            {"image": 42, "image_format": "png"},
            {"image": "", "image_format": "png"},
            {"image": protocol.clipboard_msg(None, b"GIF89a junk")["data"]["image"], "image_format": "png"},
        ):
            with self.subTest(data=data):
                self.assertIsNone(protocol.clipboard_image(data))


class FakeMem32:
    """kernel32 stand-in whose handles are real ctypes buffers, so the
    string_at/memmove in clipboard_win run for real on this Mac."""

    def __init__(self):
        self.buffers = {}
        self.freed = []

    def handle_for(self, payload):
        # A str becomes a wchar_t buffer, since wstring_at reads wchar_t and
        # that is four bytes here and two on Windows.
        if isinstance(payload, str):
            buffer = ctypes.create_unicode_buffer(payload)
        else:
            buffer = ctypes.create_string_buffer(payload, len(payload))
        handle = ctypes.addressof(buffer)
        self.buffers[handle] = buffer
        return handle

    def GlobalAlloc(self, flags, size):
        return self.handle_for(b"\x00" * size)

    def GlobalSize(self, handle):
        return ctypes.sizeof(self.buffers[handle])

    def GlobalLock(self, handle):
        return handle

    def GlobalUnlock(self, handle):
        return 1

    def GlobalFree(self, handle):
        self.freed.append(handle)

    def contents(self, handle):
        return self.buffers[handle].raw


class FakeWin32:
    def __init__(self, mem32, formats=None):
        self.mem32 = mem32
        self.formats = dict(formats or {})
        self.emptied = 0
        self.open_owner = None
        self.destroyed = []

    def CreateWindowExW(self, *args):
        return 123

    def DestroyWindow(self, owner):
        self.destroyed.append(owner)
        return 1

    def OpenClipboard(self, owner):
        self.open_owner = owner
        return 1

    def CloseClipboard(self):
        return 1

    def EmptyClipboard(self):
        self.emptied += 1
        self.formats.clear()
        return 1

    def GetClipboardData(self, clipboard_format):
        return self.formats.get(clipboard_format, 0)

    def IsClipboardFormatAvailable(self, clipboard_format):
        return clipboard_format in self.formats

    def SetClipboardData(self, clipboard_format, handle):
        self.formats[clipboard_format] = handle
        return handle

    def RegisterClipboardFormatW(self, name):
        return 49152 + sum(name.encode("ascii")) % 1000

    def GetClipboardSequenceNumber(self):
        return getattr(self, "sequence", 1)


def fake_png_to_bgra(png):
    if not png.startswith(b"FAKEPNG"):
        return None
    width, height = struct.unpack("<II", png[7:15])
    return width, height, png[15:]


def fake_bgra_to_png(width, height, bgra):
    return b"FAKEPNG" + struct.pack("<II", width, height) + bgra


class ClipboardOrchestrationTests(unittest.TestCase):
    def setUp(self):
        patches = [
            mock.patch.object(clipboard_win, "_png_to_bgra", fake_png_to_bgra),
            mock.patch.object(clipboard_win, "_bgra_to_png", fake_bgra_to_png),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.mem32 = FakeMem32()

    def test_strict_read_retries_native_contention_without_another_copy(self):
        win32 = FakeWin32(self.mem32, {CF_UNICODETEXT: self.mem32.handle_for("copied")})
        with mock.patch.multiple(clipboard_win, _IS_WINDOWS=True, user32=win32, kernel32=self.mem32), \
                mock.patch.object(clipboard_win, "_open_clipboard", return_value=False):
            with self.assertRaises(OSError):
                clipboard_win.read_contents()
            self.assertEqual(clipboard_win.get_contents(), (None, None))
        with mock.patch.multiple(clipboard_win, _IS_WINDOWS=True, user32=win32, kernel32=self.mem32):
            self.assertEqual(clipboard_win.read_contents(), ("copied", None))

    def test_strict_read_distinguishes_a_native_exception_from_empty(self):
        win32 = FakeWin32(self.mem32)
        with mock.patch.multiple(clipboard_win, _IS_WINDOWS=True, user32=win32, kernel32=self.mem32):
            self.assertEqual(clipboard_win.read_contents(), (None, None))
            with mock.patch.object(win32, "GetClipboardData", side_effect=OSError("provider busy")):
                with self.assertRaises(OSError):
                    clipboard_win.read_contents()
                self.assertEqual(clipboard_win.get_contents(), (None, None))

    def test_strict_read_preserves_valid_text_when_the_image_provider_fails(self):
        win32 = FakeWin32(self.mem32, {CF_UNICODETEXT: self.mem32.handle_for("copied")})
        original = win32.GetClipboardData
        def read(kind):
            if kind == CF_DIB:
                raise OSError("image provider busy")
            return original(kind)
        with mock.patch.multiple(clipboard_win, _IS_WINDOWS=True, user32=win32, kernel32=self.mem32), \
                mock.patch.object(win32, "GetClipboardData", side_effect=read):
            self.assertEqual(clipboard_win.read_contents(), ("copied", None))

    def test_strict_read_reports_an_offered_format_that_cannot_be_read(self):
        win32 = FakeWin32(self.mem32, {CF_UNICODETEXT: self.mem32.handle_for("copied")})
        with mock.patch.multiple(clipboard_win, _IS_WINDOWS=True, user32=win32, kernel32=self.mem32):
            with mock.patch.object(win32, "GetClipboardData", return_value=0):
                with self.assertRaises(OSError):
                    clipboard_win.read_contents()
            with mock.patch.object(self.mem32, "GlobalLock", return_value=0):
                with self.assertRaises(OSError):
                    clipboard_win.read_contents()

    def test_get_contents_reads_text_and_converts_the_dib(self):
        win32 = FakeWin32(
            self.mem32,
            {
                CF_UNICODETEXT: self.mem32.handle_for("hi"),
                CF_DIB: self.mem32.handle_for(bgra_to_dib(2, 2, BGRA)),
            },
        )
        self.assertEqual(clipboard_win._get_contents(win32, self.mem32), ("hi", fake_bgra_to_png(2, 2, BGRA)))

    def test_get_contents_with_an_undecodable_dib_still_returns_the_text(self):
        win32 = FakeWin32(
            self.mem32,
            {
                CF_UNICODETEXT: self.mem32.handle_for("hi"),
                CF_DIB: self.mem32.handle_for(b"garbage" * 10),
            },
        )
        self.assertEqual(clipboard_win._get_contents(win32, self.mem32), ("hi", None))

    def test_set_contents_writes_text_and_dib_after_one_empty(self):
        win32 = FakeWin32(self.mem32)
        self.assertTrue(clipboard_win._set_contents(win32, self.mem32, "hi", fake_bgra_to_png(2, 2, BGRA)))
        self.assertEqual(win32.emptied, 1)
        self.assertEqual(self.mem32.contents(win32.formats[CF_UNICODETEXT]), "hi".encode("utf-16-le") + b"\x00\x00")
        self.assertEqual(self.mem32.contents(win32.formats[CF_DIB]), bgra_to_dib(2, 2, BGRA))
        self.assertEqual(self.mem32.freed, [])

    def test_write_uses_an_owned_message_window_even_when_the_panel_is_hidden(self):
        win32 = FakeWin32(self.mem32)
        original = win32.SetClipboardData
        win32.SetClipboardData = lambda kind, handle: original(kind, handle) if win32.open_owner else 0
        self.assertTrue(clipboard_win._set_contents(win32, self.mem32, "hidden", None))
        self.assertEqual(win32.destroyed, [123])

    def test_failed_empty_does_not_write_into_another_apps_clipboard(self):
        win32 = FakeWin32(self.mem32)
        win32.EmptyClipboard = lambda: 0
        self.assertFalse(clipboard_win._set_contents(win32, self.mem32, "new", None))
        self.assertEqual(win32.formats, {})

    def test_image_read_failure_does_not_drop_valid_text(self):
        win32 = FakeWin32(self.mem32, {CF_UNICODETEXT: self.mem32.handle_for("copied")})
        original = win32.GetClipboardData
        def read(kind):
            if kind == CF_DIB:
                raise OSError("image provider failed")
            return original(kind)
        win32.GetClipboardData = read
        self.assertEqual(clipboard_win._get_contents(win32, self.mem32), ("copied", None))

    def test_copy_after_close_is_not_marked_as_written_from_peer(self):
        win32 = FakeWin32(self.mem32)
        win32.sequence = 10
        win32.GetClipboardSequenceNumber = lambda: win32.sequence
        def close():
            win32.sequence += 1
            return 1
        win32.CloseClipboard = close
        with mock.patch.multiple(clipboard_win, _IS_WINDOWS=True, user32=win32, kernel32=self.mem32, _synced_sequence=None):
            self.assertTrue(clipboard_win.set_contents("peer", None))
            self.assertEqual(clipboard_win._synced_sequence, 10)

    def test_contention_releases_the_owner_window_and_the_next_attempt_succeeds(self):
        win32 = FakeWin32(self.mem32)
        opened = win32.OpenClipboard
        calls = []
        def contended(owner):
            calls.append(owner)
            return 0 if len(calls) <= 5 else opened(owner)
        win32.OpenClipboard = contended
        self.assertFalse(clipboard_win._set_contents(win32, self.mem32, "copy", None, sleep=lambda _: None))
        self.assertTrue(clipboard_win._set_contents(win32, self.mem32, "copy", None, sleep=lambda _: None))
        self.assertEqual(win32.destroyed, [123, 123])
        self.assertEqual(self.mem32.contents(win32.formats[CF_UNICODETEXT]), b"c\x00o\x00p\x00y\x00\x00\x00")

    def test_rapid_large_unicode_writes_keep_the_latest_text(self):
        win32 = FakeWin32(self.mem32)
        copies = [f"copy {n}: café 日本語 \U0001d11e" for n in range(200)]
        copies.append("\U0001d11e" * (protocol.CLIPBOARD_MAX_BYTES // 4))
        for text in copies:
            self.assertTrue(clipboard_win._set_contents(win32, self.mem32, text, None))
            self.assertEqual(clipboard_win._get_contents(win32, self.mem32), (text, None))

    def test_set_contents_writes_windows_line_endings(self):
        win32 = FakeWin32(self.mem32)
        self.assertTrue(clipboard_win._set_contents(win32, self.mem32, "one\ntwo\r\nthree", None))
        self.assertEqual(
            self.mem32.contents(win32.formats[CF_UNICODETEXT]),
            "one\r\ntwo\r\nthree".encode("utf-16-le") + b"\x00\x00",
        )

    def test_set_contents_with_an_undecodable_png_sets_the_text_only(self):
        win32 = FakeWin32(self.mem32)
        self.assertTrue(clipboard_win._set_contents(win32, self.mem32, "hi", b"\x89PNG but not really"))
        self.assertEqual(sorted(win32.formats), [CF_UNICODETEXT])

    def test_set_contents_with_only_an_undecodable_png_leaves_the_clipboard_alone(self):
        win32 = FakeWin32(self.mem32, {CF_UNICODETEXT: self.mem32.handle_for(b"k\x00\x00\x00")})
        self.assertFalse(clipboard_win._set_contents(win32, self.mem32, None, b"junk"))
        self.assertEqual(win32.emptied, 0)

    @unittest.skipIf(sys.platform == "win32", "the real clipboard answers on Windows")
    def test_get_contents_never_raises_off_windows(self):
        # The public entry points are the ones the app calls on both platforms, so they must
        # return empty rather than raise where there is no Windows clipboard. On Windows they
        # really do write, which is why this is skipped there rather than asserted the other
        # way -- it would then be a live-clipboard test, and the build must not clobber
        # whatever the person running it had copied.
        self.assertEqual(clipboard_win.get_contents(), (None, None))
        self.assertFalse(clipboard_win.set_contents("x", PNG))

    def test_the_change_stamp_is_the_sequence_number(self):
        # Wire version 6's responder compares it across a visit (core.receiver.LinkResponder).
        if sys.platform == "win32":
            self.assertEqual(clipboard_win.change_stamp(), clipboard_win.user32.GetClipboardSequenceNumber())
        else:
            self.assertIsNone(clipboard_win.change_stamp())





class FileListTests(unittest.TestCase):
    def test_hdrop_round_trips_wide_paths(self):
        paths = ["C:/Users/a/notes.txt", "C:/Users/a/dossier \u00e9t\u00e9", "D:/x"]
        data = clipboard_win.build_hdrop(paths)
        self.assertEqual(struct.unpack_from("<IiiII", data, 0), (20, 0, 0, 0, 1))
        self.assertTrue(data.endswith(b"\x00\x00\x00\x00"))
        self.assertEqual(clipboard_win.parse_hdrop(data), paths)

    def test_hdrop_ansi_form_is_read(self):
        data = struct.pack("<IiiII", 20, 0, 0, 0, 0) + b"C:/a.txt\x00C:/b.txt\x00\x00"
        with mock.patch.object(clipboard_win, "_IS_WINDOWS", False):
            self.assertEqual(clipboard_win.parse_hdrop(data), ["C:/a.txt", "C:/b.txt"])

    def test_hdrop_junk_never_raises(self):
        for data in (b"", b"\x00" * 5, struct.pack("<IiiII", 999, 0, 0, 0, 1), struct.pack("<IiiII", 4, 0, 0, 0, 1), b"\xff" * 40):
            with self.subTest(data=data):
                self.assertIsInstance(clipboard_win.parse_hdrop(data), list)

    def test_set_files_writes_the_list_and_a_copy_effect_and_remembers_the_write(self):
        mem32 = FakeMem32()
        win32 = FakeWin32(mem32)
        win32.sequence = 77
        seen = []
        self.assertTrue(clipboard_win._set_files(win32, mem32, ["C:/t/a.txt"], written=seen.append))
        self.assertEqual(clipboard_win.parse_hdrop(mem32.contents(win32.formats[clipboard_win.CF_HDROP])), ["C:/t/a.txt"])
        effect = [handle for fmt, handle in win32.formats.items() if fmt != clipboard_win.CF_HDROP]
        self.assertEqual(struct.unpack("<I", mem32.contents(effect[0]))[0], clipboard_win.DROPEFFECT_COPY)
        self.assertEqual(seen, [77])
        self.assertEqual(win32.destroyed, [123])

    def test_set_files_with_nothing_leaves_the_clipboard(self):
        mem32 = FakeMem32()
        win32 = FakeWin32(mem32, {CF_UNICODETEXT: mem32.handle_for("keep")})
        self.assertFalse(clipboard_win._set_files(win32, mem32, []))
        self.assertEqual(win32.emptied, 0)

    def test_read_files_reads_what_set_files_wrote(self):
        mem32 = FakeMem32()
        win32 = FakeWin32(mem32)
        clipboard_win._set_files(win32, mem32, ["C:/t/a.txt", "C:/t/dir"])
        self.assertEqual(clipboard_win._get_files(win32, mem32), ["C:/t/a.txt", "C:/t/dir"])

    def test_read_files_is_empty_without_a_list_and_off_windows(self):
        mem32 = FakeMem32()
        self.assertEqual(clipboard_win._get_files(FakeWin32(mem32, {CF_UNICODETEXT: mem32.handle_for("t")}), mem32), [])
        with mock.patch.object(clipboard_win, "_IS_WINDOWS", False):
            self.assertEqual(clipboard_win.read_files(), [])
            self.assertFalse(clipboard_win.set_files(["x"]))


if __name__ == "__main__":
    unittest.main()
