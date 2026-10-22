"""Windows clipboard access via ctypes (user32 + kernel32), guarded like
input_injector.py so this module stays importable -- and its orchestration
testable -- on a machine without ctypes.windll, e.g. the Mac this was
developed on.

The clipboard is a contended, system-wide resource: any other process that
briefly holds it makes OpenClipboard fail, which is normal and expected, not
exceptional. `_open_clipboard` retries a handful of times with a short sleep
before giving up. This still wants confirmation on real Windows hardware
under load -- contention behaviour and timing were chosen to match common
guidance, not measured on a real desktop.

Images cross the wire as PNG but live on this clipboard as CF_DIB, so each
direction converts. The DIB half (`dib_to_bgra`, `bgra_to_dib`) is pure
Python over bytes and slices, so it runs and is tested on the Mac; the PNG
codec is Qt's QImage, which the app already ships for its window, imported
lazily so this module needs neither Qt nor Windows to load. No Pillow: it
would be a second imaging library and a binary wheel in the PyInstaller
build for a job QtGui already does.
"""

import ctypes
import logging
import struct
import sys
import threading
import time
from typing import Optional, Tuple


LOGGER = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

user32 = ctypes.WinDLL("user32", use_last_error=True) if _IS_WINDOWS else None
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True) if _IS_WINDOWS else None

if _IS_WINDOWS:
    user32.OpenClipboard.restype = ctypes.c_int
    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.CloseClipboard.restype = ctypes.c_int
    user32.CloseClipboard.argtypes = []
    user32.EmptyClipboard.restype = ctypes.c_int
    user32.EmptyClipboard.argtypes = []
    user32.GetClipboardData.restype = ctypes.c_void_p
    user32.GetClipboardData.argtypes = [ctypes.c_uint]
    user32.IsClipboardFormatAvailable.restype = ctypes.c_int
    user32.IsClipboardFormatAvailable.argtypes = [ctypes.c_uint]
    user32.SetClipboardData.restype = ctypes.c_void_p
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = ctypes.c_int
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalFree.restype = ctypes.c_void_p
    kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    kernel32.GlobalSize.restype = ctypes.c_size_t
    kernel32.GlobalSize.argtypes = [ctypes.c_void_p]
    user32.GetClipboardSequenceNumber.restype = ctypes.c_uint32
    user32.GetClipboardSequenceNumber.argtypes = []
    user32.CreateWindowExW.restype = ctypes.c_void_p
    user32.CreateWindowExW.argtypes = [
        ctypes.c_uint32, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    user32.DestroyWindow.restype = ctypes.c_int
    user32.DestroyWindow.argtypes = [ctypes.c_void_p]
    user32.RegisterClipboardFormatW.restype = ctypes.c_uint
    user32.RegisterClipboardFormatW.argtypes = [ctypes.c_wchar_p]

CF_DIB = 8
CF_UNICODETEXT = 13
CF_HDROP = 15
DROPFILES_SIZE = 20
DROPEFFECT_COPY = 1
GMEM_MOVEABLE = 0x0002

BI_RGB = 0
BI_BITFIELDS = 3
BITMAPINFOHEADER_SIZE = 40
# BITMAPINFOHEADER, V2, V3, V4 and V5. Windows hands a CF_DIBV5 source back
# through CF_DIB with its 124-byte header intact, so the reader must take
# all of them, not just the 40-byte one it writes.
_DIB_HEADER_SIZES = frozenset({40, 52, 56, 108, 124})
_BGR_MASKS = (0x00FF0000, 0x0000FF00, 0x000000FF)
_ALPHA_MASK = 0xFF000000

# The clipboard's sequence number when it was last sent to the Mac or written from it; None until
# then and after forget_sync. A switch sends the clipboard only when the number has moved since,
# so an unchanged screenshot is not pushed across on every switch.
_synced_sequence = None
_written_sequence = None
_LOCK = threading.RLock()

OPEN_RETRY_ATTEMPTS = 5
OPEN_RETRY_DELAY_SECONDS = 0.05


def dib_to_bgra(dib: bytes):
    """Decode a CF_DIB payload to (width, height, BGRA bytes, top-down rows),
    or None for anything this does not handle: 24/32-bit BI_RGB and 32-bit
    BI_BITFIELDS with the standard BGR(A) masks are what every screenshot
    tool, browser and Office app writes, so a palette, 16-bit or RLE DIB is
    rare enough to drop rather than decode. Never raises on junk."""
    try:
        if len(dib) < BITMAPINFOHEADER_SIZE:
            return None
        size, width, height, planes, bpp, compression = struct.unpack_from("<IiiHHI", dib, 0)
        clr_used = struct.unpack_from("<I", dib, 32)[0]
        if size not in _DIB_HEADER_SIZES or width <= 0 or height == 0 or bpp not in (24, 32):
            return None
        rows = abs(height)
        offset = size + clr_used * 4
        alpha_declared = False
        if compression == BI_BITFIELDS:
            if bpp != 32:
                return None
            masks = struct.unpack_from("<IIII" if size > BITMAPINFOHEADER_SIZE else "<III", dib, BITMAPINFOHEADER_SIZE)
            if size == BITMAPINFOHEADER_SIZE:
                offset += 12
            if tuple(masks[:3]) != _BGR_MASKS:
                return None
            alpha_declared = len(masks) == 4 and masks[3] == _ALPHA_MASK
        elif compression != BI_RGB:
            return None
        stride = (width * bpp // 8 + 3) & ~3
        if len(dib) < offset + stride * rows:
            return None
        row_bytes = width * bpp // 8
        lines = [dib[offset + i * stride: offset + i * stride + row_bytes] for i in range(rows)]
        if height > 0:
            lines.reverse()
        count = width * rows
        if bpp == 24:
            bgr = b"".join(lines)
            out = bytearray(count * 4)
            out[0::4] = bgr[0::3]
            out[1::4] = bgr[1::3]
            out[2::4] = bgr[2::3]
            out[3::4] = b"\xff" * count
        else:
            out = bytearray(b"".join(lines))
            # Under BI_RGB the fourth byte is officially reserved: GDI leaves
            # it zero, which read as alpha is a fully transparent image, while
            # Chrome and Paint.NET put real alpha there. All-zero means opaque;
            # anything else is taken as alpha, as those readers do.
            if not alpha_declared and out[3::4].count(0) == count:
                out[3::4] = b"\xff" * count
        return width, rows, bytes(out)
    except (struct.error, ValueError, OverflowError, MemoryError):
        return None


def bgra_to_dib(width: int, height: int, bgra: bytes) -> bytes:
    """Encode top-down BGRA rows as a 32-bit bottom-up BITMAPINFOHEADER DIB,
    the shape Windows' own tools put on the clipboard. Alpha rides in the
    fourth byte under BI_RGB, which the apps that care (Teams, Office,
    browsers) honour; a screenshot is opaque so nothing is lost either way."""
    stride = width * 4
    header = struct.pack(
        "<IiiHHIIiiII", BITMAPINFOHEADER_SIZE, width, height, 1, 32, BI_RGB, stride * height, 2835, 2835, 0, 0
    )
    rows = [bgra[i * stride: (i + 1) * stride] for i in range(height)]
    return header + b"".join(reversed(rows))


def build_hdrop(paths) -> bytes:
    """A CF_HDROP payload for `paths`: a DROPFILES header (offset of the list, a zero point,
    not in the non-client area, wide characters) and each path UTF-16, NUL-ended, with one more
    NUL closing the list. Pure."""
    body = b"".join(str(path).encode("utf-16-le") + b"\x00\x00" for path in paths) + b"\x00\x00"
    return struct.pack("<IiiII", DROPFILES_SIZE, 0, 0, 0, 1) + body


def parse_hdrop(data: bytes):
    """The paths in a CF_HDROP payload, wide or ANSI, as a list of str; [] for junk. Never raises."""
    try:
        if len(data) < DROPFILES_SIZE:
            return []
        offset, _x, _y, _nc, wide = struct.unpack_from("<IiiII", data, 0)
        if offset < DROPFILES_SIZE or offset > len(data):
            return []
        body = data[offset:]
        if wide:
            text = body[: len(body) - len(body) % 2].decode("utf-16-le", errors="replace")
        else:
            text = body.decode("mbcs" if _IS_WINDOWS else "utf-8", errors="replace")
        paths = []
        for item in text.split("\x00"):
            if not item:
                break
            paths.append(item)
        return paths
    except (struct.error, ValueError, OverflowError):
        return []


def _png_to_bgra(png: bytes):
    """Decode PNG to (width, height, BGRA top-down) with QImage, or None if it
    will not decode. QImage is safe off the GUI thread and needs no
    application object for the built-in PNG format."""
    from PySide6.QtGui import QImage

    image = QImage.fromData(png, "PNG")
    if image.isNull():
        return None
    image = image.convertToFormat(QImage.Format.Format_ARGB32)
    # ARGB32 is one native-endian uint32 per pixel, which on x86 lands in
    # memory as B, G, R, A: the DIB order, so the bytes are used as they are.
    return image.width(), image.height(), bytes(image.constBits())


def _bgra_to_png(width: int, height: int, bgra: bytes) -> Optional[bytes]:
    from PySide6.QtCore import QBuffer, QIODevice
    from PySide6.QtGui import QImage

    image = QImage(bgra, width, height, width * 4, QImage.Format.Format_ARGB32)
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "PNG"):
        return None
    return bytes(buffer.data())


def png_to_dib(png: bytes) -> Optional[bytes]:
    """PNG bytes to a CF_DIB payload, or None. Never raises."""
    try:
        decoded = _png_to_bgra(png)
    except Exception:
        LOGGER.exception("failed to decode a clipboard PNG")
        return None
    if decoded is None:
        return None
    width, height, bgra = decoded
    if width <= 0 or height <= 0 or len(bgra) != width * height * 4:
        return None
    return bgra_to_dib(width, height, bgra)


def dib_to_png(dib: bytes) -> Optional[bytes]:
    """CF_DIB payload to PNG bytes, or None. Never raises."""
    decoded = dib_to_bgra(dib)
    if decoded is None:
        return None
    try:
        return _bgra_to_png(*decoded)
    except Exception:
        LOGGER.exception("failed to encode the clipboard image as PNG")
        return None


def _open_clipboard(win32, attempts=OPEN_RETRY_ATTEMPTS, delay=OPEN_RETRY_DELAY_SECONDS, sleep=time.sleep, owner=None):
    """Try OpenClipboard(owner) up to `attempts` times, sleeping `delay`
    seconds between tries. Pure orchestration over an injected `win32` (the
    real user32 WinDLL, or a duck-typed fake) so the retry/backoff decision
    is testable without a real Windows clipboard."""
    for attempt in range(attempts):
        if win32.OpenClipboard(owner):
            return True
        if attempt + 1 < attempts:
            sleep(delay)
    return False


def _read_clipboard_text(win32, mem32):
    """Copy CF_UNICODETEXT out of the (already-open) clipboard, or return None
    if there is none. Caller holds the clipboard open."""
    handle = win32.GetClipboardData(CF_UNICODETEXT)
    if not handle:
        if win32.IsClipboardFormatAvailable(CF_UNICODETEXT):
            raise OSError("the clipboard text provider could not be read")
        return None
    pointer = mem32.GlobalLock(handle)
    if not pointer:
        raise OSError("the clipboard text could not be locked")
    try:
        text = ctypes.wstring_at(pointer)
    finally:
        mem32.GlobalUnlock(handle)
    return text if text else None


def _read_clipboard_dib(win32, mem32):
    """Copy the CF_DIB payload out of the (already-open) clipboard, or return
    None if there is no image. Windows synthesises CF_DIB from a CF_BITMAP
    or CF_DIBV5 source, so this one format covers them all."""
    handle = win32.GetClipboardData(CF_DIB)
    if not handle:
        if win32.IsClipboardFormatAvailable(CF_DIB):
            raise OSError("the clipboard image provider could not be read")
        return None
    size = mem32.GlobalSize(handle)
    if not size:
        raise OSError("the clipboard image size could not be read")
    pointer = mem32.GlobalLock(handle)
    if not pointer:
        raise OSError("the clipboard image could not be locked")
    try:
        return ctypes.string_at(pointer, size)
    finally:
        mem32.GlobalUnlock(handle)


def _put_clipboard_data(win32, mem32, clipboard_format, payload):
    """Hand `payload` to the (already-open, already-emptied) clipboard under
    `clipboard_format`. Returns True on success. On success, ownership of the
    allocated global memory handle passes to the system and must not be freed
    here; on failure it is freed before returning."""
    handle = mem32.GlobalAlloc(GMEM_MOVEABLE, len(payload))
    if not handle:
        return False
    pointer = mem32.GlobalLock(handle)
    if not pointer:
        mem32.GlobalFree(handle)
        return False
    try:
        ctypes.memmove(pointer, payload, len(payload))
    finally:
        mem32.GlobalUnlock(handle)
    if not win32.SetClipboardData(clipboard_format, handle):
        mem32.GlobalFree(handle)
        return False
    return True


def _read_clipboard_files(win32, mem32):
    """The paths of a CF_HDROP file list on the (already-open) clipboard, or [] if there is none."""
    if not win32.IsClipboardFormatAvailable(CF_HDROP):
        return []
    handle = win32.GetClipboardData(CF_HDROP)
    if not handle:
        raise OSError("the clipboard file list could not be read")
    size = mem32.GlobalSize(handle)
    pointer = mem32.GlobalLock(handle) if size else None
    if not pointer:
        raise OSError("the clipboard file list could not be locked")
    try:
        data = ctypes.string_at(pointer, size)
    finally:
        mem32.GlobalUnlock(handle)
    return parse_hdrop(data)


def _get_files(win32, mem32, attempts=OPEN_RETRY_ATTEMPTS, delay=OPEN_RETRY_DELAY_SECONDS, sleep=time.sleep):
    if not _open_clipboard(win32, attempts, delay, sleep):
        LOGGER.warning("could not open the clipboard for reading files")
        return []
    try:
        return _read_clipboard_files(win32, mem32)
    finally:
        win32.CloseClipboard()


def _set_files(win32, mem32, paths, attempts=OPEN_RETRY_ATTEMPTS, delay=OPEN_RETRY_DELAY_SECONDS, sleep=time.sleep, written=None):
    """Replace the clipboard with `paths` as a file list a copy-paste in Explorer takes (CF_HDROP
    and a "copy" Preferred DropEffect)."""
    if not paths:
        return False
    owner = win32.CreateWindowExW(0, "STATIC", None, 0, 0, 0, 0, 0, -3, None, None, None)
    if not owner:
        LOGGER.warning("could not create the clipboard owner")
        return False
    try:
        if not _open_clipboard(win32, attempts, delay, sleep, owner=owner):
            LOGGER.warning("could not open the clipboard for writing")
            return False
        try:
            if not win32.EmptyClipboard():
                return False
            if not _put_clipboard_data(win32, mem32, CF_HDROP, build_hdrop(paths)):
                return False
            effect = win32.RegisterClipboardFormatW("Preferred DropEffect")
            if effect:
                _put_clipboard_data(win32, mem32, effect, struct.pack("<I", DROPEFFECT_COPY))
            if written is not None:
                written(win32.GetClipboardSequenceNumber())
            return True
        finally:
            win32.CloseClipboard()
    finally:
        win32.DestroyWindow(owner)


def _get_contents(win32, mem32, attempts=OPEN_RETRY_ATTEMPTS, delay=OPEN_RETRY_DELAY_SECONDS, sleep=time.sleep,
                  strict=False):
    if not _open_clipboard(win32, attempts, delay, sleep):
        LOGGER.warning("could not open the clipboard for reading")
        if strict:
            raise OSError("could not open the clipboard for reading")
        return None, None
    failed = False
    try:
        try:
            text = _read_clipboard_text(win32, mem32)
        except Exception:
            failed = True
            LOGGER.exception("failed to read clipboard text")
            text = None
        try:
            dib = _read_clipboard_dib(win32, mem32)
        except Exception:
            failed = True
            LOGGER.exception("failed to read the clipboard image")
            dib = None
    finally:
        win32.CloseClipboard()
    # Converted after the clipboard is closed: a Retina-sized DIB takes
    # tens of milliseconds to encode and nothing else can use the clipboard
    # while it is held.
    png = dib_to_png(dib) if dib is not None else None
    if strict and failed and text is None and png is None:
        raise OSError("the clipboard provider could not be read")
    return text, png


def _set_contents(win32, mem32, text, png, attempts=OPEN_RETRY_ATTEMPTS, delay=OPEN_RETRY_DELAY_SECONDS, sleep=time.sleep, written=None):
    dib = png_to_dib(png) if png is not None else None
    if png is not None and dib is None:
        LOGGER.warning("inbound clipboard image did not decode; setting the text only")
    if text is None and dib is None:
        return False
    # EmptyClipboard with a NULL owner makes SetClipboardData fail. A message-only window
    # belongs to this worker even when the app has no visible panel, and never takes focus.
    owner = win32.CreateWindowExW(0, "STATIC", None, 0, 0, 0, 0, 0, -3, None, None, None)
    if not owner:
        LOGGER.warning("could not create the clipboard owner")
        return False
    try:
        if not _open_clipboard(win32, attempts, delay, sleep, owner=owner):
            LOGGER.warning("could not open the clipboard for writing")
            return False
        try:
            if not win32.EmptyClipboard():
                return False
            wrote = False
            if text is not None:
                # CF_UNICODETEXT is CRLF by convention; a Mac's LF alone reads as one
                # line in the older Windows apps.
                text = text.replace("\r\n", "\n").replace("\n", "\r\n")
                wrote |= _put_clipboard_data(win32, mem32, CF_UNICODETEXT, text.encode("utf-16-le") + b"\x00\x00")
            if dib is not None:
                wrote |= _put_clipboard_data(win32, mem32, CF_DIB, dib)
            if wrote and written is not None:
                written(win32.GetClipboardSequenceNumber())
            return wrote
        finally:
            win32.CloseClipboard()
    finally:
        win32.DestroyWindow(owner)


def get_contents(*, strict=False) -> Tuple[Optional[str], Optional[bytes]]:
    """(text, png) from the Windows clipboard, either None when absent, both
    None if this isn't Windows or reading failed for any reason. Never
    raises."""
    if not _IS_WINDOWS:
        if strict:
            raise OSError("the Windows clipboard is unavailable")
        return None, None
    try:
        with _LOCK:
            return _get_contents(user32, kernel32, strict=strict)
    except Exception as exc:
        LOGGER.exception("failed to read the Windows clipboard")
        if strict:
            raise OSError("failed to read the Windows clipboard") from exc
        return None, None


def read_contents() -> Tuple[Optional[str], Optional[bytes]]:
    """Like get_contents, but raise OSError on failure so a caller can retry an unchanged copy."""
    return get_contents(strict=True)


def changed_contents() -> Tuple[Optional[str], Optional[bytes]]:
    """get_contents(), or (None, None) when the clipboard has not changed since it was last sent
    to or written from the Mac. Either way the current contents count as sent from here on."""
    global _synced_sequence
    if not _IS_WINDOWS:
        return None, None
    sequence = user32.GetClipboardSequenceNumber()
    if sequence and sequence == _synced_sequence:
        return None, None
    text, png = get_contents()
    # Only what was read counts as sent: a failed read must not hold this clipboard back.
    if text is not None or png is not None:
        _synced_sequence = sequence
    return text, png


def change_stamp() -> Optional[int]:
    """A number that changes whenever the clipboard does, or None off Windows."""
    return user32.GetClipboardSequenceNumber() if _IS_WINDOWS else None


def forget_sync() -> None:
    """A new link: the Mac may have lost what it was sent, so the next switch sends again."""
    global _synced_sequence
    _synced_sequence = None


def write_stamp():
    """The last peer write, rather than a newer copy made after CloseClipboard."""
    return _written_sequence


def set_contents(text: Optional[str], png: Optional[bytes]) -> bool:
    """Replace the Windows clipboard with `text` and/or `png`. Returns True
    when at least one was set, False if this isn't Windows or the write
    failed. Never raises."""
    if not _IS_WINDOWS:
        return False
    def remember(sequence):
        global _synced_sequence, _written_sequence
        _synced_sequence = _written_sequence = sequence
    try:
        with _LOCK:
            return _set_contents(user32, kernel32, text, png, written=remember)
    except Exception:
        LOGGER.exception("failed to set the Windows clipboard")
        return False


def read_files() -> list:
    """The absolute paths of the files and folders copied in Explorer, [] when there are none or
    this isn't Windows. Never raises."""
    if not _IS_WINDOWS:
        return []
    try:
        with _LOCK:
            return _get_files(user32, kernel32)
    except Exception:
        LOGGER.exception("failed to read the Windows clipboard's files")
        return []


def set_files(paths) -> bool:
    """Replace the Windows clipboard with the files at `paths`, to paste in Explorer. Returns True
    when set, False if this isn't Windows or the write failed. Never raises."""
    if not _IS_WINDOWS:
        return False
    def remember(sequence):
        global _synced_sequence, _written_sequence
        _synced_sequence = _written_sequence = sequence
    try:
        with _LOCK:
            return _set_files(user32, kernel32, list(paths), written=remember)
    except Exception:
        LOGGER.exception("failed to set the Windows clipboard's files")
        return False
