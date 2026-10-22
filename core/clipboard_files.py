"""Files and folders on the clipboard (WIRE.md section 3, `clipboard`, and section 5).

A copied file list crosses the wire as `files`: entries `{"path", "data"}` for a file (standard
base64) and `{"path": "dir/", "dir": true}` for a folder, every path relative, '/'-separated, from
the parent of the copied item. This module is pure: it builds that list from paths, judges a list
that arrived, and writes it into a fresh folder under Beamer's cache, so a platform clipboard only
has to offer the resulting top-level paths as files. Nothing here touches a clipboard.

A list from a peer is untrusted. `parse_entries` accepts it whole or not at all, and a path that
could leave the folder it is written to, or name something Windows cannot hold, is a reason to
drop the list, never to repair it.
"""

import base64
import binascii
import datetime
import logging
import os
import shutil
import stat
import sys
import uuid

LOGGER = logging.getLogger(__name__)

# Total raw bytes of every file, as for an image: the base64 frame stays under the 16 MiB cap
# beside the text and the paths.
MAX_BYTES = 8 * 1024 * 1024
MAX_ENTRIES = 1024
MAX_PATH_BYTES = 1024
MAX_DEPTH = 32
# What a file system holds in one name on every platform Beamer runs on.
MAX_NAME_BYTES = 255
KEEP_TRANSFERS = 5

_FORBIDDEN_CHARS = frozenset('<>:"|?*\\')
_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL"} | {"COM%d" % n for n in range(1, 10)} | {"LPT%d" % n for n in range(1, 10)})


def valid_path(path, is_dir=False) -> bool:
    """Whether `path` is a safe relative path to write under a folder: '/'-separated, no empty,
    `.` or `..` component, no drive letter, backslash, control character or any of `<>:"|?*`, no
    component ending in a dot or space or named like a Windows device (with or without an
    extension), at most MAX_DEPTH components and MAX_PATH_BYTES of UTF-8. A folder's path ends in
    '/', a file's does not."""
    if not isinstance(path, str) or not path:
        return False
    try:
        size = len(path.encode("utf-8"))
    except UnicodeEncodeError:
        return False
    if size > MAX_PATH_BYTES:
        return False
    if path.endswith("/") != bool(is_dir):
        return False
    parts = path[:-1].split("/") if is_dir else path.split("/")
    if not 1 <= len(parts) <= MAX_DEPTH:
        return False
    for part in parts:
        if not part or part in (".", ".."):
            return False
        if len(part.encode("utf-8")) > MAX_NAME_BYTES or part[-1] in ". ":
            return False
        if any(ord(char) < 32 or ord(char) == 127 or char in _FORBIDDEN_CHARS for char in part):
            return False
        if part.split(".", 1)[0].upper().rstrip() in _RESERVED:
            return False
    return True


def parse_entries(entries):
    """The list a `files` key carries as [(path, bytes)] for a file and [(path, None)] for a
    folder, or None when anything is wrong: not a list, more than MAX_ENTRIES, a path
    `valid_path` refuses, a path twice (compared case-insensitively, a folder and a file of one
    name counting as twice), a file below a path that is itself a file, base64 that is not
    standard and strict, or more than MAX_BYTES of file in all. Never raises."""
    if not isinstance(entries, list) or not entries or len(entries) > MAX_ENTRIES:
        return None
    seen, parsed, total, folders, files = set(), [], 0, set(), set()
    for entry in entries:
        if not isinstance(entry, dict):
            return None
        path = entry.get("path")
        is_dir = entry.get("dir")
        if is_dir is not None and is_dir is not True:
            return None
        if not valid_path(path, bool(is_dir)):
            return None
        key = path.rstrip("/").casefold()
        if key in seen:
            return None
        seen.add(key)
        parts = key.split("/")
        for depth in range(1, len(parts)):
            folders.add("/".join(parts[:depth]))
        if is_dir:
            folders.add(key)
            parsed.append((path, None))
            continue
        files.add(key)
        encoded = entry.get("data")
        # Length first, so an oversized payload costs no decode.
        if not isinstance(encoded, str) or len(encoded) > MAX_BYTES * 4 // 3 + 4:
            return None
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            return None
        total += len(data)
        if total > MAX_BYTES:
            return None
        parsed.append((path, data))
    if files & folders:
        return None
    return parsed


def entries_from_parsed(parsed):
    """The `files` list for what `parse_entries` returned, to pass a list that arrived on."""
    return [{"path": path, "dir": True} if data is None else {"path": path, "data": base64.b64encode(data).decode("ascii")}
            for path, data in parsed]


def _is_link(path) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return True
    # A Windows junction or mount point is not a symlink to Python, but is one to a copy.
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def encode_paths(paths):
    """The `files` list for the files and folders at `paths` (a folder with its whole tree), or
    None when there is nothing to send or it is past a limit (logged, so the rest of the clipboard
    goes without it). Symlinks are skipped, never followed, as is a name the receiving side would
    refuse; two top-level items of one name keep the first. Reads each file once, bounded by what
    is left of MAX_BYTES, and stops walking at the entry limit."""
    entries, seen = [], set()
    state = {"bytes": 0, "over": False}

    def add(path, relative):
        if state["over"] or _is_link(path):
            return
        key = relative.casefold()
        if key in seen:
            return
        if len(entries) >= MAX_ENTRIES:
            state["over"] = True
            return
        if os.path.isdir(path):
            if not valid_path(relative + "/", True):
                LOGGER.info("Clipboard folder %r has a name that cannot cross; skipping it", relative)
                return
            seen.add(key)
            entries.append({"path": relative + "/", "dir": True})
            try:
                names = sorted(os.listdir(path))
            except OSError:
                return
            for name in names:
                add(os.path.join(path, name), relative + "/" + name)
            return
        if not os.path.isfile(path):
            return
        if not valid_path(relative):
            LOGGER.info("Clipboard file %r has a name that cannot cross; skipping it", relative)
            return
        try:
            with open(path, "rb") as handle:
                data = handle.read(MAX_BYTES - state["bytes"] + 1)
        except OSError:
            LOGGER.info("Clipboard file %r could not be read; skipping it", relative)
            return
        state["bytes"] += len(data)
        if state["bytes"] > MAX_BYTES:
            state["over"] = True
            return
        seen.add(key)
        entries.append({"path": relative, "data": base64.b64encode(data).decode("ascii")})

    for path in paths or ():
        path = os.fspath(path)
        add(path, os.path.basename(path.rstrip("/\\")))
    if state["over"]:
        LOGGER.warning("The copied files are past what a clipboard may carry (%d bytes, %d entries); sending without them", MAX_BYTES, MAX_ENTRIES)
        return None
    return entries or None


def cache_root(platform=None, environ=None) -> str:
    """Where received files are kept: ~/Library/Caches/Beamer/Clipboard on a Mac,
    %LOCALAPPDATA%\\Beamer\\Clipboard on Windows, $XDG_CACHE_HOME/beamer/clipboard elsewhere."""
    platform = platform or sys.platform
    environ = os.environ if environ is None else environ
    home = environ.get("HOME") or environ.get("USERPROFILE") or os.path.expanduser("~")
    if platform == "darwin":
        return os.path.join(home, "Library", "Caches", "Beamer", "Clipboard")
    if platform == "win32":
        base = environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        return os.path.join(base, "Beamer", "Clipboard")
    return os.path.join(environ.get("XDG_CACHE_HOME") or os.path.join(home, ".cache"), "beamer", "clipboard")


def write_transfer(parsed, root=None, keep=KEEP_TRANSFERS):
    """Writes `parsed` (from `parse_entries`) into a fresh `<timestamp>-<id>` folder under `root`
    (cache_root() by default), then deletes all but the newest `keep` transfer folders. Returns
    the absolute paths of the top-level items, the files and folders a file manager is to copy.
    Raises OSError when it cannot write."""
    root = root or cache_root()
    name = "%s-%s" % (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S%f"), uuid.uuid4().hex[:8])
    folder = os.path.abspath(os.path.join(root, name))
    os.makedirs(folder)
    tops = []
    for path, data in parsed:
        relative = path.rstrip("/")
        target = os.path.abspath(os.path.join(folder, *relative.split("/")))
        if os.path.commonpath([folder, target]) != folder or target == folder:
            raise OSError("a clipboard path leaves its folder")
        if data is None:
            os.makedirs(target, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "xb") as handle:
                handle.write(data)
        top = os.path.join(folder, relative.split("/")[0])
        if top not in tops:
            tops.append(top)
    prune(root, keep)
    return tops


def prune(root, keep=KEEP_TRANSFERS) -> None:
    """Deletes all but the newest `keep` transfer folders under `root`."""
    try:
        folders = sorted(entry for entry in os.listdir(root) if os.path.isdir(os.path.join(root, entry)))
    except OSError:
        return
    for old in folders[:-keep] if keep else folders:
        shutil.rmtree(os.path.join(root, old), ignore_errors=True)
