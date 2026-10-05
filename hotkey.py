"""
hotkey.py — Global hotkey via the Win32 ``RegisterHotKey`` API. Stdlib only.

Why not a low-level keyboard hook (what the `keyboard` library installs): a
hook matches the combo against its *own* model of which keys are currently
down, and that model goes stale permanently whenever Windows switches to the
secure desktop — Win+L, Ctrl+Alt+Del, a UAC prompt, the lock screen after
sleep, a Remote Desktop switch. The key-release event never reaches the hook,
the key stays "down" forever, and the combo silently never matches again.
Windows also silently unhooks callbacks that overrun ``LowLevelHooksTimeout``.
Neither failure is logged, so the app looks alive while the hotkey is dead.

``RegisterHotKey`` has neither failure mode. Windows matches the combo from its
own key state and posts ``WM_HOTKEY`` to the registering thread's message
queue: there is no pressed-key bookkeeping to go stale and no callback for
Windows to time out. The registration survives sleep, lock, and UAC.

The trade-off is that the combo becomes exclusive process-wide. If another
application already owns it, registration fails with
``ERROR_HOTKEY_ALREADY_REGISTERED`` (1409) — loudly, which is the point — and
the caller falls back to the tray menu.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import traceback
from collections.abc import Callable
from ctypes import wintypes

# ------------------------------------------------------------------ constants

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
# Without this, holding the combo repeats WM_HOTKEY at the key-repeat rate and
# recording toggles on and off many times per second.
MOD_NOREPEAT = 0x4000

WM_QUIT = 0x0012
WM_HOTKEY = 0x0312

ERROR_HOTKEY_ALREADY_REGISTERED = 1409

# Only one hotkey per listener, so a fixed id is enough. Ids are per-thread for
# thread-registered hotkeys, so this cannot collide with another application.
_HOTKEY_ID = 1

# Returned by start() when the listener thread never reported back at all.
ERROR_LISTENER_TIMEOUT = -1

_MODIFIERS = {
    "ctrl": MOD_CONTROL,
    "control": MOD_CONTROL,
    "shift": MOD_SHIFT,
    "alt": MOD_ALT,
    "win": MOD_WIN,
    "windows": MOD_WIN,
}

# Named virtual-key codes for the non-alphanumeric keys worth binding.
_NAMED_KEYS = {
    "space": 0x20,
    "pause": 0x13,
    "insert": 0x2D,
    "delete": 0x2E,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
    "esc": 0x1B,
}

_VK_F1 = 0x70

# ---------------------------------------------------------------- win32 setup

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_user32.RegisterHotKey.argtypes = (
    wintypes.HWND,
    ctypes.c_int,
    wintypes.UINT,
    wintypes.UINT,
)
_user32.RegisterHotKey.restype = wintypes.BOOL

_user32.UnregisterHotKey.argtypes = (wintypes.HWND, ctypes.c_int)
_user32.UnregisterHotKey.restype = wintypes.BOOL

# GetMessageW returns -1 on error, so the return type must be signed.
_user32.GetMessageW.argtypes = (
    ctypes.POINTER(wintypes.MSG),
    wintypes.HWND,
    wintypes.UINT,
    wintypes.UINT,
)
_user32.GetMessageW.restype = ctypes.c_int

_user32.PostThreadMessageW.argtypes = (
    wintypes.DWORD,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
)
_user32.PostThreadMessageW.restype = wintypes.BOOL

_kernel32.GetCurrentThreadId.argtypes = ()
_kernel32.GetCurrentThreadId.restype = wintypes.DWORD


# ------------------------------------------------------------------- parsing

def parse_combo(text: str) -> tuple[int, int]:
    """
    Parse ``"ctrl+shift+r"`` into ``(modifier_flags, virtual_key_code)``.

    Accepts the syntax already used in ``config.toml``: any number of
    ``ctrl``/``control``, ``shift``, ``alt``, ``win``/``windows`` modifiers
    followed by one main key — a single letter or digit, ``f1``…``f24``, or one
    of the named keys in ``_NAMED_KEYS``. Raises ``ValueError`` naming the
    offending token.
    """
    tokens = [part.strip().lower() for part in str(text).split("+")]
    tokens = [part for part in tokens if part]
    if not tokens:
        raise ValueError(f"empty hotkey combo: {text!r}")

    mods = MOD_NOREPEAT
    key_token: str | None = None
    for token in tokens:
        if token in _MODIFIERS:
            if key_token is not None:
                raise ValueError(
                    f"modifier {token!r} appears after the main key in {text!r}"
                )
            mods |= _MODIFIERS[token]
            continue
        if key_token is not None:
            raise ValueError(
                f"hotkey {text!r} names more than one main key "
                f"({key_token!r} and {token!r})"
            )
        key_token = token

    if key_token is None:
        raise ValueError(f"hotkey {text!r} has modifiers but no main key")

    return mods, _key_code(key_token, text)


def _key_code(token: str, combo: str) -> int:
    if len(token) == 1 and (token.isalpha() or token.isdigit()):
        return ord(token.upper())
    if token in _NAMED_KEYS:
        return _NAMED_KEYS[token]
    if token.startswith("f") and token[1:].isdigit():
        number = int(token[1:])
        if 1 <= number <= 24:
            return _VK_F1 + number - 1
        raise ValueError(f"function key {token!r} in {combo!r} is out of range (f1-f24)")
    raise ValueError(
        f"unsupported key {token!r} in hotkey {combo!r}. Supported: a single "
        f"letter or digit, f1-f24, or one of: {', '.join(sorted(_NAMED_KEYS))}"
    )


def error_text(code: int) -> str:
    """Human-readable description of a code returned by ``HotkeyListener.start``."""
    if code == ERROR_HOTKEY_ALREADY_REGISTERED:
        return (
            f"error {code}: another application has already claimed this combo"
        )
    if code == ERROR_LISTENER_TIMEOUT:
        return "the hotkey thread did not report back"
    try:
        return f"error {code}: {ctypes.FormatError(code).strip()}"
    except Exception:
        return f"error {code}"


# ------------------------------------------------------------------ listener

class HotkeyListener:
    """
    Registers one global hotkey and calls *callback* each time it fires.

    ``RegisterHotKey`` binds to the calling thread and ``WM_HOTKEY`` is posted
    to that thread's message queue, so registration and the message pump both
    live on the listener thread — never on the caller's.

    The callback runs on the listener thread and must return promptly: a slow
    callback delays subsequent presses (it cannot kill the registration the way
    a slow hook callback could, but the queue still backs up).
    """

    def __init__(self, combo: str, callback: Callable[[], None]) -> None:
        self.combo = combo
        self._callback = callback
        self._mods, self._vk = parse_combo(combo)
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._ready = threading.Event()
        self._error: int | None = None

    def start(self) -> int | None:
        """
        Register the hotkey and start pumping messages.

        Blocks until registration has been attempted. Returns ``None`` on
        success, or the Win32 error code on failure (1409 is
        ``ERROR_HOTKEY_ALREADY_REGISTERED``).
        """
        if self._thread is not None and self._thread.is_alive():
            return self._error
        self._ready.clear()
        self._error = None
        self._thread_id = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="hotkey")
        self._thread.start()
        if not self._ready.wait(timeout=5):
            return ERROR_LISTENER_TIMEOUT
        return self._error

    def stop(self) -> None:
        """Unregister the hotkey and stop the message pump."""
        thread, thread_id = self._thread, self._thread_id
        self._thread = None
        if thread is None or thread_id is None or not thread.is_alive():
            return
        _user32.PostThreadMessageW(thread_id, WM_QUIT, 0, 0)
        thread.join(timeout=5)

    # ----------------------------------------------------------------- thread

    def _run(self) -> None:
        try:
            # Captured before the ready event so stop() can never read a None
            # id from a thread that has already registered.
            self._thread_id = _kernel32.GetCurrentThreadId()
            if not _user32.RegisterHotKey(None, _HOTKEY_ID, self._mods, self._vk):
                self._error = ctypes.get_last_error() or ERROR_LISTENER_TIMEOUT
        except Exception:
            self._error = ERROR_LISTENER_TIMEOUT
            traceback.print_exc(file=sys.stderr)
        finally:
            self._ready.set()

        if self._error is not None:
            return

        try:
            self._pump()
        finally:
            _user32.UnregisterHotKey(None, _HOTKEY_ID)

    def _pump(self) -> None:
        msg = wintypes.MSG()
        while True:
            result = _user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if result in (0, -1):  # WM_QUIT, or a message-queue error
                return
            if msg.message == WM_HOTKEY:
                try:
                    self._callback()
                except Exception:
                    # A failing callback must never take the registration down.
                    traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    combo = sys.argv[1] if len(sys.argv) > 1 else "ctrl+shift+r"
    listener = HotkeyListener(combo, lambda: print(f"{combo} fired", flush=True))
    err = listener.start()
    if err is not None:
        print(f"Could not register {combo}: {error_text(err)}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Registered {combo}. Press it, or Ctrl+C here to exit.", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        listener.stop()
