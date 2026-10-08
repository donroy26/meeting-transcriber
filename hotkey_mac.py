"""
hotkey_mac.py — Global hotkey on macOS via ``pynput`` (Quartz event tap).

Same public surface as the Windows ``hotkey.py`` so ``main.py`` can use either:
``HotkeyListener(combo, callback).start() -> error code | None``, ``.stop()``,
``error_text(code)``, ``ERROR_LISTENER_TIMEOUT``.

macOS specifics:
  - The combo is NOT exclusive. The keystroke still reaches the focused app,
    so pick a combo nothing else uses (the default ctrl+shift+r is safe).
  - Event taps need the Accessibility permission (System Settings → Privacy &
    Security → Accessibility) for the process that runs Python — usually
    Terminal. Without it the listener starts but never receives a key, so
    ``start()`` reports ``ERROR_NOT_TRUSTED`` and asks macOS to show the
    permission prompt. After granting it, "Re-register hotkey" in the menu
    bar creates a fresh tap; no restart needed.
  - The "stuck modifier after lock screen" failure that drove the Windows
    RegisterHotKey rewrite does not exist here: pynput's hotkey matcher sees
    every event the tap sees, and the tap is not suspended by the lock screen.
"""

from __future__ import annotations

import sys
import threading
import traceback
from collections.abc import Callable

# Returned by start() when the listener thread never reported back at all.
ERROR_LISTENER_TIMEOUT = -1
# Returned by start() when macOS has not granted Accessibility to this process.
ERROR_NOT_TRUSTED = -2

_MODIFIERS = {
    "ctrl": "<ctrl>",
    "control": "<ctrl>",
    "shift": "<shift>",
    "alt": "<alt>",
    "option": "<alt>",
    "win": "<cmd>",
    "windows": "<cmd>",
    "cmd": "<cmd>",
    "command": "<cmd>",
}

# config.toml key name -> pynput key name. Same vocabulary as hotkey.py.
_NAMED_KEYS = {
    "space": "<space>",
    "insert": "<insert>",
    "delete": "<delete>",
    "home": "<home>",
    "end": "<end>",
    "pageup": "<page_up>",
    "pagedown": "<page_down>",
    "esc": "<esc>",
}


def parse_combo(text: str) -> str:
    """
    Parse ``"ctrl+shift+r"`` into pynput's ``"<ctrl>+<shift>+r"`` form.

    Accepts the syntax used in ``config.toml``: modifiers (``ctrl``, ``shift``,
    ``alt``/``option``, ``win``/``cmd``) followed by one main key — a letter or
    digit, ``f1``…``f20``, or a name in ``_NAMED_KEYS``. Raises ``ValueError``
    naming the offending token.
    """
    tokens = [part.strip().lower() for part in str(text).split("+")]
    tokens = [part for part in tokens if part]
    if not tokens:
        raise ValueError(f"empty hotkey combo: {text!r}")

    mods: list[str] = []
    key_token: str | None = None
    for token in tokens:
        if token in _MODIFIERS:
            if key_token is not None:
                raise ValueError(
                    f"modifier {token!r} appears after the main key in {text!r}"
                )
            mods.append(_MODIFIERS[token])
            continue
        if key_token is not None:
            raise ValueError(
                f"hotkey {text!r} names more than one main key "
                f"({key_token!r} and {token!r})"
            )
        key_token = token

    if key_token is None:
        raise ValueError(f"hotkey {text!r} has modifiers but no main key")

    return "+".join(mods + [_key_name(key_token, text)])


def _key_name(token: str, combo: str) -> str:
    if len(token) == 1 and (token.isalpha() or token.isdigit()):
        return token
    if token in _NAMED_KEYS:
        return _NAMED_KEYS[token]
    if token.startswith("f") and token[1:].isdigit():
        number = int(token[1:])
        if 1 <= number <= 20:  # pynput's darwin backend defines f1-f20
            return f"<f{number}>"
        raise ValueError(f"function key {token!r} in {combo!r} is out of range (f1-f20)")
    raise ValueError(
        f"unsupported key {token!r} in hotkey {combo!r}. Supported: a single "
        f"letter or digit, f1-f20, or one of: {', '.join(sorted(_NAMED_KEYS))}"
    )


def error_text(code: int) -> str:
    """Human-readable description of a code returned by ``HotkeyListener.start``."""
    if code == ERROR_NOT_TRUSTED:
        return (
            "macOS has not granted Accessibility to this process. Open System "
            "Settings → Privacy & Security → Accessibility, enable the app that "
            "runs Meeting Transcriber (Terminal), then choose Re-register hotkey"
        )
    if code == ERROR_LISTENER_TIMEOUT:
        return "the hotkey listener did not start"
    return f"error {code}"


def _process_is_trusted() -> bool:
    """Ask macOS whether this process may monitor input; prompt if not."""
    try:
        from HIServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt

        return bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}))
    except Exception:
        # pyobjc not importable the way we expected: let pynput try anyway.
        return True


class HotkeyListener:
    """
    Registers one global hotkey and calls *callback* each time it fires.

    The callback runs on pynput's listener thread and must return promptly.
    """

    def __init__(self, combo: str, callback: Callable[[], None]) -> None:
        self.combo = combo
        self._callback = callback
        self._pynput_combo = parse_combo(combo)
        self._hotkeys = None

    def start(self) -> int | None:
        """
        Start listening. Returns ``None`` on success, ``ERROR_NOT_TRUSTED`` when
        Accessibility is missing (the listener is still started, so it begins
        working as soon as the permission is granted and re-registered).
        """
        self.stop()  # re-register = fresh event tap, picks up new permissions
        try:
            from pynput import keyboard

            self._hotkeys = keyboard.GlobalHotKeys({self._pynput_combo: self._fire})
            self._hotkeys.daemon = True
            self._hotkeys.start()
            self._hotkeys.wait()  # returns once the tap is created
        except Exception:
            self._hotkeys = None
            traceback.print_exc(file=sys.stderr)
            return ERROR_LISTENER_TIMEOUT
        if not _process_is_trusted():
            return ERROR_NOT_TRUSTED
        return None

    def stop(self) -> None:
        hotkeys, self._hotkeys = self._hotkeys, None
        if hotkeys is None:
            return
        try:
            hotkeys.stop()
        except Exception:
            traceback.print_exc(file=sys.stderr)

    def _fire(self) -> None:
        try:
            self._callback()
        except Exception:
            # A failing callback must never take the listener down.
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    # Self-check for the parser (runs anywhere), then a live test on macOS.
    assert parse_combo("ctrl+shift+r") == "<ctrl>+<shift>+r"
    assert parse_combo("Cmd+Alt+Space") == "<cmd>+<alt>+<space>"
    assert parse_combo("f9") == "<f9>"
    for bad in ("", "ctrl+", "ctrl+a+b", "ctrl+f25", "ctrl+bogus"):
        try:
            parse_combo(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should have been rejected")
    print("parse_combo self-check passed")

    if sys.platform == "darwin":
        combo = sys.argv[1] if len(sys.argv) > 1 else "ctrl+shift+r"
        listener = HotkeyListener(combo, lambda: print(f"{combo} fired", flush=True))
        err = listener.start()
        if err is not None:
            print(f"Could not register {combo}: {error_text(err)}", file=sys.stderr)
        print(f"Listening for {combo}. Press it, or Ctrl+C here to exit.", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            listener.stop()
