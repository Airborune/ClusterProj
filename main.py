from __future__ import annotations

from pathlib import Path
import sys

from src.stream_ts_clustering.app import launch_application


def _configure_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main() -> None:
    _configure_console()
    launch_application(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
