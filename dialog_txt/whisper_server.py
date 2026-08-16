"""Backward-compatible module entry point for the network server."""

from .network_server import Handler as WhisperRequestHandler
from .network_server import main

__all__ = ["WhisperRequestHandler", "main"]


if __name__ == "__main__":
    main()
