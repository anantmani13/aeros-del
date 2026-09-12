"""
Server Startup Script — SIH AQI System

Usage:
    python backend/run.py
    python backend/run.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is importable regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="SIH Delhi NCR AQI Forecasting System"
    )
    parser.add_argument("--host", default=None,
                        help="Bind host (default: env HOST or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None,
                        help="Bind port (default: env PORT or 8000)")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")
    parser.add_argument("--demo", action="store_true",
                        help="Force demo mode (no external APIs)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Import settings AFTER sys.path injection
    from backend.app.config import settings

    if args.demo:
        import os
        os.environ["DEMO_MODE"] = "true"

    host = args.host or settings.host
    port = args.port or settings.port

    import uvicorn

    print(f"\n  SIH AQI System  (demo={settings.demo_mode})")
    print(f"  API + Dashboard → http://{host}:{port}\n")

    uvicorn.run(
        "backend.app.main:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()