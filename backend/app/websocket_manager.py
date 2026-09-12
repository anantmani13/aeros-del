"""
WebSocket Connection Manager

Manages multiple concurrent WebSocket connections from dashboard clients.
Responsible for:
- Registering / unregistering live connections
- Broadcasting station updates, AISI changes, and new alerts
- Heartbeat detection of dead clients
"""

import asyncio
import logging
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Centralized WebSocket connection registry and broadcaster.
    """

    def __init__(self):
        self._connections: Set[Any] = set()
        self._lock = asyncio.Lock()
        self._heartbeat_task = None

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    async def connect(self, websocket) -> None:
        """Accept and register a new client connection."""
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info("WebSocket client connected (%d active)",
                    self.connection_count)

    async def disconnect(self, websocket) -> None:
        """Unregister a disconnected client."""
        async with self._lock:
            self._connections.discard(websocket)
        logger.info("WebSocket client disconnected (%d active)",
                    self.connection_count)

    async def broadcast(self, message: Dict[str, Any]) -> int:
        """
        Send a message to all connected clients.

        Returns:
            Number of clients that successfully received the message.
        """
        dead = []
        sent = 0

        for ws in list(self._connections):
            try:
                await ws.send_json(message)
                sent += 1
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)
            logger.info("Pruned %d dead WebSocket clients", len(dead))

        return sent

    async def start_heartbeat(self, interval_seconds: int = 30) -> None:
        """Start the periodic heartbeat that prunes stale connections."""
        if self._heartbeat_task is not None:
            return

        async def _heartbeat():
            while True:
                await asyncio.sleep(interval_seconds)
                dead = []
                for ws in list(self._connections):
                    try:
                        await ws.send_json({"type": "heartbeat"})
                    except Exception:
                        dead.append(ws)
                if dead:
                    async with self._lock:
                        for ws in dead:
                            self._connections.discard(ws)

        self._heartbeat_task = asyncio.create_task(_heartbeat())

    async def stop_heartbeat(self) -> None:
        """Stop the heartbeat task (on shutdown)."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None