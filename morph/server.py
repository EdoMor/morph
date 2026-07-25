"""HTTP server: asyncio HTTP/1.1 with SSE, no web framework required (R-601, R-604).

Written on the stdlib on purpose. Morph is meant to run on a phone (Termux,
iSH, a Pi in a drawer) where installing a compiled ASGI stack is a real cost.
Framing here is deliberately small: request line, headers, optional body,
response — plus chunked SSE for ``/api/chat``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import unquote, urlparse

from .agent import Agent
from .api import ApiError, MorphAPI, sse
from .config import Config, load_config

log = logging.getLogger("morph.server")

WEBAPP_DIR = Path(__file__).resolve().parent.parent / "webapp"
MAX_BODY = 16 * 1024 * 1024
READ_TIMEOUT = 300.0

STATUS_TEXT = {
    200: "OK",
    204: "No Content",
    400: "Bad Request",
    404: "Not Found",
    405: "Method Not Allowed",
    413: "Payload Too Large",
    422: "Unprocessable Entity",
    500: "Internal Server Error",
}


class Request:
    def __init__(self, method: str, target: str, headers: dict[str, str], body: bytes) -> None:
        self.method = method
        self.target = target
        self.headers = headers
        self.body = body
        parsed = urlparse(target)
        self.path = unquote(parsed.path)
        self.query = parsed.query

    def json(self) -> dict[str, Any]:
        if not self.body:
            return {}
        try:
            data = json.loads(self.body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ApiError(f"Body is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ApiError("Body must be a JSON object")
        return data


class MorphServer:
    def __init__(self, api: MorphAPI, webapp_dir: Path | None = None) -> None:
        self.api = api
        self.webapp_dir = Path(webapp_dir or WEBAPP_DIR).resolve()
        self._server: asyncio.AbstractServer | None = None
        self.routes: dict[tuple[str, str], Callable[[Request], Awaitable[Any]]] = {
            ("GET", "/api/health"): lambda r: self.api.health(),
            ("GET", "/api/tools"): lambda r: self.api.tools(),
            ("GET", "/api/skills"): lambda r: self.api.skills(),
            ("GET", "/api/sessions"): lambda r: self.api.sessions(),
            ("POST", "/api/chat/sync"): lambda r: self.api.chat_sync(r.json()),
            ("POST", "/api/image"): lambda r: self.api.image(r.json()),
        }

    # ------------------------------------------------------------------
    async def serve(self, host: str = "127.0.0.1", port: int = 8787) -> None:
        await self.api.start()
        self._server = await asyncio.start_server(self._handle, host, port)
        sockets = self._server.sockets or []
        bound = sockets[0].getsockname() if sockets else (host, port)
        log.info("Morph serving on http://%s:%s", bound[0], bound[1])
        async with self._server:
            await self._server.serve_forever()

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        """Start listening and return the bound port. Used by tests."""
        await self.api.start()
        self._server = await asyncio.start_server(self._handle, host, port)
        return int((self._server.sockets or [])[0].getsockname()[1])

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        await self.api.close()

    # ------------------------------------------------------------------
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                request = await self._read_request(reader)
                if request is None:
                    return
                keep_alive = await self._dispatch(request, writer)
                await writer.drain()
                if not keep_alive:
                    return
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            return
        except asyncio.TimeoutError:
            return
        except Exception:  # noqa: BLE001 - one bad connection must not stop the server
            log.exception("Connection handler failed")
            return
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError):  # pragma: no cover
                pass

    async def _read_request(self, reader: asyncio.StreamReader) -> Request | None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=READ_TIMEOUT)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionResetError):
            return None
        except asyncio.LimitOverrunError:
            return None

        lines = head.decode("latin-1").split("\r\n")
        parts = lines[0].split(" ")
        if len(parts) < 2:
            return None
        method, target = parts[0].upper(), parts[1]

        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" in line:
                key, _, value = line.partition(":")
                headers[key.strip().lower()] = value.strip()

        length = int(headers.get("content-length") or 0)
        if length > MAX_BODY:
            return None
        body = await reader.readexactly(length) if length else b""
        return Request(method, target, headers, body)

    # ------------------------------------------------------------------
    async def _dispatch(self, request: Request, writer: asyncio.StreamWriter) -> bool:
        if request.method == "OPTIONS":
            self._write_response(writer, 204, b"", "text/plain")
            return True

        if request.path == "/api/chat" and request.method == "POST":
            await self._stream_chat(request, writer)
            return False  # SSE owns the connection until it ends

        handler = self.routes.get((request.method, request.path))
        if handler is not None:
            try:
                payload = await handler(request)
            except ApiError as exc:
                self._write_json(writer, {"error": exc.message}, status=exc.status)
                return True
            except Exception as exc:  # noqa: BLE001
                log.exception("Handler failed for %s", request.path)
                self._write_json(
                    writer, {"error": f"{type(exc).__name__}: {exc}"}, status=500
                )
                return True
            self._write_json(writer, payload)
            return True

        if request.path.startswith("/api/sessions/"):
            return await self._session_route(request, writer)

        if request.method in {"GET", "HEAD"}:
            return self._serve_static(request, writer)

        self._write_json(writer, {"error": f"No route for {request.method} {request.path}"}, 404)
        return True

    async def _session_route(self, request: Request, writer: asyncio.StreamWriter) -> bool:
        session_id = request.path.removeprefix("/api/sessions/").strip("/")
        try:
            if request.method == "GET":
                payload = await self.api.session(session_id)
            elif request.method == "DELETE":
                payload = await self.api.delete_session(session_id)
            else:
                self._write_json(writer, {"error": "Method not allowed"}, 405)
                return True
        except ApiError as exc:
            self._write_json(writer, {"error": exc.message}, status=exc.status)
            return True
        self._write_json(writer, payload)
        return True

    async def _stream_chat(self, request: Request, writer: asyncio.StreamWriter) -> None:
        try:
            payload = request.json()
        except ApiError as exc:
            self._write_json(writer, {"error": exc.message}, exc.status)
            return

        self._write_head(
            writer,
            200,
            {
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache, no-transform",
                "Connection": "close",
                "X-Accel-Buffering": "no",
            },
        )
        try:
            async for event in self.api.chat(payload):
                writer.write(sse(event))
                await writer.drain()
        except ApiError as exc:
            writer.write(sse({"type": "error", "message": exc.message}))
        except (ConnectionResetError, BrokenPipeError):
            return  # client navigated away mid-stream
        except Exception as exc:  # noqa: BLE001
            log.exception("Chat stream failed")
            writer.write(sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"}))
        try:
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    def _serve_static(self, request: Request, writer: asyncio.StreamWriter) -> bool:
        rel = request.path.lstrip("/") or "index.html"
        target = (self.webapp_dir / rel).resolve()

        # The webapp directory is its own confinement boundary (R-605).
        if target != self.webapp_dir and self.webapp_dir not in target.parents:
            self._write_json(writer, {"error": "Not found"}, 404)
            return True
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            # SPA fallback so deep links work when installed to a home screen.
            target = self.webapp_dir / "index.html"
            if not target.is_file():
                self._write_json(writer, {"error": "Not found"}, 404)
                return True

        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {
            "application/javascript",
            "application/json",
            "application/manifest+json",
        }:
            content_type += "; charset=utf-8"
        body = b"" if request.method == "HEAD" else target.read_bytes()
        self._write_response(
            writer,
            200,
            body,
            content_type,
            extra={"Content-Length": str(target.stat().st_size)} if request.method == "HEAD" else None,
        )
        return True

    # ------------------------------------------------------------------
    @staticmethod
    def _write_head(
        writer: asyncio.StreamWriter, status: int, headers: dict[str, str]
    ) -> None:
        reason = STATUS_TEXT.get(status, "OK")
        lines = [f"HTTP/1.1 {status} {reason}"]
        base = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
        }
        for key, value in {**base, **headers}.items():
            lines.append(f"{key}: {value}")
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("utf-8"))

    def _write_response(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        body: bytes,
        content_type: str,
        extra: dict[str, str] | None = None,
    ) -> None:
        headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
            "Connection": "keep-alive",
            **(extra or {}),
        }
        self._write_head(writer, status, headers)
        if body:
            writer.write(body)

    def _write_json(
        self, writer: asyncio.StreamWriter, payload: Any, status: int = 200
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self._write_response(writer, status, body, "application/json; charset=utf-8")


def build_server(config: Config | None = None, agent: Agent | None = None) -> MorphServer:
    cfg = config or load_config()
    return MorphServer(MorphAPI(agent or Agent(config=cfg)))


async def serve(config: Config | None = None) -> None:
    cfg = config or load_config()
    server = build_server(cfg)
    try:
        await server.serve(cfg.host, cfg.port)
    finally:
        await server.stop()
