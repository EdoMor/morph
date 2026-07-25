"""Image flow: pluggable image generation (R-501 … R-505).

Backends
--------
``flux``    hosted FLUX endpoint (fal.ai / Replicate-compatible)
``gemini``  Google image model via the generativelanguage API
``local``   local Diffusers / ComfyUI HTTP endpoint
``stub``    deterministic, offline, pure-Python — the default for tests and CI

Every backend takes the same :class:`ImageRequest` and returns the same
:class:`ImageResult`, so switching is configuration, not code.
"""

from __future__ import annotations

import base64
import hashlib
import os
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from . import ToolError, ToolRegistry
from .files import resolve_in_root

PREVIEW_BYTE_LIMIT = 1_500_000
MAX_DIMENSION = 2048


@dataclass
class ImageRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    seed: int | None = None
    count: int = 1

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ToolError("prompt must not be empty")
        for name, value in (("width", self.width), ("height", self.height)):
            if value <= 0 or value > MAX_DIMENSION:
                raise ToolError(f"{name} must be between 1 and {MAX_DIMENSION}, got {value}")
        if not 1 <= self.count <= 8:
            raise ToolError(f"count must be between 1 and 8, got {self.count}")

    def fingerprint(self) -> str:
        blob = f"{self.prompt}|{self.negative_prompt}|{self.width}x{self.height}|{self.seed}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class ImageResult:
    paths: list[str] = field(default_factory=list)
    previews: list[str] = field(default_factory=list)  # data URIs (R-504)
    backend: str = "stub"
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Minimal PNG encoder — keeps the stub backend dependency-free
# ---------------------------------------------------------------------------


def encode_png(pixels: list[bytes], width: int, height: int) -> bytes:
    """Encode RGB scanlines (``height`` rows of ``width * 3`` bytes) as a PNG."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + row for row in pixels)  # filter type 0 per scanline
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _stub_pixels(request: ImageRequest, index: int) -> list[bytes]:
    """Deterministic image derived from the request fingerprint (R-503).

    Same prompt + same seed => byte-identical output. That property is what lets
    the benchmark assert on image generation without a GPU.
    """
    digest = hashlib.sha256(f"{request.fingerprint()}|{index}".encode()).digest()
    r0, g0, b0, r1, g1, b1 = digest[:6]
    swirl = digest[6] or 1

    rows: list[bytes] = []
    for y in range(request.height):
        row = bytearray()
        fy = y / max(request.height - 1, 1)
        for x in range(request.width):
            fx = x / max(request.width - 1, 1)
            ripple = ((x * swirl) ^ (y * (digest[7] or 3))) & 0xFF
            row.append(int(r0 * (1 - fx) + r1 * fx) ^ (ripple >> 4))
            row.append(int(g0 * (1 - fy) + g1 * fy) ^ (ripple >> 5))
            row.append(int(b0 * (1 - fx * fy) + b1 * fx * fy) ^ (ripple >> 6))
        rows.append(bytes(row))
    return rows


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class StubBackend:
    name = "stub"

    async def generate(self, request: ImageRequest) -> list[bytes]:
        return [
            encode_png(_stub_pixels(request, i), request.width, request.height)
            for i in range(request.count)
        ]


class FluxBackend:
    """Hosted FLUX image flow (fal.ai by default, any compatible endpoint works)."""

    name = "flux"
    key_env = "FLUX_API_KEY"
    endpoint_env = "FLUX_ENDPOINT"
    default_endpoint = "https://fal.run/fal-ai/flux/schnell"

    async def generate(self, request: ImageRequest) -> list[bytes]:
        api_key = os.environ.get(self.key_env)
        if not api_key:
            # Actionable, names the exact variable (R-505).
            raise ToolError(
                f"Image backend 'flux' needs {self.key_env} in the environment. "
                f"Set it, or switch to MORPH_IMAGE_BACKEND=stub for offline use."
            )
        endpoint = os.environ.get(self.endpoint_env, self.default_endpoint)
        payload = {
            "prompt": request.prompt,
            "image_size": {"width": request.width, "height": request.height},
            "num_images": request.count,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    endpoint, json=payload, headers={"Authorization": f"Key {api_key}"}
                )
                response.raise_for_status()
                data = response.json()
                return await _collect_images(client, data)
        except httpx.HTTPStatusError as exc:
            raise ToolError(
                f"FLUX endpoint returned {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"FLUX request failed: {exc}") from exc


class GeminiBackend:
    """Google image model via generativelanguage."""

    name = "gemini"
    key_env = "GOOGLE_API_KEY"
    default_model = "gemini-2.5-flash-image"

    async def generate(self, request: ImageRequest) -> list[bytes]:
        api_key = os.environ.get(self.key_env)
        if not api_key:
            raise ToolError(
                f"Image backend 'gemini' needs {self.key_env} in the environment. "
                "Set it, or switch to MORPH_IMAGE_BACKEND=stub for offline use."
            )
        model = os.environ.get("MORPH_IMAGE_MODEL", self.default_model)
        prompt = request.prompt
        if request.negative_prompt:
            prompt = f"{prompt}\n\nAvoid: {request.negative_prompt}"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(url, json=body, headers={"x-goog-api-key": api_key})
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise ToolError(
                f"Gemini image API returned {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"Gemini image request failed: {exc}") from exc

        images: list[bytes] = []
        for candidate in data.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    images.append(base64.b64decode(inline["data"]))
        if not images:
            raise ToolError("Gemini returned no image data — the prompt may have been blocked.")
        return images


class LocalBackend:
    """Local Diffusers / ComfyUI HTTP endpoint — fully offline image flow."""

    name = "local"
    endpoint_env = "MORPH_LOCAL_IMAGE_ENDPOINT"
    default_endpoint = "http://127.0.0.1:7860/sdapi/v1/txt2img"

    async def generate(self, request: ImageRequest) -> list[bytes]:
        endpoint = os.environ.get(self.endpoint_env, self.default_endpoint)
        payload = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "batch_size": request.count,
            "seed": request.seed if request.seed is not None else -1,
        }
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise ToolError(
                f"No local image server at {endpoint}. Start one, or set "
                f"{self.endpoint_env}, or use MORPH_IMAGE_BACKEND=stub."
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"Local image request failed: {exc}") from exc

        images = [base64.b64decode(b) for b in data.get("images", [])]
        if not images:
            raise ToolError("Local image server returned no images.")
        return images


async def _collect_images(client: httpx.AsyncClient, data: dict[str, Any]) -> list[bytes]:
    """Normalise the several shapes hosted endpoints use for their output."""
    entries = data.get("images") or data.get("output") or []
    if isinstance(entries, str):
        entries = [entries]

    images: list[bytes] = []
    for entry in entries:
        if isinstance(entry, dict):
            if entry.get("b64_json"):
                images.append(base64.b64decode(entry["b64_json"]))
                continue
            entry = entry.get("url") or entry.get("image") or ""
        if not isinstance(entry, str) or not entry:
            continue
        if entry.startswith("data:"):
            images.append(base64.b64decode(entry.split(",", 1)[1]))
        elif entry.startswith("http"):
            fetched = await client.get(entry)
            fetched.raise_for_status()
            images.append(fetched.content)
        else:
            images.append(base64.b64decode(entry))

    if not images:
        raise ToolError("Image endpoint returned no usable image data.")
    return images


BACKENDS: dict[str, type] = {
    "stub": StubBackend,
    "flux": FluxBackend,
    "gemini": GeminiBackend,
    "local": LocalBackend,
}


def get_backend(name: str) -> Any:
    """Instantiate an image backend by name (R-502)."""
    backend = BACKENDS.get(name)
    if backend is None:
        raise ToolError(
            f"Unknown image backend {name!r}. Available: {', '.join(sorted(BACKENDS))}"
        )
    return backend()


def register_image_backend(name: str, backend: type) -> None:
    BACKENDS[name] = backend


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


async def run_image_flow(request: ImageRequest, config: Any) -> ImageResult:
    """Generate images and persist them into the workspace (R-504)."""
    request.validate()
    backend_name = getattr(config, "image_backend", "stub")
    backend = get_backend(backend_name)

    blobs = await backend.generate(request)

    out_dir = resolve_in_root(Path(config.root), getattr(config, "image_output_dir", ".morph/images"))
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    short = request.fingerprint()[:10]
    paths: list[str] = []
    previews: list[str] = []

    for index, blob in enumerate(blobs):
        suffix = "png" if blob[:8] == b"\x89PNG\r\n\x1a\n" else "jpg"
        target = out_dir / f"{stamp}-{short}-{index}.{suffix}"
        target.write_bytes(blob)
        paths.append(str(target.relative_to(Path(config.root))))
        if len(blob) <= PREVIEW_BYTE_LIMIT:
            mime = "image/png" if suffix == "png" else "image/jpeg"
            previews.append(f"data:{mime};base64,{base64.b64encode(blob).decode('ascii')}")

    return ImageResult(
        paths=paths,
        previews=previews,
        backend=backend_name,
        meta={
            "prompt": request.prompt,
            "seed": request.seed,
            "size": f"{request.width}x{request.height}",
            "count": len(paths),
        },
    )


def register_image_tools(registry: ToolRegistry, config: Any) -> None:
    async def generate_image(
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        seed: int | None = None,
        count: int = 1,
    ) -> tuple[str, dict[str, Any]]:
        request = ImageRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed,
            count=count,
        )
        result = await run_image_flow(request, config)
        listing = "\n".join(f"- {p}" for p in result.paths)
        return (
            f"Generated {len(result.paths)} image(s) with the {result.backend} backend:\n{listing}",
            {"images": result.paths, "previews": result.previews, "backend": result.backend},
        )

    registry.register(
        "generate_image",
        (
            "Generate images from a text prompt. Writes PNG/JPEG files into the workspace "
            "and returns their paths plus inline previews."
        ),
        {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "negative_prompt": {"type": "string"},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "seed": {"type": "integer", "description": "Fix for reproducible output"},
                "count": {"type": "integer", "description": "1-8 images"},
            },
            "required": ["prompt"],
        },
        generate_image,
    )
