"""Command line entry point: ``morph <command>``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from .agent import Agent
from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="morph", description="Self-hosted coding agent platform")
    parser.add_argument("-C", "--workspace", help="Workspace root (default: cwd)")
    parser.add_argument("--provider", help="Model provider: ollama | google | echo")
    parser.add_argument("--model", help="Model name, e.g. gemma3:12b")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="Run one prompt through the agent")
    chat.add_argument("prompt", nargs="+")
    chat.add_argument("--session", help="Resume a session id")
    chat.add_argument("--max-steps", type=int)
    chat.add_argument("--json", action="store_true", help="Emit the raw event stream")

    serve_cmd = sub.add_parser("serve", help="Serve the API and the mobile web app")
    serve_cmd.add_argument("--host")
    serve_cmd.add_argument("--port", type=int)

    sub.add_parser("tools", help="List available tools")
    sub.add_parser("skills", help="List discovered skills")
    sub.add_parser("sessions", help="List stored sessions")

    image = sub.add_parser("image", help="Generate an image")
    image.add_argument("prompt", nargs="+")
    image.add_argument("--width", type=int, default=512)
    image.add_argument("--height", type=int, default=512)
    image.add_argument("--seed", type=int)
    image.add_argument("--count", type=int, default=1)
    image.add_argument("--backend", help="stub | flux | gemini | local")

    bench = sub.add_parser("bench", help="Run the benchmark and print the scorecard")
    bench.add_argument("--output", help="Write the scorecard JSON here")

    improve = sub.add_parser("improve", help="Run the Gemma self-improvement loop")
    improve.add_argument("--iterations", type=int, default=1)
    improve.add_argument("--dry-run", action="store_true")

    return parser


def _config_from_args(args: argparse.Namespace) -> Any:
    config = load_config(args.workspace)
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model = args.model
    if getattr(args, "backend", None):
        config.image_backend = args.backend
    if getattr(args, "host", None):
        config.host = args.host
    if getattr(args, "port", None):
        config.port = args.port
    return config


async def _chat(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    prompt = " ".join(args.prompt)
    async with Agent(config=config) as agent:
        async for event in agent.stream(prompt, session=args.session, max_steps=args.max_steps):
            if args.json:
                print(json.dumps(event.to_dict(), default=str))
                continue
            if event.type == "text" and event.data.get("text"):
                print(event.data["text"])
            elif event.type == "tool_use":
                args_preview = json.dumps(event.data.get("arguments", {}), default=str)[:160]
                print(f"  → {event.data['name']}({args_preview})", file=sys.stderr)
            elif event.type == "tool_result":
                mark = "ok" if event.data.get("ok") else "FAILED"
                print(f"  ← {event.data['name']} [{mark}]", file=sys.stderr)
            elif event.type == "error":
                print(f"error: {event.data.get('message')}", file=sys.stderr)
                return 1
            elif event.type == "done":
                result = event.data["result"]
                if result.get("stop_reason") == "max_steps":
                    print("(stopped: step budget exhausted)", file=sys.stderr)
                return 0 if result.get("error") is None else 1
    return 0


async def _serve(args: argparse.Namespace) -> int:
    from .server import serve

    config = _config_from_args(args)
    print(f"Morph on http://{config.host}:{config.port}  (provider={config.provider} model={config.model})")
    print("Open that URL on your phone and 'Add to Home Screen' to install the app.")
    try:
        await serve(config)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    return 0


async def _tools(args: argparse.Namespace) -> int:
    async with Agent(config=_config_from_args(args)) as agent:
        for name in agent.tools.names():
            tool = agent.tools.get(name)
            assert tool is not None
            first_line = tool.description.strip().splitlines()[0] if tool.description else ""
            print(f"{name:<28} [{tool.source}] {first_line}")
    return 0


async def _skills(args: argparse.Namespace) -> int:
    async with Agent(config=_config_from_args(args)) as agent:
        if not len(agent.skills):
            print("No skills found. Put SKILL.md directories under ./skills/")
            return 0
        for skill in agent.skills.all():
            print(f"{skill.name:<28} {skill.description}")
    return 0


async def _sessions(args: argparse.Namespace) -> int:
    async with Agent(config=_config_from_args(args)) as agent:
        for meta in agent.sessions.list():
            print(f"{meta['id']}  {meta['messages']:>4} msgs  {meta['title']}")
    return 0


async def _image(args: argparse.Namespace) -> int:
    from .tools.image import ImageRequest, run_image_flow

    config = _config_from_args(args)
    request = ImageRequest(
        prompt=" ".join(args.prompt),
        width=args.width,
        height=args.height,
        seed=args.seed,
        count=args.count,
    )
    result = await run_image_flow(request, config)
    for path in result.paths:
        print(path)
    return 0


async def _bench(args: argparse.Namespace) -> int:
    from bench.runner import run_benchmark

    config = _config_from_args(args)
    scorecard = await run_benchmark(config)
    print(scorecard.render())
    if args.output:
        scorecard.write(args.output)
    return 0


async def _improve(args: argparse.Namespace) -> int:
    from selfimprove.loop import run_loop

    config = _config_from_args(args)
    results = await run_loop(config, iterations=args.iterations, dry_run=args.dry_run)
    for entry in results:
        verdict = "ACCEPTED" if entry["accepted"] else "rejected"
        print(f"[{verdict}] {entry['score_before']:.1f} -> {entry['score_after']:.1f}  {entry['summary']}")
    return 0


COMMANDS = {
    "chat": _chat,
    "serve": _serve,
    "tools": _tools,
    "skills": _skills,
    "sessions": _sessions,
    "image": _image,
    "bench": _bench,
    "improve": _improve,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(COMMANDS[args.command](args))
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, don't traceback
        print(f"error: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
