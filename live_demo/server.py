"""Live biomechanics demo server.

Serves the web app (web/dist), the recorded sessions (replay mode), the export endpoint, and live
mode: ``/ws/live`` runs the real streaming pipeline on this machine (see live.py) and streams every
result to the browser.

    # from the repository folder:
    python live_demo/server.py                 # then open http://localhost:8000
    python live_demo/server.py --port 8080 --open --lpf-repo ../LivePoseFormer
"""
import argparse
import asyncio
import json
import sys
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from export import build_export

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from stream_sim.lpf_repo import HELP, find_lpf_repo  # noqa: E402

SESSIONS = HERE / "sessions"
DIST = HERE / "web" / "dist"

app = FastAPI(title="Live biomechanics demo")
CONFIG = {"lpf_repo": None, "device": "cuda"}
_runner = None


def runner():
    """The live runner, created on first use (it imports torch and the pipeline)."""
    global _runner
    if _runner is None:
        from live import LiveRunner
        _runner = LiveRunner(CONFIG["lpf_repo"], CONFIG["device"])
    return _runner


@app.get("/api/sessions")
def sessions():
    out = []
    for f in sorted(SESSIONS.glob("*/session.json")):
        s = json.loads(f.read_text())
        out.append({"name": s["name"], "trial": s["trial"], "subject_id": s["subject_id"], "task": s["task"],
                    "n_frames": s["n_frames"], "fps": s["fps"], "recorded_at": s["recorded_at"],
                    "pipeline": s["pipeline"], "live": "live" in s,
                    "size_mb": round(sum(p.stat().st_size for p in f.parent.iterdir()) / 1e6, 1)})
    return out


@app.get("/api/options")
def options(session: str = ""):
    """What live mode can run on this machine (models, precisions, detectors, LivePoseFormer checkpoints)."""
    try:
        from live import options as live_options
    except Exception as e:                                   # e.g. torch not installed: replay only
        return {"available": False, "reason": f"live mode needs the pipeline's packages ({type(e).__name__}: {e})"}
    f = SESSIONS / Path(session).name / "session.json"
    s = json.loads(f.read_text()) if session and f.exists() else None
    o = live_options(CONFIG["lpf_repo"], s)
    o["available"] = bool(o["models"]) and o["data"] is not False and o["lpf_repo"]
    if not o["available"]:
        o["reason"] = ("no pose model checkpoints found" if not o["models"] else
                       f"the raw frames of {s['trial']} are not in data/" if o["data"] is False else
                       "the LivePoseFormer repository (joint-angle model) was not found: start the server with --lpf-repo")
    return o


@app.websocket("/ws/live")
async def live_ws(ws: WebSocket):
    """Client sends one JSON config, then optionally {"type": "stop"}; the server sends status JSON,
    binary rows (int32 seq + float32 values) and finally {"type": "done", "session": <saved take>}."""
    await ws.accept()
    cfg = await ws.receive_json()
    folder = SESSIONS / Path(cfg.get("session", "")).name
    if not (folder / "session.json").exists():
        await ws.send_json({"type": "error", "text": "unknown session"})
        await ws.close()
        return
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    r = runner()
    task = loop.run_in_executor(None, r.run, cfg, folder, lambda m: loop.call_soon_threadsafe(q.put_nowait, m))

    async def listen():
        try:
            while True:
                if (await ws.receive_json()).get("type") == "stop":
                    r.stop()
        except (WebSocketDisconnect, RuntimeError):
            r.stop()

    listener = asyncio.create_task(listen())
    try:
        while (msg := await q.get()) is not None:
            if isinstance(msg, bytes):
                await ws.send_bytes(msg)
            else:
                await ws.send_json(msg)
    except (WebSocketDisconnect, RuntimeError):
        r.stop()
    finally:
        listener.cancel()
        try:
            await task
        except Exception:
            pass
        try:
            await ws.close()
        except RuntimeError:
            pass


@app.post("/api/export")
def export(req: dict):
    folder = SESSIONS / Path(req.get("session", "")).name
    if not (folder / "session.json").exists():
        raise HTTPException(404, "unknown session")
    data, filename = build_export(folder, req)
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


app.mount("/sessions", StaticFiles(directory=SESSIONS), name="sessions")
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{path:path}")
def spa(path: str):
    f = DIST / path
    if path and f.is_file():
        return FileResponse(f)
    if (DIST / "index.html").exists():                       # never cached: a rebuilt app is picked up on reload
        return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-cache"})
    return Response("Web app not built yet: run `npm install` and `npm run build` in live_demo/web.", 503)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--open", action="store_true", help="open the app in the default browser")
    p.add_argument("--lpf-repo", help=HELP + "; joint-angle model and LivePoseFormer checkpoints for live mode")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    CONFIG.update(lpf_repo=find_lpf_repo(a.lpf_repo), device=a.device)
    if not CONFIG["lpf_repo"]:
        print("LivePoseFormer repository not found: replay works; live mode needs --lpf-repo")
    SESSIONS.mkdir(exist_ok=True)
    if a.open:
        webbrowser.open(f"http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
