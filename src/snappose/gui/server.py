"""Local web server for the GUI (stdlib only), same security model as the ctrack GUI: binds to 127.0.0.1,
checks the Host header (DNS rebinding) and requires `X-Requested-With: snappose` on every POST."""
from __future__ import annotations

import argparse
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

from .engine import Engine

MAX_BODY = 200 * 1024 * 1024        # CAD files can be large
LAYERS = {"prior", "result", "gt", "edges"}


class Handler(BaseHTTPRequestHandler):
    engine: Engine
    port: int
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args) -> None:
        pass

    def _host_ok(self) -> bool:
        return self.headers.get("Host", "") in (f"127.0.0.1:{self.port}", f"localhost:{self.port}")

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _error(self, code: int, message: str) -> None:
        self.close_connection = True
        self._send(code, json.dumps({"error": message}).encode(), "application/json", {"Connection": "close"})

    def do_GET(self) -> None:
        if not self._host_ok():
            return self._error(403, "bad host")
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                html = resources.files("snappose.gui").joinpath("static/index.html").read_bytes()
                return self._send(200, html, "text/html; charset=utf-8")
            if url.path == "/api/status":
                return self._json(self.engine.snapshot())
            if url.path == "/api/image.jpg":
                view = q.get("view", ["rgb"])[0]
                layers = set(q.get("layers", [""])[0].split(",")) & LAYERS
                jpg = self.engine.image_jpeg(view if view in ("rgb", "depth") else "rgb", layers)
                return self._send(200, jpg, "image/jpeg") if jpg else self._error(404, "no frame")
            if url.path == "/api/result.json":
                r = self.engine.result
                if not r:
                    return self._error(404, "no result")
                return self._send(200, json.dumps(r, indent=2).encode(), "application/json",
                                  {"Content-Disposition": 'attachment; filename="snappose_result.json"'})
            m = re.fullmatch(r"/api/bop/(\w+)/objects", url.path)
            if m:
                return self._json({"objects": self.engine.bop_objects(m.group(1))})
            return self._error(404, "not found")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            self._error(500, f"{type(e).__name__}: {e}")

    def do_POST(self) -> None:
        if not self._host_ok():
            return self._error(403, "bad host")
        if self.headers.get("X-Requested-With") != "snappose":
            return self._error(403, "missing X-Requested-With header")
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        e = self.engine
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._error(413, "Datei zu groß")
            raw = self.rfile.read(n) if n else b""       # always consume the body first
            ctype = self.headers.get("Content-Type", "")
            d = json.loads(raw) if raw and "json" in ctype else {}
            p = url.path
            if p == "/api/object/select":
                e.select_object(str(d.get("id")))
            elif p == "/api/object/onboard":
                e.onboard_upload(q.get("name", ""), q.get("ext", ""), raw, float(q.get("scale", 1.0)))
            elif p == "/api/object/delete":
                e.delete_object(str(d.get("id")))
            elif p == "/api/frame/demo":
                e.demo_frame(d.get("seed"))
            elif p == "/api/frame/bop":
                oid = d.get("obj")
                e.bop_frame(str(d.get("dataset")), int(oid) if oid not in (None, "") else None)
            elif p == "/api/frame/upload":
                e.upload_part(q.get("kind", ""), raw)
            elif p == "/api/frame/camera":
                e.set_camera(d)
            elif p == "/api/prior/reroll":
                e.reroll_prior()
            elif p == "/api/prior/set":
                e.set_prior(d.get("t", []), d.get("euler_deg", []))
            elif p == "/api/prior/guess":
                e.guess_prior()
            elif p == "/api/settings":
                e.update_settings(d)
            elif p == "/api/match":
                e.start_match()
            elif p == "/api/bench":
                e.start_bench(int(d.get("n", 20)))
            else:
                return self._error(404, "not found")
            return self._json(e.snapshot())
        except (ValueError, KeyError, LookupError, NotImplementedError) as ex:
            self._error(400, str(ex))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as ex:
            self._error(500, f"{type(ex).__name__}: {ex}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="snappose-gui", description="SnapPose Weboberfläche (nur lokal)")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--store", default="objects")
    ap.add_argument("--bop", default="data/bop", help="Ordner mit BOP-Datensätzen (optional)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    Handler.engine = Engine(a.store, a.bop)
    Handler.port = a.port
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    except OSError:
        print(f"Port {a.port} ist belegt – läuft SnapPose schon? Dann einfach http://127.0.0.1:{a.port} öffnen, "
              f"sonst anderen Port wählen: snappose gui --port {a.port + 1}")
        return 1
    srv.daemon_threads = True
    url = f"http://127.0.0.1:{a.port}"
    print(f"SnapPose läuft auf {url}  (Strg+C beendet)")
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
