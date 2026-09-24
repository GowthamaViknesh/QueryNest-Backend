"""The FastAPI app (M4). Run:  uv run api   ->  http://localhost:8000  (docs at /docs)

In production the same server also serves the built React app (Agent-Frontend/dist) and the
embed script (/embed.js), so the whole product is one process behind one URL.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from querynest.api import admin, routes
from querynest.config import settings
from querynest.logging_setup import setup_logging

log = logging.getLogger("querynest.api")

EMBED_JS = """(function () {
  // QueryNest embed: <script src="https://YOUR-SERVER/embed.js" async></script>
  var origin = new URL(document.currentScript.src).origin;
  var button = document.createElement("button");
  button.textContent = "Ask QueryNest";
  button.style.cssText = "position:fixed;right:20px;bottom:20px;z-index:2147483646;padding:12px 18px;" +
    "border:0;border-radius:24px;background:#1F4E79;color:#fff;font:600 14px system-ui;cursor:pointer;" +
    "box-shadow:0 4px 14px rgba(0,0,0,.25)";
  var frame = document.createElement("iframe");
  frame.src = origin + "/?embed=1";
  frame.title = "QueryNest";
  frame.style.cssText = "position:fixed;right:20px;bottom:76px;z-index:2147483647;width:min(440px,calc(100vw - 40px));" +
    "height:min(640px,calc(100vh - 110px));border:1px solid #d0d7de;border-radius:12px;display:none;" +
    "box-shadow:0 8px 30px rgba(0,0,0,.25);background:#fff";
  button.onclick = function () { frame.style.display = frame.style.display === "none" ? "block" : "none"; };
  document.body.appendChild(frame);
  document.body.appendChild(button);
})();
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log.info("QueryNest API starting")
    yield
    from querynest.db import close_pool
    close_pool()


def create_app() -> FastAPI:
    app = FastAPI(title="QueryNest API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["Authorization", "Content-Type"])

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        # The app itself may be framed (embed.js); everything else keeps a strict policy
        if request.url.path.startswith("/api"):
            response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse({"detail": "Internal server error"}, status_code=500)  # no stack traces to clients

    @app.get("/api/health")
    def health():
        from sqlalchemy import text

        from querynest.appdb import engine
        with engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}

    @app.get("/embed.js", include_in_schema=False)
    def embed_js():
        return Response(EMBED_JS, media_type="application/javascript")

    app.include_router(routes.router)
    app.include_router(admin.router)

    dist = settings.frontend_dist
    if (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            file = (dist / path).resolve()
            if path and file.is_file() and dist.resolve() in file.parents:
                return FileResponse(file)
            return FileResponse(dist / "index.html")  # client-side app handles the route
    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("querynest.api:app", host="127.0.0.1", port=8000)
