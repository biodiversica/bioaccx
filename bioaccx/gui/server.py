"""The HTTP layer: a small JSON API plus the single-page config editor.

The server is deliberately thin. It owns no state beyond the paths it was
started with — every document lives in the browser as YAML text and on disk as
a file — and it never imports :mod:`bioaccx.train`, so TensorFlow is never
loaded here.

Four things need Python and are therefore endpoints rather than browser code:
the form schema, the registry listing, validation through the real
``load_config``, and directory browsing (a browser cannot turn a picked folder
into an absolute path on the machine that will run the training).
"""
from __future__ import annotations

import secrets
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bioaccx import __version__
from bioaccx.gui import configio
from bioaccx.gui.schema import build_schema

WEB_DIR = Path(__file__).parent / "web"
TOKEN_COOKIE = "bioaccx_token"
TOKEN_HEADER = "x-bioaccx-token"


class DocumentBody(BaseModel):
    """A draft config: its text, plus form edits to apply to it."""

    text: str = ""
    edits: dict[str, Any] = {}


class SaveBody(BaseModel):
    path: str
    text: str


def _secret_paths(schema: dict) -> list[str]:
    out: list[str] = []
    for section in schema["sections"]:
        groups = section.get("groups", [])
        for field in section["fields"] + [f for g in groups for f in g["fields"]]:
            if field["secret"]:
                out.append(field["path"])
    return out


def _all_paths(schema: dict) -> list[str]:
    out: list[str] = []
    for section in schema["sections"]:
        for field in section["fields"]:
            out.append(field["path"])
        for group in section.get("groups", []):
            out.extend(f["path"] for f in group["fields"])
    return out


def _values(document: Any, paths: list[str]) -> dict[str, Any]:
    """Pull the value at each schema path out of a parsed document.

    Absent keys are simply missing from the result, which is how the form tells
    "left at its default" apart from "explicitly set to the default value".
    """
    out: dict[str, Any] = {}
    for dotted in paths:
        node: Any = document
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            out[dotted] = _plain(node)
    return out


def _plain(value: Any) -> Any:
    """Strip ruamel's comment-carrying wrappers so the value is JSON-safe."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def validate_text(text: str) -> dict:
    """Run the real ``load_config`` over draft YAML and report what it says.

    Validation goes through the loader the CLI uses rather than a second
    reimplementation, so a config the editor calls valid is one ``bioaccx
    train`` will accept — including the cross-field rules (grouped softmax
    without label_groups, augmentation without a source) that no per-field
    check would catch.
    """
    from bioaccx.config import load_config

    try:
        document = configio.loads(text)
    except configio.ConfigIOError as exc:
        return {"valid": False, "errors": [{"path": None, "message": str(exc)}]}

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(text)
        tmp = Path(fh.name)
    try:
        cfg = load_config(str(tmp))
    except (ValueError, TypeError, KeyError) as exc:
        return {"valid": False, "errors": [{"path": _guess_path(str(exc)),
                                            "message": str(exc)}]}
    except FileNotFoundError as exc:
        return {"valid": False, "errors": [{"path": None, "message": str(exc)}]}
    finally:
        tmp.unlink(missing_ok=True)

    return {
        "valid": True,
        "errors": [],
        "summary": {
            "foundation_model": f"{cfg.foundation_model.name} v{cfg.foundation_model.version}",
            "label_mode": cfg.dataset.label_mode,
            "classifier": cfg.training.classifier,
            "output_dir": str(cfg.output_dir),
            "model_stem": cfg.model_stem,
            "sources": len(cfg.dataset_blocks),
        },
        "document_keys": sorted(document) if isinstance(document, dict) else [],
    }


def _guess_path(message: str) -> Optional[str]:
    """Best-effort mapping of a loader error onto the field that caused it.

    The loader raises plain ValueErrors written for a terminal, so this only
    looks for a section name it recognises. A miss means the error is shown
    against the document rather than a field — never a wrong field.
    """
    for section in ("foundation_model", "dataset", "training", "output", "umap"):
        if section in message:
            return section
    return None


def create_app(*, config_path: Optional[Path] = None,
               token: Optional[str] = None) -> FastAPI:
    """Build the editor app. *config_path* is the file to open on start."""
    schema = build_schema()
    secret_paths = _secret_paths(schema)
    paths = _all_paths(schema)

    app = FastAPI(title="bioaccx", version=__version__, docs_url=None, redoc_url=None)

    def _authorized(request: Request) -> bool:
        if not token:
            return True
        given = (request.query_params.get("token")
                 or request.headers.get(TOKEN_HEADER)
                 or request.cookies.get(TOKEN_COOKIE))
        return bool(given) and secrets.compare_digest(given, token)

    @app.middleware("http")
    async def _guard(request: Request, call_next):
        if request.url.path.startswith("/health") or _authorized(request):
            response = await call_next(request)
            if token and request.query_params.get("token") == token:
                response.set_cookie(TOKEN_COOKIE, token, httponly=True,
                                    samesite="lax", path="/")
            return response
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return Response("Access token required.", status_code=401,
                        media_type="text/plain")

    @app.get("/", include_in_schema=False)
    async def index(request: Request):
        if token and request.query_params.get("token") == token:
            redirect = RedirectResponse("/", status_code=303)
            redirect.set_cookie(TOKEN_COOKIE, token, httponly=True,
                                samesite="lax", path="/")
            return redirect
        return FileResponse(WEB_DIR / "index.html",
                            headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/health", include_in_schema=False)
    async def health():
        return {"ok": True, "version": __version__}

    @app.get("/api/bootstrap")
    async def bootstrap():
        return {
            "version": __version__,
            "schema": schema,
            "secret_paths": secret_paths,
            "cwd": str(Path.cwd()),
            "open_path": str(config_path) if config_path else "",
        }

    @app.get("/api/config")
    async def read_config(path: str = Query(...)):
        """Open a config file: its text, its values, and whether it validates."""
        try:
            document = configio.load(Path(path).expanduser())
        except configio.ConfigIOError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        text = configio.dumps(document)
        return {
            "path": str(Path(path).expanduser().resolve()),
            "text": text,
            "values": _values(configio.redact(document, secret_paths), paths),
            "validation": validate_text(text),
        }

    @app.post("/api/document")
    async def edit_document(body: DocumentBody):
        """Apply form edits to draft text and hand back the exact file it becomes.

        The browser never assembles YAML itself: it sends the text it holds plus
        the fields that changed, and gets back the serialised result. What the
        preview pane shows is therefore byte-for-byte what a save would write.
        """
        try:
            document = configio.loads(body.text) if body.text.strip() else {}
        except configio.ConfigIOError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if body.edits:
            # A secret submitted as the presence marker means "unchanged".
            edits = {k: v for k, v in body.edits.items()
                     if not (k in secret_paths and v == "__SET__")}
            configio.apply_edits(document, edits)

        text = configio.dumps(document)
        return {
            "text": text,
            "values": _values(configio.redact(document, secret_paths), paths),
            "validation": validate_text(text),
        }

    @app.post("/api/save")
    async def save_config(body: SaveBody):
        target = Path(body.path).expanduser()
        try:
            document = configio.loads(body.text) if body.text.strip() else {}
            configio.save(target, document)
        except configio.ConfigIOError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=400,
                                detail=f"could not write {target}: {exc}") from exc
        return {"path": str(target.resolve()), "saved": True}

    @app.post("/api/validate")
    async def validate(body: DocumentBody):
        return validate_text(body.text)

    @app.get("/api/template")
    async def template():
        """A minimal starting document, for building a config from scratch."""
        doc = configio.loads(
            "foundation_model:\n"
            "  registry_id: '0xbb00'\n"
            "\n"
            "dataset:\n"
            "  data_dir: []\n"
            "  label_mode: subfolders\n"
            "\n"
            "training:\n"
            "  classifier: keras\n"
            "\n"
            "output:\n"
            "  output_path: ./custom_models\n"
            "  model_name: my_classifier\n"
            "  model_version: '0.1'\n"
        )
        text = configio.dumps(doc)
        return {"text": text, "values": _values(doc, paths),
                "validation": validate_text(text)}

    @app.get("/api/browse")
    async def browse(path: str = Query("")):
        """List a directory, so data_dir can be picked as an absolute path.

        A browser cannot resolve a chosen folder to a path on this machine, and
        these paths must be valid for the process that will run the training —
        so the picker walks the filesystem server-side. The server binds
        loopback unless told otherwise, and requires a token when it does not.
        """
        target = Path(path).expanduser() if path else Path.cwd()
        try:
            target = target.resolve()
            if not target.is_dir():
                target = target.parent
            entries = sorted(target.iterdir(), key=lambda p: p.name.lower())
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=400,
                                detail=f"cannot list {target}: {exc}") from exc

        def _visible(p: Path) -> bool:
            return not p.name.startswith(".")

        return {
            "path": str(target),
            "parent": str(target.parent) if target.parent != target else None,
            "dirs": [{"name": p.name, "path": str(p)}
                     for p in entries if _visible(p) and p.is_dir()],
            "files": [{"name": p.name, "path": str(p)}
                      for p in entries if _visible(p) and p.is_file()
                      and p.suffix.lower() in configio.CONFIG_SUFFIXES],
        }

    return app
