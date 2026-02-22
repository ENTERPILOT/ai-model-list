"""Schema-driven editor for models.json — FastAPI + HTMX backend."""

from __future__ import annotations

import html as html_mod
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = REPO_ROOT / "models.json"
SCHEMA_PATH = REPO_ROOT / "schema.json"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="AI Model List Editor")
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")

ENTITY_TYPES = ("providers", "models", "provider_models")
ENTITY_LABELS = {"providers": "Providers", "models": "Models", "provider_models": "Provider Models"}

# Ordered pricing fields with human-readable labels for tooltip display
PRICING_DISPLAY_ORDER = [
    ("input_per_mtok", "Input / 1M tokens"),
    ("output_per_mtok", "Output / 1M tokens"),
    ("cached_input_per_mtok", "Cached input / 1M tokens"),
    ("cache_write_per_mtok", "Cache write / 1M tokens"),
    ("reasoning_output_per_mtok", "Reasoning output / 1M tokens"),
    ("batch_input_per_mtok", "Batch input / 1M tokens"),
    ("batch_output_per_mtok", "Batch output / 1M tokens"),
    ("audio_input_per_mtok", "Audio input / 1M tokens"),
    ("audio_output_per_mtok", "Audio output / 1M tokens"),
    ("per_image", "Per generated image"),
    ("input_per_image", "Per input image"),
    ("per_second_input", "Per second of input"),
    ("per_second_output", "Per second of output"),
    ("per_character_input", "Per input character (TTS)"),
    ("per_request", "Per request"),
    ("per_page", "Per page (OCR)"),
]

# ---------------------------------------------------------------------------
# Schema cache (loaded once at startup)
# ---------------------------------------------------------------------------
_schema: dict = {}
_defs: dict = {}


def _load_schema() -> None:
    global _schema, _defs
    with open(SCHEMA_PATH) as f:
        _schema = json.load(f)
    _defs = _schema.get("$defs", {})


_load_schema()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _read_data() -> dict:
    with open(MODELS_PATH) as f:
        return json.load(f)


def _write_data(data: dict) -> None:
    data["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(MODELS_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _strip_nulls(obj: Any) -> Any:
    """Recursively remove keys with None/null values and empty dicts."""
    if isinstance(obj, dict):
        cleaned = {k: _strip_nulls(v) for k, v in obj.items() if v is not None}
        return {k: v for k, v in cleaned.items() if v != {}}
    if isinstance(obj, list):
        return [_strip_nulls(i) for i in obj]
    return obj


def _validate_data(data: dict) -> list[str]:
    """Validate data against schema + referential integrity."""
    if hasattr(jsonschema, "Draft202012Validator"):
        validator = jsonschema.Draft202012Validator(_schema)
    else:
        validator = jsonschema.Draft7Validator(_schema)
    errors = []
    for error in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in error.absolute_path) or "(root)"
        errors.append(f"{path}: {error.message}")
    # Referential integrity
    providers = set(data.get("providers", {}).keys())
    models_set = set(data.get("models", {}).keys())
    for pm_key, pm_val in data.get("provider_models", {}).items():
        parts = pm_key.split("/", 1)
        if len(parts) != 2:
            errors.append(f"{pm_key}: invalid key format (expected 'provider/model')")
            continue
        if parts[0] not in providers:
            errors.append(f"{pm_key}: provider '{parts[0]}' not found")
        model_ref = pm_val.get("model_ref")
        if model_ref and model_ref not in models_set:
            errors.append(f"{pm_key}: model_ref '{model_ref}' not found")
    return errors


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------
@dataclass
class FieldDescriptor:
    name: str
    label: str
    type: str  # "string", "integer", "number", "boolean", "array", "object"
    nullable: bool = False
    required: bool = False
    description: str = ""
    enum: list | None = None
    format: str | None = None
    default: Any = None
    const: Any = None
    children: list[FieldDescriptor] = field(default_factory=list)
    items_enum: list | None = None
    additional_properties_schema: list[FieldDescriptor] | None = None


def _resolve_ref(ref: str) -> dict:
    parts = ref.lstrip("#/").split("/")
    node = _schema
    for p in parts:
        node = node[p]
    return node


def _resolve_field(name: str, prop_schema: dict, required_set: set) -> FieldDescriptor:
    nullable = False
    resolved = prop_schema

    # Handle oneOf: [{$ref}, {type: null}]
    if "oneOf" in resolved:
        variants = resolved["oneOf"]
        non_null = [v for v in variants if v.get("type") != "null"]
        null_variants = [v for v in variants if v.get("type") == "null"]
        if null_variants and non_null:
            nullable = True
            resolved = non_null[0]
            if "$ref" in resolved:
                resolved = _resolve_ref(resolved["$ref"])
        elif non_null:
            resolved = non_null[0]
            if "$ref" in resolved:
                resolved = _resolve_ref(resolved["$ref"])

    if "$ref" in resolved:
        resolved = _resolve_ref(resolved["$ref"])

    # Handle nullable type arrays: ["string", "null"]
    raw_type = resolved.get("type", "string")
    if isinstance(raw_type, list):
        types = [t for t in raw_type if t != "null"]
        if "null" in raw_type:
            nullable = True
        raw_type = types[0] if types else "string"

    fd = FieldDescriptor(
        name=name,
        label=name.replace("_", " ").title(),
        type=raw_type,
        nullable=nullable,
        required=name in required_set,
        description=resolved.get("description", ""),
        enum=resolved.get("enum"),
        format=resolved.get("format"),
        default=resolved.get("default"),
        const=resolved.get("const"),
    )

    # Array with enum items
    if raw_type == "array" and "items" in resolved:
        items = resolved["items"]
        if "$ref" in items:
            items = _resolve_ref(items["$ref"])
        if "enum" in items:
            fd.items_enum = items["enum"]

    # Object with explicit properties -> recurse
    if raw_type == "object" and "properties" in resolved:
        req = set(resolved.get("required", []))
        for child_name, child_schema in resolved["properties"].items():
            fd.children.append(_resolve_field(child_name, child_schema, req))

    # Object with additionalProperties (dynamic key-value)
    if raw_type == "object" and "additionalProperties" in resolved and "properties" not in resolved:
        ap = resolved["additionalProperties"]
        if "$ref" in ap:
            ap = _resolve_ref(ap["$ref"])
        if ap.get("type") == "object" and "properties" in ap:
            req = set(ap.get("required", []))
            children = []
            for child_name, child_schema in ap["properties"].items():
                children.append(_resolve_field(child_name, child_schema, req))
            fd.additional_properties_schema = children

    return fd


def resolve_schema(entity_type: str) -> list[FieldDescriptor]:
    def_name = {"provider_models": "provider_model", "models": "model", "providers": "provider"}.get(entity_type)
    if not def_name:
        return []
    entity_schema = _defs.get(def_name, {})
    required_set = set(entity_schema.get("required", []))
    return [_resolve_field(n, s, required_set) for n, s in entity_schema.get("properties", {}).items()]


# ---------------------------------------------------------------------------
# Form parsing
# ---------------------------------------------------------------------------
def _coerce_value(value: str, fd: FieldDescriptor) -> Any:
    if value == "" or value is None:
        return None
    if fd.type == "integer":
        try:
            return int(value)
        except ValueError:
            return int(float(value))
    if fd.type == "number":
        return float(value)
    if fd.type == "boolean":
        return value.lower() in ("true", "on", "1", "yes")
    return value


def _set_nested(result: dict, parts: list[str], value: str,
                field_map: dict[str, FieldDescriptor]) -> None:
    if len(parts) == 1:
        fd = field_map.get(parts[0])
        if fd:
            coerced = _coerce_value(value, fd)
            if coerced is not None:
                result[parts[0]] = coerced
        elif value:
            result[parts[0]] = value
        return

    top = parts[0]
    if top not in result:
        result[top] = {}
    top_fd = field_map.get(top)
    child_map = {c.name: c for c in top_fd.children} if top_fd and top_fd.children else {}
    _set_nested(result[top], parts[1:], value, child_map)


def parse_form_data(form_items: list[tuple[str, str]], fields: list[FieldDescriptor]) -> dict:
    """Parse flat form data (dot-notation keys) into a nested dict."""
    result: dict = {}
    field_map = {f.name: f for f in fields}

    # Collect multi-values (for arrays with [])
    multi_values: dict[str, list[str]] = {}
    single_values: list[tuple[str, str]] = []

    for key, value in form_items:
        if key.startswith("_"):
            continue
        if key.endswith("[]"):
            multi_values.setdefault(key[:-2], []).append(value)
        else:
            single_values.append((key, value))

    # Process single values
    for key, value in single_values:
        parts = key.split(".")
        _set_nested(result, parts, value, field_map)

    # Process multi-values (array checkboxes)
    for key, values in multi_values.items():
        parts = key.split(".")
        if len(parts) == 1:
            result[parts[0]] = values
        elif len(parts) == 2:
            if parts[0] not in result:
                result[parts[0]] = {}
            result[parts[0]][parts[1]] = values

    # Handle boolean children in objects (capabilities etc.)
    submitted_keys = {k for k, _ in form_items}
    for fd in fields:
        if fd.children and fd.name in result and isinstance(result[fd.name], dict):
            for child in fd.children:
                if child.type == "boolean":
                    child_key = f"{fd.name}.{child.name}"
                    if child_key in submitted_keys:
                        result[fd.name][child.name] = True
                    else:
                        result[fd.name].pop(child.name, None)

    # Convert comma-separated string arrays
    for fd in fields:
        if fd.type == "array" and not fd.items_enum and fd.name in result:
            val = result[fd.name]
            if isinstance(val, str):
                result[fd.name] = [s.strip() for s in val.split(",") if s.strip()]

        if fd.children:
            obj = result.get(fd.name)
            if not isinstance(obj, dict):
                continue
            for child in fd.children:
                if child.type == "array" and not child.items_enum and child.name in obj:
                    val = obj[child.name]
                    if isinstance(val, str):
                        obj[child.name] = [s.strip() for s in val.split(",") if s.strip()]

    # Handle additionalProperties objects (parameters, rankings)
    for fd in fields:
        if fd.additional_properties_schema and fd.name in result:
            raw = result[fd.name]
            if isinstance(raw, dict):
                result[fd.name] = _rebuild_kv_object(raw, fd.additional_properties_schema)

    return result


def _rebuild_kv_object(raw: dict, child_fields: list[FieldDescriptor]) -> dict:
    """Rebuild a dynamic key-value object from form data.

    The form sends keys as `fieldname._keys` and values as `fieldname.KEYNAME.childfield`.
    """
    rebuilt: dict = {}
    for entry_key, entry_val in raw.items():
        if entry_key == "_keys":
            continue
        if isinstance(entry_val, dict):
            # Coerce child values
            coerced: dict = {}
            child_map = {c.name: c for c in child_fields}
            for ck, cv in entry_val.items():
                cfd = child_map.get(ck)
                if cfd:
                    val = _coerce_value(cv, cfd) if isinstance(cv, str) else cv
                    if val is not None:
                        coerced[ck] = val
                elif cv:
                    coerced[ck] = cv
            if coerced:
                rebuilt[entry_key] = coerced
    return rebuilt


# ---------------------------------------------------------------------------
# Template context helpers
# ---------------------------------------------------------------------------
def _base_ctx(request: Request, data: dict | None = None) -> dict:
    if data is None:
        data = _read_data()
    return {
        "request": request,
        "entity_labels": ENTITY_LABELS,
        "counts": {et: len(data.get(et, {})) for et in ENTITY_TYPES},
        "pricing_display_order": PRICING_DISPLAY_ORDER,
    }


def _get_filter_options(entity_type: str, data: dict) -> dict[str, list[str]]:
    entities = data.get(entity_type, {})
    options: dict[str, set[str]] = {}
    if entity_type == "models":
        options["owned_by"] = set()
        options["mode"] = set()
        for v in entities.values():
            if v.get("owned_by"):
                options["owned_by"].add(v["owned_by"])
            for m in v.get("modes", []):
                options["mode"].add(m)
    elif entity_type == "provider_models":
        options["provider"] = set()
        for k in entities:
            parts = k.split("/", 1)
            if parts:
                options["provider"].add(parts[0])
    elif entity_type == "providers":
        options["api_type"] = set()
        for v in entities.values():
            if v.get("api_type"):
                options["api_type"].add(v["api_type"])
    return {k: sorted(v) for k, v in options.items()}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=RedirectResponse)
async def index():
    return RedirectResponse(url="/providers", status_code=302)


@app.get("/api/keys/{section}", response_class=JSONResponse)
async def api_keys(section: str):
    data = _read_data()
    return JSONResponse(content=sorted(data.get(section, {}).keys()))


@app.get("/{entity_type}", response_class=HTMLResponse)
async def entity_list(
    request: Request,
    entity_type: str,
    search: str = "",
    filter_field: str = "",
    filter_value: str = "",
    page: int = 1,
    page_size: int = 50,
):
    if entity_type not in ENTITY_TYPES:
        return HTMLResponse("Not found", status_code=404)

    data = _read_data()
    items = list(data.get(entity_type, {}).items())

    if search:
        q = search.lower()
        items = [(k, v) for k, v in items
                 if q in k.lower() or q in v.get("display_name", "").lower()
                 or q in v.get("model_ref", "").lower()]

    if filter_field and filter_value:
        if filter_field == "provider":
            items = [(k, v) for k, v in items if k.startswith(filter_value + "/")]
        elif filter_field == "mode":
            items = [(k, v) for k, v in items if filter_value in v.get("modes", [])]
        else:
            items = [(k, v) for k, v in items
                     if str(v.get(filter_field, "")) == filter_value]

    total = len(items)
    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, total_pages))
    page_items = items[(page - 1) * page_size: page * page_size]

    ctx = {
        **_base_ctx(request, data),
        "entity_type": entity_type,
        "items": page_items,
        "search": search,
        "filter_field": filter_field,
        "filter_value": filter_value,
        "filter_options": _get_filter_options(entity_type, data),
        "page": page,
        "total_pages": total_pages,
        "total": total,
        "view": "list",
    }

    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("_list.html", ctx)
    return templates.TemplateResponse("layout.html", ctx)


@app.get("/{entity_type}/new", response_class=HTMLResponse)
async def new_entity_form(request: Request, entity_type: str):
    if entity_type not in ENTITY_TYPES:
        return HTMLResponse("Not found", status_code=404)
    data = _read_data()
    ctx = {
        **_base_ctx(request, data),
        "entity_type": entity_type,
        "fields": resolve_schema(entity_type),
        "entity_key": "",
        "entity_data": {},
        "is_new": True,
        "errors": [],
        "providers_list": sorted(data.get("providers", {}).keys()),
        "models_list": sorted(data.get("models", {}).keys()),
        "view": "form",
    }
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("_form.html", ctx)
    return templates.TemplateResponse("layout.html", ctx)


@app.get("/{entity_type}/{key:path}/edit", response_class=HTMLResponse)
async def edit_entity_form(request: Request, entity_type: str, key: str):
    if entity_type not in ENTITY_TYPES:
        return HTMLResponse("Not found", status_code=404)
    data = _read_data()
    entities = data.get(entity_type, {})
    if key not in entities:
        return HTMLResponse("Entity not found", status_code=404)

    ctx = {
        **_base_ctx(request, data),
        "entity_type": entity_type,
        "fields": resolve_schema(entity_type),
        "entity_key": key,
        "entity_data": entities[key],
        "is_new": False,
        "errors": [],
        "providers_list": sorted(data.get("providers", {}).keys()),
        "models_list": sorted(data.get("models", {}).keys()),
        "view": "form",
    }
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("_form.html", ctx)
    return templates.TemplateResponse("layout.html", ctx)


@app.get("/{entity_type}/{key:path}/json", response_class=HTMLResponse)
async def json_preview(request: Request, entity_type: str, key: str):
    data = _read_data()
    entities = data.get(entity_type, {})
    if key not in entities:
        return HTMLResponse("Not found", status_code=404)
    formatted = html_mod.escape(json.dumps(entities[key], indent=2, ensure_ascii=False))
    safe_key = html_mod.escape(key)
    return HTMLResponse(
        f'<dialog open id="json-dialog">'
        f'<article><header><button aria-label="Close" rel="prev" '
        f'onclick="this.closest(\'dialog\').remove()"></button>'
        f'<h3>{safe_key}</h3></header>'
        f'<pre><code>{formatted}</code></pre></article></dialog>'
    )


@app.post("/{entity_type}/new", response_class=HTMLResponse)
async def create_entity(request: Request, entity_type: str):
    if entity_type not in ENTITY_TYPES:
        return HTMLResponse("Not found", status_code=404)

    form = await request.form()
    form_items = list(form.multi_items())
    form_dict = dict(form)

    entity_key = form_dict.get("_key", "").strip()
    if not entity_key:
        return await _form_error(request, entity_type, "", {}, True, ["Entity key is required"])

    if entity_type == "provider_models" and "/" not in entity_key:
        provider = form_dict.get("_provider", "")
        entity_key = f"{provider}/{entity_key}" if provider else entity_key

    data = _read_data()
    if entity_key in data.get(entity_type, {}):
        return await _form_error(request, entity_type, entity_key, {}, True,
                                 [f"Key '{entity_key}' already exists"])

    fields = resolve_schema(entity_type)
    entity_data = parse_form_data(form_items, fields)
    entity_data = _strip_nulls(entity_data)

    data[entity_type][entity_key] = entity_data
    errors = _validate_data(data)
    if errors:
        return await _form_error(request, entity_type, entity_key, entity_data, True, errors)

    _write_data(data)
    return _redirect(request, f"/{entity_type}")


@app.post("/{entity_type}/{key:path}/edit", response_class=HTMLResponse)
async def update_entity(request: Request, entity_type: str, key: str):
    if entity_type not in ENTITY_TYPES:
        return HTMLResponse("Not found", status_code=404)
    data = _read_data()
    if key not in data.get(entity_type, {}):
        return HTMLResponse("Entity not found", status_code=404)

    form = await request.form()
    form_items = list(form.multi_items())

    fields = resolve_schema(entity_type)
    entity_data = parse_form_data(form_items, fields)
    entity_data = _strip_nulls(entity_data)

    data[entity_type][key] = entity_data
    errors = _validate_data(data)
    if errors:
        return await _form_error(request, entity_type, key, entity_data, False, errors)

    _write_data(data)
    return _redirect(request, f"/{entity_type}")


@app.delete("/{entity_type}/{key:path}", response_class=HTMLResponse)
async def delete_entity(request: Request, entity_type: str, key: str):
    data = _read_data()
    entities = data.get(entity_type, {})
    if key not in entities:
        return HTMLResponse("Not found", status_code=404)
    del entities[key]
    _write_data(data)
    return _redirect(request, f"/{entity_type}")


@app.post("/validate", response_class=HTMLResponse)
async def validate_all(request: Request):
    data = _read_data()
    errors = _validate_data(data)
    counts = {et: len(data.get(et, {})) for et in ENTITY_TYPES}
    if errors:
        items = "".join(f"<li><code>{html_mod.escape(e)}</code></li>" for e in errors)
        body = f"<ul>{items}</ul>"
    else:
        body = (f'<p style="color:var(--pico-ins-color);">PASSED: '
                f'{counts["providers"]} providers, {counts["models"]} models, '
                f'{counts["provider_models"]} provider_models</p>')
    return HTMLResponse(f'<article id="validation-results"><header>'
                        f'<strong>Validation Results</strong></header>{body}</article>')


def _redirect(request: Request, url: str):
    if request.headers.get("HX-Request"):
        return HTMLResponse(content="", headers={"HX-Redirect": url})
    return RedirectResponse(url=url, status_code=303)


async def _form_error(request: Request, entity_type: str, key: str,
                      entity_data: dict, is_new: bool, errors: list[str]) -> HTMLResponse:
    data = _read_data()
    ctx = {
        **_base_ctx(request, data),
        "entity_type": entity_type,
        "fields": resolve_schema(entity_type),
        "entity_key": key,
        "entity_data": entity_data,
        "is_new": is_new,
        "errors": errors,
        "providers_list": sorted(data.get("providers", {}).keys()),
        "models_list": sorted(data.get("models", {}).keys()),
        "view": "form",
    }
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("_form.html", ctx)
    return templates.TemplateResponse("layout.html", ctx)
