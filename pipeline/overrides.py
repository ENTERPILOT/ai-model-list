"""Apply reviewed pricing corrections on top of resolved source evidence.

Upstream price feeds occasionally publish a stale or plainly wrong rate for a
model, and the resolver has no way to tell a wrong authoritative number from a
right one — the highest-ranked source simply wins. When that happens the
registry ships the bad price twice a day until the upstream catches up.

``registry/curated/pricing_overrides.json`` is the reviewed escape hatch: a
narrow, human-audited assertion of what a vendor's published page actually
says, applied after resolution. Each entry names the fields it corrects and
nothing else, so upstream stays authoritative for every price component the
override does not mention.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def apply_pricing_overrides(
    registry: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Patch curated pricing corrections into ``registry`` in place.

    Returns one finding per override entry so the build report can show what was
    corrected — and, just as importantly, which overrides have gone stale.
    """
    findings: list[dict[str, Any]] = []
    for entry in (overrides or {}).get("overrides", []):
        if not isinstance(entry, Mapping):
            continue
        finding = _apply_override_entry(registry, entry)
        if finding is not None:
            findings.append(finding)
    return findings


def _apply_override_entry(
    registry: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any] | None:
    model_key = entry.get("model")
    pricing = entry.get("pricing")
    if not isinstance(model_key, str) or not model_key:
        return None
    if not isinstance(pricing, Mapping) or not pricing:
        return None

    source_url = entry.get("source_url") if isinstance(entry.get("source_url"), str) else None
    targets = _override_targets(registry, model_key, entry.get("providers"))

    changed_targets: list[str] = []
    changed_fields: set[str] = set()
    for target_key, target in targets:
        fields = _patch_pricing(target, pricing, source_url)
        if fields:
            changed_targets.append(target_key)
            changed_fields.update(fields)

    if not targets:
        status = "unmatched"
    elif changed_targets:
        status = "applied"
    else:
        status = "noop"

    finding: dict[str, Any] = {
        "model": model_key,
        "status": status,
        "changed_targets": sorted(changed_targets),
        "changed_fields": sorted(changed_fields),
    }
    if source_url:
        finding["source_url"] = source_url
    if isinstance(entry.get("verified_on"), str):
        finding["verified_on"] = entry["verified_on"]
    return finding


def _override_targets(
    registry: Mapping[str, Any],
    model_key: str,
    providers: Any,
) -> list[tuple[str, dict[str, Any]]]:
    """Collect the canonical model and every named provider's serving record.

    Provider slugs are opt-in: a vendor's own published rate says nothing about
    what a reseller charges for the same model, so an override never reaches a
    provider the entry did not list.
    """
    targets: list[tuple[str, dict[str, Any]]] = []

    model = registry.get("models", {}).get(model_key)
    if isinstance(model, dict):
        targets.append((model_key, model))

    provider_slugs = {slug for slug in _as_str_list(providers)}
    if not provider_slugs:
        return targets

    for provider_model_key, provider_model in registry.get("provider_models", {}).items():
        if not isinstance(provider_model, dict):
            continue
        if provider_model.get("model_ref") != model_key:
            continue
        if provider_model_key.split("/", 1)[0] not in provider_slugs:
            continue
        targets.append((provider_model_key, provider_model))

    return targets


def _patch_pricing(
    target: dict[str, Any],
    pricing: Mapping[str, Any],
    source_url: str | None,
) -> list[str]:
    current = target.get("pricing")
    if not isinstance(current, dict):
        current = {}
        target["pricing"] = current

    changed = [key for key, value in pricing.items() if current.get(key) != value]
    if not changed:
        return []

    current.update(pricing)
    current.setdefault("currency", "USD")

    if source_url and "pricing_source_url" in target:
        target["pricing_source_url"] = source_url
    if source_url and isinstance(target.get("source_urls"), list):
        if source_url not in target["source_urls"]:
            target["source_urls"] = sorted({*target["source_urls"], source_url})

    return changed


def _as_str_list(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str) and item]
    return []
