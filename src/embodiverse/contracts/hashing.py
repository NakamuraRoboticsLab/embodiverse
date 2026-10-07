"""Content hashes: identify a validated model by its contents, not its formatting.

`content_hash` hashes a model's canonical JSON form, so files that validate
to the same model get the same hash whatever their formatting, comments,
key order or location, and whether a number is written `100` or `100.0`.

Fields equal to their defaults are left out of the canonical form. A
schema can gain optional fields with defaults without a version bump, and
such an addition must not change the hash of existing models, or every
policy trained against them would stop matching. It also means an omitted
default and the same default written out hash identically.
"""

import hashlib
import json

from pydantic import BaseModel

__all__ = ["canonical_json", "content_hash"]


def canonical_json(model: BaseModel) -> str:
    """Return the canonical JSON form of `model`: the text `content_hash` hashes.

    Fields equal to their defaults are left out, keys are sorted, and there
    is no whitespace between tokens. Raises ValueError if a float is not
    finite.
    """
    data = model.model_dump(mode="json", exclude_defaults=True)
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def content_hash(model: BaseModel) -> str:
    """Return the SHA-256 hash of `model`'s canonical JSON, as `sha256:<hex>`."""
    digest = hashlib.sha256(canonical_json(model).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
