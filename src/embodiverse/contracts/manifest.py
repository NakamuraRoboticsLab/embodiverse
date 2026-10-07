"""Run manifests: everything that went into one run, so its results can be traced.

A result is only useful if it can be traced to exactly what produced it,
and a failure on a robot is only debuggable if it is known what was
running. A manifest identifies everything that went into one run of a
policy: the platform's version, the robot description and the adapter,
the policy file and its descriptor, the Python version, the commit of the
code and whether it had uncommitted changes, when the run started and on
which machine. Descriptions and descriptors are identified by their
content hash, and policy files by the SHA-256 of their bytes, so two runs
with the same hashes ran the same robot description, policy and
descriptor.

The manifest is the platform's record of provenance. Every recording of a
run carries one, and a policy descriptor's `provenance` field, `external`
for a policy trained outside the platform, can later hold one for a policy
trained on it.
`embodiverse.runtime.recording.collect_manifest` fills one in for a run.
docs/policies.md describes the fields.
"""

from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

__all__ = ["RunManifest"]

_Hash = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
"""`sha256:` and 64 lowercase hex digits."""

_GitSha = Annotated[str, StringConstraints(pattern=r"^([0-9a-f]{40}|[0-9a-f]{64})$")]
"""A full commit name: 40 lowercase hex digits, or 64 in a SHA-256 repository."""


class RunManifest(BaseModel):
    """What went into one run: the platform, robot, policy, code and machine.

    Every field is required, so a manifest always states all of them;
    a field that does not apply, such as the policy file of a policy built
    into the platform, is None. Raises pydantic's ValidationError, a
    ValueError, for an unknown field, a malformed hash or commit name, a
    `started_utc` that is not timezone-aware UTC, or `git_dirty` set
    without `git_sha`.
    """

    # Manifests are read-only, and misspelled keys are errors rather than
    # silently ignored.
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = Field(description="Version of this schema.")
    platform_version: str = Field(
        description="The version of Embodiverse that ran, e.g. `0.1.0`."
    )
    robot: str = Field(
        min_length=1, description="The robot description's `name`, e.g. `testbot`."
    )
    description_hash: _Hash = Field(
        description="`content_hash` of the robot description, as `sha256:` and 64 "
        "lowercase hex digits."
    )
    adapter: str = Field(
        min_length=1,
        description="The adapter's registered name, e.g. `mujoco`, or, for an "
        "adapter that was not found through the registry, its class as "
        "`module:QualifiedName`.",
    )
    policy_sha256: _Hash | None = Field(
        description="SHA-256 of the policy file's bytes, as the descriptor records "
        "it; None for a policy built into the platform, which has no file."
    )
    descriptor_hash: _Hash = Field(
        description="`content_hash` of the policy descriptor, as `sha256:` and 64 "
        "lowercase hex digits."
    )
    python_version: str = Field(
        description="The version of the Python that ran, e.g. `3.10.12`."
    )
    git_sha: _GitSha | None = Field(
        description="The commit checked out in the repository of the directory the "
        "run started in; None if there is no repository, no commit or no git."
    )
    git_dirty: bool | None = Field(
        description="Whether that repository had changes not committed, untracked "
        "files included; None if unknown, as whenever `git_sha` is None."
    )
    started_utc: AwareDatetime = Field(
        description="When the run started, in UTC, to the microsecond."
    )
    host: str = Field(description="The name of the machine that ran it.")

    @field_validator("started_utc")
    @classmethod
    def _check_utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if offset != timedelta(0):
            raise ValueError(f"must be in UTC, but its offset from UTC is {offset}")
        return value

    @model_validator(mode="after")
    def _check_git(self) -> "RunManifest":
        if self.git_sha is None and self.git_dirty is not None:
            raise ValueError("git_dirty must be None when git_sha is None")
        return self
