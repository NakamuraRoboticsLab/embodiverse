"""Tests of the run manifest's schema."""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from embodiverse.contracts.hashing import content_hash
from embodiverse.contracts.manifest import RunManifest

HASH = "sha256:" + "0123456789abcdef" * 4
COMMIT = "0123456789abcdef0123456789abcdef01234567"
STARTED = datetime(2026, 10, 5, 9, 30, 15, 123456, tzinfo=timezone.utc)


def manifest_data(**changes: Any) -> dict[str, Any]:
    """Return the data of a valid manifest, with `changes`."""
    data: dict[str, Any] = {
        "schema_version": 1,
        "platform_version": "0.0.1",
        "robot": "testbot",
        "description_hash": HASH,
        "adapter": "mujoco",
        "policy_sha256": None,
        "descriptor_hash": HASH,
        "python_version": "3.10.12",
        "git_sha": COMMIT,
        "git_dirty": False,
        "started_utc": STARTED,
        "host": "lab-pc",
    }
    data.update(changes)
    return data


def test_a_manifest_holds_what_it_is_given() -> None:
    manifest = RunManifest.model_validate(manifest_data(policy_sha256=HASH))
    assert manifest.robot == "testbot"
    assert manifest.policy_sha256 == HASH
    assert manifest.started_utc == STARTED
    assert manifest.git_dirty is False


def test_a_manifest_round_trips_through_json() -> None:
    manifest = RunManifest.model_validate(manifest_data())
    text = manifest.model_dump_json()
    assert '"started_utc":"2026-10-05T09:30:15.123456Z"' in text
    again = RunManifest.model_validate_json(text)
    assert again == manifest
    assert content_hash(again) == content_hash(manifest)


def test_a_manifest_is_immutable() -> None:
    manifest = RunManifest.model_validate(manifest_data())
    with pytest.raises(ValidationError, match="frozen"):
        manifest.robot = "g1_29dof"  # pyright: ignore[reportAttributeAccessIssue]


def test_every_field_is_required() -> None:
    for field in manifest_data():
        data = manifest_data()
        del data[field]
        with pytest.raises(ValidationError, match=f"{field}\n  Field required"):
            RunManifest.model_validate(data)


def test_without_a_repository_the_commit_is_none() -> None:
    manifest = RunManifest.model_validate(manifest_data(git_sha=None, git_dirty=None))
    assert manifest.git_sha is None and manifest.git_dirty is None


@pytest.mark.parametrize(
    "started",
    [
        pytest.param(STARTED, id="datetime.timezone.utc"),
        pytest.param(STARTED.replace(tzinfo=timezone(timedelta(0))), id="offset 0"),
        pytest.param("2026-10-05T09:30:15.123456Z", id="JSON with Z"),
        pytest.param("2026-10-05T09:30:15.123456+00:00", id="JSON with +00:00"),
    ],
)
def test_started_utc_accepts_any_form_of_utc(started: datetime | str) -> None:
    manifest = RunManifest.model_validate(manifest_data(started_utc=started))
    assert manifest.started_utc == STARTED


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        pytest.param(
            {"started_utc": STARTED.replace(tzinfo=None)},
            "started_utc\n  Input should have timezone info",
            id="naive started_utc",
        ),
        pytest.param(
            {"started_utc": STARTED.astimezone(timezone(timedelta(hours=2)))},
            "started_utc\n  Value error, must be in UTC, but its offset from UTC is"
            " 2:00:00",
            id="started_utc in another time zone",
        ),
        pytest.param(
            {"seed": 3},
            "seed\n  Extra inputs are not permitted",
            id="unknown key",
        ),
        pytest.param(
            {"schema_version": 2},
            "schema_version\n  Input should be 1",
            id="other schema version",
        ),
        pytest.param(
            {"description_hash": "sha256:" + "0123456789ABCDEF" * 4},
            "description_hash\n  String should match pattern",
            id="uppercase hash",
        ),
        pytest.param(
            {"descriptor_hash": "0123456789abcdef" * 4},
            "descriptor_hash\n  String should match pattern",
            id="hash without its algorithm",
        ),
        pytest.param(
            {"policy_sha256": "sha256:0123"},
            "policy_sha256\n  String should match pattern",
            id="short policy hash",
        ),
        pytest.param(
            {"git_sha": COMMIT[:12]},
            "git_sha\n  String should match pattern",
            id="abbreviated commit",
        ),
        pytest.param(
            {"git_sha": None, "git_dirty": True},
            "Value error, git_dirty must be None when git_sha is None",
            id="dirty without a commit",
        ),
        pytest.param(
            {"robot": ""},
            "robot\n  String should have at least 1 character",
            id="no robot name",
        ),
        pytest.param(
            {"adapter": ""},
            "adapter\n  String should have at least 1 character",
            id="no adapter",
        ),
    ],
)
def test_rejects(changes: dict[str, Any], problem: str) -> None:
    with pytest.raises(ValidationError) as error:
        RunManifest.model_validate(manifest_data(**changes))
    assert problem in str(error.value)


def test_accepts_a_commit_of_a_sha_256_repository() -> None:
    commit = "0123456789abcdef" * 4
    assert RunManifest.model_validate(manifest_data(git_sha=commit)).git_sha == commit
