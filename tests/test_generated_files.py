"""The generated files in the repository are up to date with their sources.

Every generator runs in check mode on the repository, so a source changed
without regenerating the files derived from it fails `uv run nox`, and
with it the pre-merge and pre-push hooks.
"""

from pathlib import Path

from embodiverse.tools.generators import generate

ROOT = Path(__file__).parents[1]


def test_the_generated_files_are_up_to_date() -> None:
    stale = generate(ROOT, check=True)
    assert not stale, "\n".join(
        [
            "generated files are out of date with their sources:",
            *(f"- {path}" for path in stale),
            "Run 'uv run embodiverse generate' and commit the result.",
        ]
    )
