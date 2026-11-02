"""Execute and type-check each documentation example independently."""

import contextlib
import io
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DOCUMENTS = (ROOT / "docs" / "math.md", ROOT / "docs" / "usage.md")
BLOCK = re.compile(r"^```(\w+)\n(.*?)^```$", re.DOTALL | re.MULTILINE)


@dataclass(frozen=True)
class Example:
    name: str
    source: str
    output: str


def examples() -> list[Example]:
    result = []
    for document in DOCUMENTS:
        text = document.read_text()
        blocks: list[tuple[str, str]] = BLOCK.findall(text)
        assert text.count("```") == 2 * len(blocks)
        assert {language for language, _ in blocks} <= {"python", "text"}
        for index, (language, source) in enumerate(blocks):
            if language == "text":
                assert index > 0
                assert blocks[index - 1][0] == "python"
                continue
            following = blocks[index + 1] if index + 1 < len(blocks) else ("", "")
            result.append(
                Example(
                    f"{document.stem}_{index}",
                    source,
                    following[1] if following[0] == "text" else "",
                )
            )
    return result


@pytest.mark.parametrize("example", examples(), ids=lambda example: example.name)
def test_example(example: Example) -> None:
    namespace: dict[str, object] = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example.source, example.name, "exec"), namespace)
    assert output.getvalue() == example.output


def test_example_types(tmp_path: Path) -> None:
    paths = []
    for example in examples():
        path = tmp_path / f"{example.name}.py"
        path.write_text(example.source)
        paths.append(str(path))
    checked = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyrefly",
            "check",
            "--config",
            str(ROOT / "pyproject.toml"),
            *paths,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
