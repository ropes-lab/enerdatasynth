"""Every example is laid out per resource and ends with reproduction instructions."""
import ast
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SCRIPTS = sorted(EXAMPLES.glob("*/*.py"))


def test_examples_are_nested_per_resource_like_data():
    assert (EXAMPLES / "pjm_load").is_dir() and (EXAMPLES / "wind").is_dir()
    assert not list(EXAMPLES.glob("*.py")), "scripts belong in examples/<resource>/"
    for resource in ("pjm_load", "wind"):
        assert (EXAMPLES / resource / "README.md").exists()


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_example_docstring_ends_with_reproduce_block(script):
    doc = ast.get_docstring(ast.parse(script.read_text()), clean=True)
    assert doc and "Reproduce:" in doc, f"{script} needs a 'Reproduce:' block"
    reproduce = doc.split("Reproduce:", 1)[1]
    assert "python" in reproduce or "enerdatasynth" in reproduce
