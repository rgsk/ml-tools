# any-check's decisions, on basedpyright output written out by hand: which
# findings it reads, and which partially unknown names still complete in the
# editor and so aren't reported
import json
from pathlib import Path

import pytest

from ml_tools.any_check import Cell, findings, notebook, per_file, still_completes, untyped

# what `byname tool basedpyright` prints with reportUnknownVariableType on: the
# finding, then the full type on the line under it
PARTIAL = """\
  src/x.pyn:256:5 - error: Type of "norms" is partially unknown
    Type of "norms" is "list[Unknown | Tensor]" (reportUnknownVariableType)
"""
ANY = """\
  src/x.pyn:12:9 - error: Type of "logits" is Any (reportAny)
"""


def test_a_partial_finding_carries_its_full_type():
    (found,) = findings(PARTIAL)
    assert (found.line, found.col, found.name) == (256, 5, "norms")
    assert found.kind == "partially unknown"
    assert found.type_text == "list[Unknown | Tensor]"


def test_an_any_finding_has_no_full_type():
    (found,) = findings(ANY)
    assert (found.name, found.kind, found.type_text) == ("logits", "Any", "")


@pytest.mark.parametrize(
    "type_text",
    [
        "list[Unknown | Tensor]",  # torch's Tensor.norm returns an untyped call
        "Unknown | Tensor",
        "list[Tensor | Unknown]",
        "dict[str, Unknown | int]",
    ],
)
def test_a_union_with_a_known_type_still_completes(type_text: str):
    assert still_completes(type_text)


@pytest.mark.parametrize(
    "type_text",
    [
        "list[Unknown]",  # xs = [] then untyped appends: elements show white
        "dict[str, Unknown]",
        "Unknown | None",  # None has nothing to complete
        "",  # no full type printed, so nothing to judge by
    ],
)
def test_a_bare_unknown_is_still_reported(type_text: str):
    assert not still_completes(type_text)



# one `byname tool basedpyright` run over two files: paths relative to the
# project, a finding with its full type on the line under it, indented with
# non-breaking spaces, then the run's totals
ROOT = Path("/p")
RUN = """\
src/model.pyn
  src/model.pyn:256:5 - error: Type of "norms" is partially unknown
  \xa0\xa0Type of "norms" is "list[Unknown | Tensor]" (reportUnknownVariableType)
src/train.pyn
  src/train.pyn:12:9 - error: Type of "logits" is Any (reportAny)
2 errors, 0 warnings, 0 notes
"""


def test_one_runs_output_splits_per_file_with_each_findings_full_type():
    files = per_file(RUN, ROOT)
    assert list(files) == [ROOT / "src/model.pyn", ROOT / "src/train.pyn"]
    (norms,) = findings(files[ROOT / "src/model.pyn"])
    assert norms.type_text == "list[Unknown | Tensor]"
    (logits,) = findings(files[ROOT / "src/train.pyn"])
    assert logits.name == "logits"


# an attribute finding is reported where its expression starts. It's kept when
# the root is declared, so the Any comes from the expression, and dropped when
# the root is an undeclared Any name, which is reported itself
SOURCE = """\
def show(axes: list, logits):
    axes[0].imshow
    logits.shape
"""
ATTRIBUTES = """\
  x.py:1:22 - error: Type of "logits" is Any (reportAny)
  x.py:2:5 - error: Type of "imshow" is Any (reportAny)
  x.py:3:5 - error: Type of "shape" is Any (reportAny)
"""


def test_an_attribute_is_reported_only_when_its_root_is_declared(tmp_path: Path):
    path = tmp_path / "x.py"
    path.write_text(SOURCE)
    reports = untyped(path, findings(ATTRIBUTES))
    assert reports == [f"{path}:1:22 - logits is Any", f"{path}:2:5 - axes[0].imshow is Any"]


def test_an_unannotated_parameter_is_a_finding():
    (found,) = findings(
        '  x.py:116:16 - error: Type of parameter "event" is unknown (reportUnknownParameterType)\n'
    )
    assert (found.name, found.kind) == ("event", "unknown")


def test_a_name_the_file_never_writes_is_bynames_and_not_reported(tmp_path: Path):
    # `(state=) = load()` translates to `_ds = load(); state = _ds.state`
    path = tmp_path / "x.py"
    path.write_text("state = load()\n")
    found = findings('  x.py:1:1 - error: Type of "_ds" is partially unknown (reportUnknownVariableType)\n')
    assert untyped(path, found) == []


# a byname notebook: a markdown cell, a cell of magics, then code
NOTEBOOK = {
    "cells": [
        {"cell_type": "markdown", "source": ["# title"]},
        {"cell_type": "code", "source": ["%load_ext byname\n", "%autoreload 2"]},
        {"cell_type": "code", "source": ["import numpy as np\n", "\n", "def f(x):\n", "    return x"]},
    ]
}


def test_a_notebook_is_its_code_cells_joined_with_magics_blanked(tmp_path: Path):
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(NOTEBOOK))
    text, cells, uses_byname = notebook(path)
    assert text == "\n\nimport numpy as np\n\ndef f(x):\n    return x"  # no line moves
    assert cells == [Cell(number=2, first=1), Cell(number=3, first=3)]
    assert uses_byname


def test_a_notebook_finding_is_reported_by_cell(tmp_path: Path):
    # without byname: its translation needs byname, which this environment doesn't have
    magics = {"cell_type": "code", "source": ["%autoreload 2\n", "x = 1"]}
    plain = {"cells": [NOTEBOOK["cells"][0], magics, NOTEBOOK["cells"][2]]}
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(plain))
    found = findings('  nb.py:5:7 - error: Type of parameter "x" is unknown (reportUnknownParameterType)\n')
    assert untyped(path, found) == [f"{path}:cell 3:3:7 - x is unknown"]
