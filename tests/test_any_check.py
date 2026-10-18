# any-check's decisions, on basedpyright output written out by hand: which
# findings it reads, and which partially unknown names still complete in the
# editor and so aren't reported
from pathlib import Path

import pytest

from ml_tools.any_check import Annotation, findings, probed, reveals, still_completes, widened

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


# --redundant: which annotations get probed, how the probes' reveals are read
# back, and how a literal's revealed type becomes a variable's type
PROBED_SOURCE = """\
from typing import Final, NamedTuple


class Cfg(NamedTuple):
    lr: float = 1e-3  # a field, not a variable


def train(cfg: Cfg):
    steps: int = 10
    LIMIT: Final[int] = 5
    row: Cfg = _rec_lr(lr=1.0)
    sizes: list[int] = [1, 2]  # checks each element, so kept
    n: int
    out: float = 1.0
    out = cfg.lr * out  # rebound: the annotation types this line too
"""


def probed_source(tmp_path: Path) -> tuple[str, list[Annotation]]:
    path = tmp_path / "train.py"
    path.write_text(PROBED_SOURCE)
    return probed(path)


def test_only_a_plain_annotated_assignment_is_probed(tmp_path: Path):
    _source, annotations = probed_source(tmp_path)
    # the class field, Final, the record literal, the container literal, the
    # bare `n: int` and the rebound `out` are left alone
    assert annotations == [Annotation(line=9, col=4, name="steps", annotation="int")]


def test_the_probes_go_before_the_assignment(tmp_path: Path):
    source, _annotations = probed_source(tmp_path)
    probe_lines = [
        "_redundant_probe_9_4_inferred = 10",
        "reveal_type(_redundant_probe_9_4_inferred)",
        "def _redundant_probe_9_4_declared() -> int:",
        "reveal_type(_redundant_probe_9_4_declared())",
        "steps: int = 10",
    ]
    stripped = [line.strip() for line in source.splitlines() if line.strip()]
    start = stripped.index(probe_lines[0])
    assert stripped[start : start + 6] == [*probe_lines[:3], "...", *probe_lines[3:]]


# what basedpyright prints for the two probes of `steps: int = 10` on line 9
REVEALS = """\
  /x/train_redundant_probe.py:12:17 - information: Type of "_redundant_probe_9_4_inferred" is "Literal[10]"
  /x/train_redundant_probe.py:15:17 - information: Type of "_redundant_probe_9_4_declared()" is "int"
"""


def test_each_reveal_is_read_by_its_source_position():
    assert reveals(REVEALS) == {(9, 4, "inferred"): "Literal[10]", (9, 4, "declared"): "int"}


@pytest.mark.parametrize(
    ("revealed", "variable_type"),
    [
        ("Literal[10]", "int"),
        ("Literal[-1]", "int"),
        ("Literal['gpt']", "str"),
        ("Literal[b'x']", "bytes"),
        ("Literal[True]", "bool"),
        ("Literal[Color.RED]", "Color"),
        ("Literal[1, 2]", "Literal[1, 2]"),  # several values: no one class to widen to
        ("Tensor", "Tensor"),
    ],
)
def test_a_literal_widens_to_its_class(revealed: str, variable_type: str):
    assert widened(revealed) == variable_type
