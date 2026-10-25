# annotation-check's decisions on a .py file and basedpyright output written out
# by hand: which annotations get probed, where the probes go, how the reveals
# are read back, and how findings move back past a probe
from pathlib import Path

import pytest

from ml_tools.annotation_check import (
    Annotation,
    Source,
    probed,
    reveals,
    source_findings,
    untyped_annotation,
    widened,
)
from ml_tools.any_check import Finding

TRAIN = """\
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
STEPS_LINE = 9
STEPS = "    steps: int = 10"
PROBE_FILE = "/x/train_redundant_probe.py"


@pytest.fixture
def train_source(tmp_path: Path) -> Source:
    path = tmp_path / "train.py"
    path.write_text(TRAIN)
    return Source(path)


def test_only_a_plain_annotated_assignment_is_probed(train_source: Source):
    _text, probes = probed(train_source)
    # the class field, Final, the record literal, the container literal, the
    # bare `n: int` and the rebound `out` are left alone
    steps = Annotation(line=STEPS_LINE, col=4, name="steps", annotation="int")
    assert [p.annotation for p in probes] == [steps]


def test_the_probes_go_on_the_line_itself_so_no_line_moves(train_source: Source):
    text, (probe,) = probed(train_source)
    lines = text.split("\n")
    assert len(lines) == len(TRAIN.split("\n"))
    probed_line = lines[STEPS_LINE - 1]
    assert probed_line[: probe.at] + probed_line[probe.at + probe.width :] == STEPS
    assert "_redundant_probe_0_inferred = 10;" in probed_line


# what basedpyright prints for the two probes of `steps: int = 10`
REVEALS = [
    f'  {PROBE_FILE}:9:96 - information: Type of "_redundant_probe_0_inferred" is "Literal[10]"',
    f'  {PROBE_FILE}:9:140 - information: Type of "_redundant_probe_0_declared" is "int"',
]


def test_each_reveal_is_read_by_its_probe_number():
    assert reveals(REVEALS) == {(0, "inferred"): "Literal[10]", (0, "declared"): "int"}


def test_a_finding_after_a_probe_moves_back_and_the_probes_own_are_dropped(train_source: Source):
    _text, (probe,) = probed(train_source)
    steps_col = STEPS.index("steps") + 1  # 1-based, as basedpyright counts
    in_probe = f'  {PROBE_FILE}:{STEPS_LINE}:{probe.at + 30} - error: Type of "_redundant_probe_0_inferred" is Any'
    # `steps` sits after the probe, so basedpyright sees it probe.width further right
    after_probe = f'  {PROBE_FILE}:{STEPS_LINE}:{steps_col + probe.width} - error: Type of "steps" is Any'
    (found,) = source_findings(train_source, [probe], [in_probe, after_probe])
    assert found == Finding(line=STEPS_LINE, col=steps_col, name="steps", kind="Any", type_text="")


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


@pytest.mark.parametrize(
    ("declared", "untyped"),
    [
        ("Unknown", True),  # `table: pa.Table` without pyarrow's stubs
        ("Any", True),
        ("ndarray[tuple[int, ...], dtype[Unknown]]", False),  # `NDArray`: still completes
        ("dict[str, Tensor]", False),
    ],
)
def test_only_a_wholly_unknown_annotation_is_untyped(declared: str, untyped: bool):
    assert untyped_annotation(declared) == untyped
