# any-check's two decisions, on basedpyright output written out by hand: which
# findings it reads, and which partially unknown names still complete in the
# editor and so aren't reported
import pytest

from ml_tools.any_check import findings, still_completes

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
