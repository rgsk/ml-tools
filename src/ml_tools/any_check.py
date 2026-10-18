"""Flag local names whose type is Any/Unknown (shown white in the editor), e.g. `logits = model(x)`.

The two rules are switched on by a temporary config outside the project that extends the project's
own (for a .pyn, the one in byname's mirror), so no file is written into the project.
basedpyright's reportAny also flags returns, arguments and assignments to declared names; those are
fine, so only `Type of "name" is ...` is kept, and only when `name` has no annotation in its scope.
A partially unknown name is kept only if part of it shows white: `list[Unknown | Tensor]` still
completes as a Tensor (torch's own untyped returns cause these), `list[Unknown]` doesn't.
A line with `# any:ok` in its comment is skipped: an Any that's fine (`std *= n ** -0.5  # any:ok`).

--redundant goes the other way: it lists annotations that inference already gives, e.g.
`cfg: TrainCfg = TrainCfg(...)`. Before each `name: T = value` in a function or module body, a copy
of the file gets two probes, `reveal_type` of value alone and of a function returning T, and the
annotation is redundant when the two agree. It's wrong when T itself is Unknown or Any to the
checker (`table: pa.Table` without pyarrow's stubs): the name still shows white, though the main
check, seeing an annotation, passes it. The copy goes in byname's mirror, next to the file's own
copy so its imports resolve, and is removed after the run. A redundant annotation kept on purpose,
for the reader, is marked `# any:explicit` on its line (`batch: Batch = to_device(...)  # any:explicit`);
the mark doesn't hide an Unknown T, which still shows white. A container literal's annotation
(`metrics: dict[str, float] = {...}`) is never redundant: it checks each element.
Usage: uv run any-check [--redundant] FILE.pyn|FILE.py ...
"""

import ast
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import NamedTuple

# the rules a normal run leaves off, switched on for this check only
RULES = {"reportAny": True, "reportUnknownVariableType": True}
CONFIGS = ("pyrightconfig.json", "pyproject.toml")  # where a project configures the checker
HIT = re.compile(
    r':(\d+):(\d+) - error: Type of "(\w+)" is (Any|partially unknown|unknown)'
)
# the line basedpyright prints under a partially unknown finding, with the full type
DETAIL = re.compile(r'^\s+Type of "(\w+)" is "(.*)"')
# an Unknown that shares a union with another type, at either end of the union
UNKNOWN_IN_UNION = re.compile(r"Unknown \| | \| Unknown")
Func = ast.FunctionDef | ast.AsyncFunctionDef
SKIP = re.compile(r"#.*\bany:ok\b")
EXPLICIT = re.compile(r"#.*\bany:explicit\b")  # a redundant annotation kept for the reader


class Finding(NamedTuple):
    line: int
    col: int
    name: str
    kind: str  # Any, partially unknown or unknown
    type_text: str  # the full type, when basedpyright prints it; "" otherwise


def findings(out: str) -> list[Finding]:
    # basedpyright's output -> every `Type of "name" is ...` finding, with the
    # full type from the line under it when there is one
    out_lines = out.splitlines()
    found: list[Finding] = []
    for i, text in enumerate(out_lines):
        hit = HIT.search(text)
        if hit is None:
            continue
        line_s, col_s, name, kind = hit.groups()
        detail = DETAIL.match(out_lines[i + 1]) if i + 1 < len(out_lines) else None
        same_name = detail is not None and detail.group(1) == name
        type_text = detail.group(2) if detail is not None and same_name else ""
        found.append(Finding(int(line_s), int(col_s), name, kind, type_text))
    return found


def still_completes(type_text: str) -> bool:
    # every Unknown in the type shares a union with a known type, so the editor
    # still colours the name and completes it from the known part
    # (list[Unknown | Tensor] completes as Tensor). A bare Unknown
    # (list[Unknown]) shows white, and so does a union whose only known part is
    # None, because None has nothing to complete
    if "Unknown" not in type_text:
        return False  # no full type to judge by, so keep the finding
    known = UNKNOWN_IN_UNION.sub("", type_text)
    return "Unknown" not in known and known != "None"


def source_tree(path: Path) -> ast.Module:
    """AST with the file's own line numbers."""
    if path.suffix != ".pyn":
        return ast.parse(path.read_text())
    from byname import source_ast

    return source_ast(path.read_text(), str(path))


def declared(tree: ast.Module, line: int, name: str) -> bool:
    """`name` is annotated (parameter or `name: T`) in the innermost function around `line`."""
    scope: ast.Module | Func = tree
    for node in ast.walk(tree):
        if (
            isinstance(node, Func)
            and node.lineno <= line <= (node.end_lineno or 0)
            and (isinstance(scope, ast.Module) or node.lineno >= scope.lineno)
        ):
            scope = node
    if isinstance(scope, Func):
        a = scope.args
        if any(
            p.arg == name and p.annotation
            for p in [*a.posonlyargs, *a.args, *a.kwonlyargs]
        ):
            return True
    return any(
        isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == name
        for n in ast.walk(scope)
    )


def project_root(path: Path) -> Path:
    # the nearest folder above path holding pyproject.toml
    for d in path.parents:
        if (d / "pyproject.toml").exists():
            return d
    raise RuntimeError(f"no pyproject.toml above {path}")


def base_config(path: Path) -> Path:
    # the checker config the project's own runs use. A .pyn is checked in byname's
    # mirror (translated, under ~/.cache/byname), so its config is the mirror's
    # copy, whose relative paths (extraPaths) then point into the mirror
    root = project_root(path)
    found = [name for name in CONFIGS if (root / name).exists()]
    assert found, f"no {' or '.join(CONFIGS)} in {root}"
    if path.suffix == ".pyn":
        from byname.tools import mirror_dir

        return mirror_dir(root, "tools") / found[0]
    return root / found[0]


def basedpyright_output(path: Path) -> str:
    # one file checked with RULES on, through a config in a temp folder that
    # extends the project's, so nothing is written into the project
    cmd = ["byname", "tool", "basedpyright"] if path.suffix == ".pyn" else ["basedpyright"]
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "pyrightconfig.json"
        config.write_text(json.dumps({"extends": str(base_config(path)), **RULES}))
        run = subprocess.run(
            [*cmd, "-p", str(config), str(path)],
            capture_output=True, text=True, check=False, cwd=project_root(path),
        )
    return run.stdout


def check(path: Path) -> list[str]:
    # one file -> its untyped names, as "path:line:col - name is kind"
    out = basedpyright_output(path)
    tree = source_tree(path)
    lines = path.read_text().splitlines()
    # a library's own Unknowns (`from jaxtyping import jaxtyped`): nothing a local annotation can fix
    imported = {
        (line, (a.asname or a.name).split(".")[0])
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for line in range(n.lineno, (n.end_lineno or n.lineno) + 1)
        for a in n.names
    }
    hits: set[tuple[int, int, str, str]] = set()
    for found in findings(out):
        line = found.line
        col = found.col
        name = found.name
        if not lines[line - 1][col - 1 :].startswith(name):
            continue  # `logits.shape`, reported at `logits`: an attribute of an Any value, not a name
        if SKIP.search(lines[line - 1]):
            continue
        if (line, name) in imported:
            continue
        if found.kind == "partially unknown" and still_completes(found.type_text):
            continue
        if not declared(tree, line, name):
            hits.add((line, col, name, found.kind))
    return [f"{path}:{line}:{col} - {name} is {kind}" for line, col, name, kind in sorted(hits)]


PROBE = "_redundant_probe"  # prefix of every name the probes add
REVEAL = re.compile(
    rf'information: Type of "{PROBE}_(\d+)_(\d+)_(inferred|declared)(?:\(\))?" is "(.*)"'
)
LITERAL = re.compile(r"Literal\[(.*)\]")
# annotations that say more than the type, so never redundant
QUALIFIERS = {"Final", "ClassVar", "TypeAlias"}
# a container literal's annotation checks each element (`xs: list[int] = [1, "a"]` is an error,
# unannotated it's a list[int | str]), so it catches bugs rather than repeating the type
CONTAINERS = (ast.List, ast.Dict, ast.Set, ast.Tuple, ast.ListComp, ast.DictComp, ast.SetComp)


class Annotation(NamedTuple):
    line: int
    col: int
    name: str
    annotation: str  # as written


def checkable(stmt: ast.AnnAssign, lines: list[str]) -> bool:
    # `name: T = value` the file itself writes (not byname's generated code), with T a plain type
    if stmt.value is None or not isinstance(stmt.target, ast.Name):
        return False
    if not lines[stmt.lineno - 1][stmt.col_offset :].startswith(stmt.target.id):
        return False
    # a byname record literal, `row: EvalRow = (step=10, ...)`, is typed by its annotation
    value = stmt.value
    if isinstance(value, ast.Call) and getattr(value.func, "id", "").startswith("_rec_"):
        return False
    if isinstance(value, CONTAINERS):
        return False
    root = stmt.annotation
    if isinstance(root, ast.Subscript):
        root = root.value  # Final[int] -> Final
    root_name = root.attr if isinstance(root, ast.Attribute) else getattr(root, "id", "")
    return root_name not in QUALIFIERS


def rebound(scope: ast.Module | Func) -> set[str]:
    # names bound more than once in scope. Nested functions count as part of it, so a name only
    # reused in one of those is skipped too (a missed finding, never a wrong one)
    binds = Counter(
        n.id for n in ast.walk(scope) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    )
    return {name for name, count in binds.items() if count > 1}


class Probes(ast.NodeTransformer):
    """Puts two probes before each checkable annotation, in function and module bodies. A class
    body's annotations declare its fields (NamedTuple, dataclass), so only its methods are visited.
    A name bound again in its scope is skipped too: the annotation declares the type for every later
    binding (`out: Tensor = attention(...)`, then `out = self.proj(out)` from an untyped call)."""

    def __init__(self, lines: list[str]):
        self.lines = lines
        self.found: list[Annotation] = []
        self.rebound: set[str] = set()  # names bound more than once in the current scope

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.rebound = rebound(node)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        outer = self.rebound
        self.rebound = rebound(node)
        self.generic_visit(node)
        self.rebound = outer
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        outer = self.rebound
        self.rebound = rebound(node)
        self.generic_visit(node)
        self.rebound = outer
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.body = [self.visit(s) if isinstance(s, Func) else s for s in node.body]
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt | list[ast.stmt]:
        if not checkable(node, self.lines):
            return node
        assert isinstance(node.target, ast.Name) and node.value is not None
        if node.target.id in self.rebound:
            return node
        annotation = ast.unparse(node.annotation)
        self.found.append(Annotation(node.lineno, node.col_offset, node.target.id, annotation))
        # the value's type with no annotation to guide it, then T's own type
        probe = f"{PROBE}_{node.lineno}_{node.col_offset}"
        probes = ast.parse(
            f"{probe}_inferred = {ast.unparse(node.value)}\n"
            f"reveal_type({probe}_inferred)\n"
            f"def {probe}_declared() -> {annotation}: ...\n"
            f"reveal_type({probe}_declared())\n"
        ).body
        return [*probes, node]


def probed(path: Path) -> tuple[str, list[Annotation]]:
    # the file as Python with the probes in, and the annotations they probe. A .pyn's translation keeps
    # its line numbers, and its prelude (byname's helpers) goes in after probing so it isn't probed
    text = path.read_text()
    lines = text.splitlines()
    prelude: list[ast.stmt] = []
    if path.suffix == ".pyn":
        from byname import transform
        from byname.transform import prelude_index

        translation = transform(text, str(path))
        tree = ast.parse(translation.body)
        prelude = ast.parse(translation.prelude).body
    else:
        tree = ast.parse(text)
    probes = Probes(lines)
    tree = probes.visit(tree)
    if prelude:
        k = prelude_index(tree)
        tree.body[k:k] = prelude
    return ast.unparse(tree), probes.found


def reveals(out: str) -> dict[tuple[int, int, str], str]:
    # basedpyright's output -> each probe's revealed type, by (line, col, inferred|declared)
    found: dict[tuple[int, int, str], str] = {}
    for text in out.splitlines():
        hit = REVEAL.search(text)
        if hit is not None:
            line_s, col_s, which, type_text = hit.groups()
            found[(int(line_s), int(col_s), which)] = type_text
    return found


def widened(type_text: str) -> str:
    # a literal value's type as a variable's: `x = 0` reveals Literal[0], but x is an int
    hit = LITERAL.fullmatch(type_text)
    if hit is None or "," in hit.group(1):
        return type_text
    value = hit.group(1)
    if value in ("True", "False"):
        return "bool"
    if value.startswith(("b'", 'b"')):
        return "bytes"
    if value.startswith(("'", '"')):
        return "str"
    if re.fullmatch(r"-?\d+", value):
        return "int"
    return value.rsplit(".", 1)[0]  # an enum member: Literal[Color.RED] -> Color


def redundant(path: Path) -> list[str]:
    # one file -> its annotations inference already gives, as "path:line:col - name: T ..."
    from byname.tools import mirror, mirrored

    source, annotations = probed(path)
    root = project_root(path)
    out = mirror(root)
    rel = mirrored(root, path)
    assert rel is not None, f"{path} isn't mirrored by byname"
    probe_file = out / rel.with_name(f"{path.stem}{PROBE}.py")
    probe_file.write_text(source)
    try:
        run = subprocess.run(
            ["basedpyright", "--pythonpath", sys.executable, str(probe_file)],
            capture_output=True, text=True, check=False, cwd=out,
        )
    finally:
        # the editor's byname sync drops mirror files with no source, the probe too, when the
        # project changes mid-run; basedpyright may not have read it, so no reveals isn't a pass
        removed = not probe_file.exists()
        probe_file.unlink(missing_ok=True)
    if removed:
        return [f"{path} - probe removed mid-run (editor sync), rerun"]
    types = reveals(run.stdout)
    lines = path.read_text().splitlines()
    reports: list[str] = []
    for a in annotations:
        inferred = types.get((a.line, a.col, "inferred"))
        declared = types.get((a.line, a.col, "declared"))
        if inferred is None or declared is None:
            continue  # unreachable code: basedpyright reveals nothing there
        if "Unknown" in declared or declared == "Any":
            # T is untyped itself (pa.Table without pyarrow's stubs), so the name still shows white
            reports.append(f"{path}:{a.line}:{a.col + 1} - {a.name}: {a.annotation} is {declared} to the checker")
        elif widened(inferred) == declared and not EXPLICIT.search(lines[a.line - 1]):
            reports.append(f"{path}:{a.line}:{a.col + 1} - {a.name}: {a.annotation} is inferred anyway")
    return reports


def main() -> int:
    args = sys.argv[1:]
    run_redundant = "--redundant" in args
    paths = [Path(p).resolve() for p in args if p != "--redundant"]
    assert paths, "usage: any-check [--redundant] FILE.pyn|FILE.py ..."
    one_file = redundant if run_redundant else check
    reports = [report for path in paths for report in one_file(path)]
    for report in reports:
        print(report)
    return 1 if reports else 0


if __name__ == "__main__":
    sys.exit(main())
