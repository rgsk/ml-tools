"""Flag local names whose type is Any/Unknown (shown white in the editor), e.g. `logits = model(x)`.

The two rules are switched on by a temporary config outside the project that extends the project's
own (for a .pyn, the one in byname's mirror), so no file is written into the project.
basedpyright's reportAny also flags returns, arguments and assignments to declared names; those are
fine, so only `Type of "name" is ...` is kept, and only when `name` has no annotation in its scope.
A partially unknown name is kept only if part of it shows white: `list[Unknown | Tensor]` still
completes as a Tensor (torch's own untyped returns cause these), `list[Unknown]` doesn't.
A line with `# any:ok` in its comment is skipped: an Any that's fine (`std *= n ** -0.5  # any:ok`).
Usage: uv run any-check FILE.pyn|FILE.py ...
"""

import ast
import json
import re
import subprocess
import sys
import tempfile
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


def main() -> int:
    paths = [Path(p).resolve() for p in sys.argv[1:]]
    assert paths, "usage: any-check FILE.pyn|FILE.py ..."
    reports = [report for path in paths for report in check(path)]
    for report in reports:
        print(report)
    return 1 if reports else 0


if __name__ == "__main__":
    sys.exit(main())
