"""Flag local names whose type is Any/Unknown (shown white in the editor), e.g. `logits = model(x)`.

The three rules are switched on by a temporary config outside the project that extends the project's
own (for a .pyn, the one in byname's mirror), so no file is written into the project.
basedpyright's reportAny also flags returns, arguments and assignments to declared names; those are
fine, so only `Type of "name" is ...` (or `Type of parameter "name"`) is kept, and only when `name`
has no annotation in its scope.
A partially unknown name is kept only if part of it shows white: `list[Unknown | Tensor]` still
completes as a Tensor (torch's own untyped returns cause these), `list[Unknown]` doesn't.
A line with `# any:ok` in its comment is skipped: an Any that's fine (`std *= n ** -0.5  # any:ok`).
A project's files are checked in one run per config (.pyn or .py), so torch's types load once.
A notebook is checked as its code cells joined into one file (see notebook), in annotation-check's run
with no probes, and reported by cell: `nb.ipynb:cell 3:2:5`.
A name the file never writes is byname's (`_ds` behind `(a=, b=) = r`) and isn't reported.
Usage: uv run any-check FILE.pyn|FILE.py|FILE.ipynb ...
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
RULES = {"reportAny": True, "reportUnknownVariableType": True, "reportUnknownParameterType": True}
CONFIGS = ("pyrightconfig.json", "pyproject.toml")  # where a project configures the checker
HIT = re.compile(
    r':(\d+):(\d+) - error: Type of (?:parameter )?"(\w+)" is (Any|partially unknown|unknown)'
)  # a parameter with no annotation and no default (`def on_key(event)`) is reported as unknown
# the line basedpyright prints under a partially unknown finding, with the full type
DETAIL = re.compile(r'^\s+Type of "(\w+)" is "(.*)"')
# a diagnostic line and its file, as `byname tool` (project-relative) or basedpyright (absolute) prints
# it; the lines under it, indented with \xa0s, carry no location
LOCATION = re.compile(r"^\s+(\S+\.pyn?):\d+:\d+ - ")
# an Unknown that shares a union with another type, at either end of the union
UNKNOWN_IN_UNION = re.compile(r"Unknown \| | \| Unknown")
Func = ast.FunctionDef | ast.AsyncFunctionDef
SKIP = re.compile(r"#.*\bany:ok\b")
MAGIC = re.compile(r"^\s*[%!]")  # an IPython magic or shell line, not Python
LOAD_BYNAME = re.compile(r"^[ \t]*%load_ext[ \t]+byname\b", re.MULTILINE)  # a byname notebook


class Finding(NamedTuple):
    line: int
    col: int
    name: str
    kind: str  # Any, partially unknown or unknown
    type_text: str  # the full type, when basedpyright prints it; "" otherwise


class Cell(NamedTuple):
    number: int  # 1-based, counting every cell, as Ruff does
    first: int  # its first line in the notebook's joined source, 1-based


def notebook(path: Path) -> tuple[str, list[Cell], bool]:
    # a notebook's code cells as one source, magics blanked so no line moves; where each cell
    # starts in it; whether a cell loads byname (then the source is a .pyn's)
    nb = json.loads(path.read_text())
    parts: list[str] = []
    cells: list[Cell] = []
    line = 1
    uses_byname = False
    for number, c in enumerate(nb.get("cells", []), 1):
        if c.get("cell_type") != "code":
            continue
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        uses_byname |= LOAD_BYNAME.search(src) is not None
        lines = ["" if MAGIC.match(text) else text for text in src.split("\n")]
        cells.append(Cell(number, line))
        parts.append("\n".join(lines))
        line += len(lines)
    return "\n".join(parts), cells, uses_byname


def source_text(path: Path) -> str:
    # the source the checker's findings are positioned in: a notebook's joined code cells
    return notebook(path)[0] if path.suffix == ".ipynb" else path.read_text()


def is_byname(path: Path) -> bool:
    # path's source is written in byname, so it's checked through its translation
    return path.suffix == ".pyn" or (path.suffix == ".ipynb" and notebook(path)[2])


def where(path: Path, line: int, col: int) -> str:
    # a 1-based position in path's source as reported: a notebook's by cell
    if path.suffix != ".ipynb":
        return f"{path}:{line}:{col}"
    cell = [c for c in notebook(path)[1] if c.first <= line][-1]
    return f"{path}:cell {cell.number}:{line - cell.first + 1}:{col}"


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
    """AST with the file's own line numbers (a notebook's: its joined source's)."""
    if not is_byname(path):
        return ast.parse(source_text(path))
    from byname import source_ast

    return source_ast(source_text(path), str(path))


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


def checker_config(base: Path) -> str:
    # a config that switches RULES on over base. A config's folder is the project root, so the
    # temp folder would be; the execution root stays base's folder, where `from oracle import
    # Oracle` in tests/ finds tests/oracle.py (a project's own executionEnvironments would be lost)
    root = {"root": str(base.parent)}
    return json.dumps({"extends": str(base), **RULES, "executionEnvironments": [root]})


def basedpyright_output(paths: list[Path]) -> str:
    # files of one project and one kind (.pyn or .py) checked with RULES on in one run, through a
    # config in a temp folder that extends the project's, so nothing is written into the project
    first = paths[0]
    cmd = ["byname", "tool", "basedpyright"] if first.suffix == ".pyn" else ["basedpyright"]
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "pyrightconfig.json"
        config.write_text(checker_config(base_config(first)))
        run = subprocess.run(
            [*cmd, "-p", str(config), *map(str, paths)],
            capture_output=True, text=True, check=False, cwd=project_root(first),
        )
    return run.stdout


def per_file(out: str, root: Path) -> dict[Path, str]:
    # one run's output -> each file's diagnostics, a location line with the lines under it
    files: dict[Path, list[str]] = {}
    current: list[str] | None = None
    for text in out.splitlines():
        hit = LOCATION.match(text)
        if hit is not None:
            current = files.setdefault((root / hit.group(1)).resolve(), [])
            current.append(text)
        elif current is not None and text[:1].isspace():
            current.append(text)  # the full type under a finding
        else:
            current = None
    return {path: "\n".join(lines) for path, lines in files.items()}


def check(paths: list[Path]) -> list[str]:
    # files -> their untyped names, as "path:line:col - name is kind", file by file, notebooks last
    groups: dict[tuple[Path, str], list[Path]] = {}
    notebooks = [p for p in paths if p.suffix == ".ipynb"]
    paths = [p for p in paths if p.suffix != ".ipynb"]
    for path in paths:
        kind = ".pyn" if path.suffix == ".pyn" else ".py"  # each kind runs with its own config
        groups.setdefault((project_root(path), kind), []).append(path)
    outputs: dict[Path, str] = {}
    for (root, _kind), group in groups.items():
        outputs |= per_file(basedpyright_output(group), root)
    reports = [report for path in paths for report in untyped(path, findings(outputs.get(path, "")))]
    if notebooks:
        from ml_tools.annotation_check import check_all

        reports += check_all(notebooks, annotations=False)
    return reports


def untyped(path: Path, found_list: list[Finding]) -> list[str]:
    # path's findings, at its own line numbers -> the names that show white, as "path:line:col - name is kind"
    tree = source_tree(path)
    source = source_text(path)
    lines = source.splitlines()
    written = set(re.findall(r"\w+", source))  # a name not in here is byname's, nothing to fix
    # a library's own Unknowns (`from jaxtyping import jaxtyped`): nothing a local annotation can fix
    imported = {
        (line, (a.asname or a.name).split(".")[0])
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for line in range(n.lineno, (n.end_lineno or n.lineno) + 1)
        for a in n.names
    }
    # undeclared names that are Any themselves. An attribute of one (`logits.shape`) is fixed by
    # annotating the name, so only the name is reported
    any_names = {
        f.name
        for f in found_list
        if lines[f.line - 1][f.col - 1 :].startswith(f.name) and not declared(tree, f.line, f.name)
    }
    hits: set[tuple[int, int, str, str]] = set()
    for found in found_list:
        line = found.line
        col = found.col
        name = found.name
        text = lines[line - 1]
        if name not in written:
            continue
        if not text[col - 1 :].startswith(name):
            # an attribute, reported where its expression starts. Its root is declared, so the Any
            # comes from the expression (`axes[b, 0].imshow` with `axes: np.ndarray`)
            root = re.match(r"\w*", text[col - 1 :])
            if (root is not None and root.group() in any_names) or SKIP.search(text):
                continue
            end = text.find(f".{name}", col - 1)
            shown = text[col - 1 : end + 1 + len(name)] if end >= 0 else name
            hits.add((line, col, shown, found.kind))
            continue
        if SKIP.search(text):
            continue
        if (line, name) in imported:
            continue
        if found.kind == "partially unknown" and still_completes(found.type_text):
            continue
        if not declared(tree, line, name):
            hits.add((line, col, name, found.kind))
    return [f"{where(path, line, col)} - {name} is {kind}" for line, col, name, kind in sorted(hits)]


def main() -> int:
    paths = [Path(p).resolve() for p in sys.argv[1:]]
    assert paths, "usage: any-check FILE.pyn|FILE.py|FILE.ipynb ..."
    reports = check(paths)
    for report in reports:
        print(report)
    return 1 if reports else 0


if __name__ == "__main__":
    sys.exit(main())
