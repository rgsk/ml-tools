"""any-check and the annotation checks in one basedpyright run, e.g. for an editor task on save.

What any-check flags (names that show white), plus annotations that don't earn their place:
- redundant: inference already gives T, e.g. `cfg: TrainCfg = TrainCfg(...)`. One kept on purpose,
  for the reader, is marked `# any:explicit` on its line.
- untyped: T itself is Unknown or Any to the checker (`table: pa.Table` without pyarrow's stubs), so
  the name still shows white, though any-check, seeing an annotation, passes it. `# any:explicit`
  doesn't hide this.
Left alone: class fields, Final/ClassVar/TypeAlias, a name bound again in its scope (the annotation
types the later bindings too), byname record literals and container literals (the annotation types,
or checks, what's inside).

Each checked `name: T = value` gets two probes at the start of its line in a copy of byname's
translation, `reveal_type` of value alone and of T, joined by `;` so no line moves. The copies go in
byname's mirror, next to each file's own so imports resolve, and every file is checked in one run, so
torch's types load once. Findings map back through byname's own source_position, the way
`byname tool` maps them, after undoing the probes' column shift.
Usage: uv run annotation-check FILE.pyn|FILE.py ...
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

from ml_tools.any_check import (
    CONFIGS,
    RULES,
    Finding,
    Func,
    findings,
    per_file,
    project_root,
    untyped,
)

PROBE = "_redundant_probe"  # prefix of every name the probes add
REVEAL = re.compile(rf'information: Type of "{PROBE}_(\d+)_(inferred|declared)" is "(.*)"')
LITERAL = re.compile(r"Literal\[(.*)\]")
EXPLICIT = re.compile(r"#.*\bany:explicit\b")  # a redundant annotation kept for the reader
# annotations that say more than the type, so never redundant
QUALIFIERS = {"Final", "ClassVar", "TypeAlias"}
# a container literal's annotation checks each element (`xs: list[int] = [1, "a"]` is an error,
# unannotated it's a list[int | str]), so it catches bugs rather than repeating the type
CONTAINERS = (ast.List, ast.Dict, ast.Set, ast.Tuple, ast.ListComp, ast.DictComp, ast.SetComp)
# a value that does something when evaluated twice, so the probe's copy could change the types after it
EFFECTS = (ast.NamedExpr, ast.Yield, ast.YieldFrom)


class Annotation(NamedTuple):
    line: int  # in the source, 1-based
    col: int  # in the source, 0-based
    name: str
    annotation: str  # as written


class Probe(NamedTuple):
    annotation: Annotation
    hidden_line: int  # in the translation, 0-based
    at: int  # where the probe text starts on that line
    width: int  # how long it is


class Source:
    """A file and the Python basedpyright checks for it: a .pyn's byname translation (`hidden`, with
    byname's helpers in), a .py as it is. Positions map hidden -> source as `byname tool` maps them."""

    def __init__(self, path: Path):
        self.path = path
        self.text = path.read_text()
        self.lines = self.text.splitlines()
        self.translation = None
        if path.suffix == ".pyn":
            from byname.srcmap import Translation

            self.translation = Translation(self.text, path)
            self.hidden = self.translation.hidden
        else:
            self.hidden = self.text

    def to_source(self, line: int, col: int, message: str) -> tuple[int, int] | None:
        # 0-based hidden position -> 0-based source one; None for byname's generated code
        if self.translation is None:
            return line, col
        from byname.tools import source_position

        pos = source_position(self.translation, line, col, message)
        if not isinstance(pos, dict):
            return None  # generated code, or "" for a walrus labelling a returned tuple
        return pos["line"], pos["character"]


def checkable(stmt: ast.AnnAssign) -> bool:
    # `name: T = value` with T a plain type and a value whose type isn't the annotation's doing
    if stmt.value is None or not isinstance(stmt.target, ast.Name):
        return False
    value = stmt.value
    # a byname record literal, `row: EvalRow = (step=10, ...)`, is typed by its annotation
    if isinstance(value, ast.Call) and getattr(value.func, "id", "").startswith("_rec_"):
        return False
    if isinstance(value, CONTAINERS):
        return False
    if any(isinstance(n, EFFECTS) for n in ast.walk(value)):
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


def candidates(tree: ast.Module) -> list[ast.AnnAssign]:
    # checkable annotations in function and module bodies. A class body's annotations declare its
    # fields (NamedTuple, dataclass), so only its methods are searched
    found: list[ast.AnnAssign] = []

    def search(body: list[ast.stmt], rebound_names: set[str]):
        for stmt in body:
            if isinstance(stmt, Func):
                search(stmt.body, rebound(stmt))
            elif isinstance(stmt, ast.ClassDef):
                for member in stmt.body:
                    if isinstance(member, Func):
                        search(member.body, rebound(member))
            elif isinstance(stmt, ast.AnnAssign):
                if checkable(stmt) and getattr(stmt.target, "id", "") not in rebound_names:
                    found.append(stmt)
            else:
                for field in ("body", "orelse", "finalbody"):
                    search(getattr(stmt, field, []), rebound_names)
                for inner in [*getattr(stmt, "handlers", []), *getattr(stmt, "cases", [])]:
                    search(inner.body, rebound_names)

    search(tree.body, rebound(tree))
    return found


def probed(source: Source) -> tuple[str, list[Probe]]:
    # source's hidden Python with the probes in, and where each one went
    hidden_lines = source.hidden.split("\n")
    probes: list[Probe] = []
    for stmt in candidates(ast.parse(source.hidden)):
        assert isinstance(stmt.target, ast.Name) and stmt.value is not None
        line = stmt.lineno - 1
        text = hidden_lines[line]
        at = len(text.encode()[: stmt.col_offset].decode())  # col_offset counts UTF-8 bytes
        mapped = source.to_source(line, at, "")
        if mapped is None:
            continue  # byname's generated code
        src_line, src_col = mapped
        if not source.lines[src_line][src_col:].startswith(stmt.target.id):
            continue
        if any(p.hidden_line == line for p in probes):
            continue  # a second statement on the line: its column shift would stack
        i = len(probes)
        probe = (
            f"import typing as {PROBE}_typing; "
            f"{PROBE}_{i}_inferred = {ast.unparse(stmt.value)}; "
            f"{PROBE}_{i}_declared = {PROBE}_typing.cast({ast.unparse(stmt.annotation)}, None); "
            f"reveal_type({PROBE}_{i}_inferred); reveal_type({PROBE}_{i}_declared); "
        )
        hidden_lines[line] = text[:at] + probe + text[at:]
        annotation = Annotation(src_line + 1, src_col, stmt.target.id, ast.unparse(stmt.annotation))
        probes.append(Probe(annotation, line, at, len(probe)))
    return "\n".join(hidden_lines), probes


def reveals(diagnostics: list[str]) -> dict[tuple[int, str], str]:
    # a file's diagnostics -> each probe's revealed type, by (probe number, inferred|declared)
    found: dict[tuple[int, str], str] = {}
    for text in diagnostics:
        hit = REVEAL.search(text)
        if hit is not None:
            i, which, type_text = hit.groups()
            found[(int(i), which)] = type_text
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


def source_findings(source: Source, probes: list[Probe], diagnostics: list[str]) -> list[Finding]:
    # any-check's findings in the probed copy -> at source positions. A probe's own names are dropped,
    # and a finding after a probe on its line moves back by the probe's width
    shifts = {p.hidden_line: p for p in probes}
    moved: list[Finding] = []
    for found in findings("\n".join(diagnostics)):
        if found.name.startswith(PROBE):
            continue
        line = found.line - 1
        col = found.col - 1
        probe = shifts.get(line)
        if probe is not None and col >= probe.at:
            col -= probe.width
        mapped = source.to_source(line, col, f'Type of "{found.name}" is {found.kind}')
        if mapped is not None:
            moved.append(found._replace(line=mapped[0] + 1, col=mapped[1] + 1))
    return moved


def annotation_reports(source: Source, probes: list[Probe], diagnostics: list[str]) -> list[str]:
    # the probes' reveals -> the redundant and the untyped annotations
    types = reveals(diagnostics)
    reports: list[str] = []
    for i, probe in enumerate(probes):
        a = probe.annotation
        inferred = types.get((i, "inferred"))
        declared = types.get((i, "declared"))
        if inferred is None or declared is None:
            continue  # unreachable code: basedpyright reveals nothing there
        where = f"{source.path}:{a.line}:{a.col + 1}"
        if "Unknown" in declared or declared == "Any":
            reports.append(f"{where} - {a.name}: {a.annotation} is {declared} to the checker")
        elif widened(inferred) == declared and not EXPLICIT.search(source.lines[a.line - 1]):
            reports.append(f"{where} - {a.name}: {a.annotation} is inferred anyway")
    return reports


def check_all(paths: list[Path]) -> list[str]:
    # every file in one basedpyright run -> its reports, file by file
    from byname.tools import mirror, mirrored

    root = project_root(paths[0])
    assert all(project_root(p) == root for p in paths), "files from one project per run"
    out = mirror(root)
    config_name = next(name for name in CONFIGS if (out / name).exists())
    sources = [Source(p) for p in paths]
    probe_files: list[Path] = []
    all_probes: list[list[Probe]] = []
    for source in sources:
        rel = mirrored(root, source.path)
        assert rel is not None, f"{source.path} isn't mirrored by byname"
        text, probes = probed(source)
        probe_file = out / rel.with_name(f"{source.path.stem}{PROBE}.py")
        probe_file.write_text(text)
        probe_files.append(probe_file)
        all_probes.append(probes)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "pyrightconfig.json"
            config.write_text(json.dumps({"extends": str(out / config_name), **RULES}))
            run = subprocess.run(
                ["basedpyright", "--pythonpath", sys.executable, "-p", str(config),
                 *map(str, probe_files)],
                capture_output=True, text=True, check=False, cwd=out,
            )
    finally:
        # the editor's byname sync drops mirror files with no source, the probes too, when the
        # project changes mid-run; basedpyright may not have read them, so silence isn't a pass
        removed = [not f.exists() for f in probe_files]
        for f in probe_files:
            f.unlink(missing_ok=True)
    diagnostics = per_file(run.stdout, out)
    reports: list[str] = []
    for source, probes, probe_file, gone in zip(sources, all_probes, probe_files, removed):
        if gone:
            reports.append(f"{source.path} - probe removed mid-run (editor sync), rerun")
            continue
        file_diagnostics = diagnostics.get(probe_file.resolve(), "").splitlines()
        reports += untyped(source.path, source_findings(source, probes, file_diagnostics))
        reports += annotation_reports(source, probes, file_diagnostics)
    return reports


def main() -> int:
    paths = [Path(p).resolve() for p in sys.argv[1:]]
    assert paths, "usage: annotation-check FILE.pyn|FILE.py ..."
    reports = check_all(paths)
    for report in reports:
        print(report)
    return 1 if reports else 0


if __name__ == "__main__":
    sys.exit(main())
