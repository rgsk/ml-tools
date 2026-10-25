# ml-tools

Dev tools shared by the ML projects next to this folder, so each lives in one
place instead of a copy per repo. Kept out of byname on purpose: byname is a
general Python tool, used by projects without torch.

## any-check

Lists local names whose type is `Any`/`Unknown`, which show white in the editor
(no colour, no completion), e.g. `logits = model(x)` from an untyped
`nn.Module` call. Fix one by annotating where the name is first bound
(`logits: Tensor = model(x)`), or mark a line `# any:ok`.
Also lists an attribute that's wholly Unknown though no name is bound, e.g.
`print(ds[0]["img"].shape)` from an untyped `__getitem__`; a library's
half-typed member (torch's `.numpy()`) still completes and isn't listed.

    uv run any-check src/model.pyn tests/test_model.pyn

## annotation-check

any-check plus two checks on annotations, all in one basedpyright run (the
editor task uses it):

- redundant: inference already gives the type, e.g.
  `cfg: TrainCfg = TrainCfg(...)` or `n: int = 0`. Mark one kept on purpose,
  for the reader, with `# any:explicit` on its line.
- untyped: the annotation's type is itself `Unknown`/`Any` to the checker,
  e.g. `table: pa.Table` without pyarrow's stubs, so the name still shows
  white. `# any:explicit` doesn't hide this one. A partly typed annotation
  (`NDArray`, whose dtype is Unknown) still completes and passes.

It leaves alone class fields, `Final`/`ClassVar`/`TypeAlias`, a name bound
again in its scope (the annotation types the later bindings too), annotations
that change the type (`x: float = 0`), byname record literals
(`row: EvalRow = (step=10, ...)`) and container literals
(`metrics: dict[str, float] = {...}`, whose annotation checks each element).
Every file goes in one run, so torch's types load once: all of llm-final in
about 2s.

    uv run annotation-check src/model.pyn tests/test_model.pyn

## In the editor, as you type

byname (0.4.8+) runs any-check live: its checker reports with any-check's
rules on, and `lsp_filter` keeps the names that show white, as warnings.
A fix clears its squiggles at once, and the ones computed from it too.
In the project's `.vscode/settings.json`:

```json
"byname.checkerRules": {
    "reportAny": "warning",
    "reportUnknownVariableType": "warning",
    "reportUnknownParameterType": "warning",
    "reportUnknownMemberType": "warning"
},
"byname.diagnosticsFilter": "ml_tools.any_check:lsp_filter"
```

The annotation checks need annotation-check's probe run, so they aren't
live: run annotation-check for them, by hand (Alt+L) or on save;
`--annotations-only` leaves out the names the editor already shows.

## Notebooks

Both take a `.ipynb` too. Its code cells are checked as one file (IPython `%`
and `!` lines left out, so cells see each other's names, as in a run-all), as
byname when a cell has `%load_ext byname`, and findings come back by cell,
counting every cell from 1 as Ruff does:

    uv run annotation-check rough/act.ipynb
    rough/act.ipynb:cell 6:1:17 - action is unknown

A name the file never writes is byname's own (`_ds` behind `(a=, b=) = r`), so
neither tool reports it.

Both commands come from one dev dependency:

    [dependency-groups]
    dev = [..., "ml-tools"]

    [tool.uv.sources]
    ml-tools = { path = "../ml-tools", editable = true }
