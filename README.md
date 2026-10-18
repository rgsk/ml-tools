# ml-tools

Dev tools shared by the ML projects next to this folder, so each lives in one
place instead of a copy per repo. Kept out of byname on purpose: byname is a
general Python tool, used by projects without torch.

## any-check

Lists local names whose type is `Any`/`Unknown`, which show white in the editor
(no colour, no completion), e.g. `logits = model(x)` from an untyped
`nn.Module` call. Fix one by annotating where the name is first bound
(`logits: Tensor = model(x)`), or mark a line `# any:ok`.

    uv run any-check src/model.pyn tests/test_model.pyn

## annotation-check

any-check plus two checks on annotations, all in one basedpyright run (the
editor task uses it):

- redundant: inference already gives the type, e.g.
  `cfg: TrainCfg = TrainCfg(...)` or `n: int = 0`. Mark one kept on purpose,
  for the reader, with `# any:explicit` on its line.
- untyped: the annotation's type is itself `Unknown`/`Any` to the checker,
  e.g. `table: pa.Table` without pyarrow's stubs, so the name still shows
  white. `# any:explicit` doesn't hide this one.

It leaves alone class fields, `Final`/`ClassVar`/`TypeAlias`, a name bound
again in its scope (the annotation types the later bindings too), annotations
that change the type (`x: float = 0`), byname record literals
(`row: EvalRow = (step=10, ...)`) and container literals
(`metrics: dict[str, float] = {...}`, whose annotation checks each element).
Every file goes in one run, so torch's types load once: all of llm-final in
about 2s, against any-check's 19s.

    uv run annotation-check src/model.pyn tests/test_model.pyn

Both commands come from one dev dependency:

    [dependency-groups]
    dev = [..., "ml-tools"]

    [tool.uv.sources]
    ml-tools = { path = "../ml-tools", editable = true }
