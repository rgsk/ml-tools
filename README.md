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

`--redundant` goes the other way: it lists annotations that inference already
gives, e.g. `cfg: TrainCfg = TrainCfg(...)` or `n: int = 0`. It leaves alone
class fields, `Final`/`ClassVar`/`TypeAlias`, annotations that change the type
(`x: float = 0`, `xs: list[int] = []`) and byname record literals
(`row: EvalRow = (step=10, ...)`), whose type comes from the annotation, and
container literals (`metrics: dict[str, float] = {...}`), whose annotation
checks each element.
Mark one kept on purpose, for the reader, with `# any:explicit` on its line.
It also lists annotations whose type is itself `Unknown`/`Any` to the checker,
e.g. `table: pa.Table` without pyarrow's stubs: the name still shows white.

    uv run any-check --redundant src/model.pyn

A project adds it as a dev dependency:

    [dependency-groups]
    dev = [..., "ml-tools"]

    [tool.uv.sources]
    ml-tools = { path = "../ml-tools", editable = true }
