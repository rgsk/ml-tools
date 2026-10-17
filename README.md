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

A project adds it as a dev dependency:

    [dependency-groups]
    dev = [..., "ml-tools"]

    [tool.uv.sources]
    ml-tools = { path = "../ml-tools", editable = true }
