# Running the agent

## The terminal UI

```bash
bias-scope-agent               # terminal UI: You > / BiasScope>, replies rendered, tool calls shown live
bias-scope-agent --autonomous  # asks only for a model id, evaluates it end to end, asks for the next
bias-scope-agent --plain       # the line-by-line REPL, raw text
```

The UI is a [Textual](https://textual.textualize.io/) app, and `textual` is a core
dependency. Type a turn at `You >`. While the agent works, the tools it calls scroll past
as dim lines, then the reply appears under `BiasScope>` with its tables and emphasis
rendered.

When a report is in, its rows appear as a table (metric, family, score, n, fidelity,
deviation) and the UI asks whether to test another model. `yes` starts a fresh session and
`no` leaves. `exit`, Esc, Ctrl-Q or Ctrl-C leaves at any time. In a pipe, or without
`textual`, the plain REPL is used automatically.

## Autonomous mode

`--autonomous` is the same UI with only one question: the model id. It works out the kind
of model (masked LM, decoder, sentence encoder, or an `openrouter/...` API model), sends
the agent the same three turns the scripted runner uses, **confirms the plan on your
behalf**, shows the table and asks for the next model.

Nothing else is asked, so use it only for a model you would have confirmed anyway. The
[gate](safeguards.md) is untouched: the plan is still shown, and confirmed in a later turn,
but the confirmation is sent for you. `--device` picks the GPU (default: CUDA if
available).

## Scripted and recorded runs

For runs with a full tool-dispatch log, recorded as JSON:

```bash
python scripts/agent/live_conversation.py                 # interactive: you type, it records
python scripts/agent/live_conversation.py --autonomous    # asks only for model ids; one transcript each
python scripts/agent/live_conversation.py \
    --scenario {encoder,causal,embedding,api} \
    --model-id MODEL_ID \
    --device cuda \
    --out-dir results/verification/agent_live              # scripted: fixed three turns

python scripts/agent/summarize_runs.py --check
```

Without flags it opens the UI and records whatever you type, until you leave.
`--autonomous` asks only for model ids and evaluates each one without further questions.
`--scenario` plays the fixed three-turn script on a terminal in the same UI (add `--plain`
for raw text; a pipe gets raw text automatically). All three read the same
`BIASSCOPE_AGENT_*` variables.

Each run records every turn, every tool call with its arguments, `summarize_report`'s own
return value, and a check listing any figure in the agent's final message that appears in no
tool result. `summarize_runs.py` tabulates recorded runs from the library's output rather
than from the agent's prose, and `--check` recomputes whether each run covered every metric
it should have. Recorded runs are in `results/verification/agent_live/`.

Two further scripts render a set of runs: `scripts/agent/results_table.py` builds one table
(rows are models, columns are metrics) and `scripts/agent/render_transcripts.py` renders each
conversation verbatim with its dispatch log.

## Recommendation coverage

A run is checked against what the library recommended for that model: recommended, then
feedable (a dataset provider serves it and the backend has the access it needs), then
planned, run and scored. A run is **complete** when every feedable metric was scored and
nothing was scored that was never recommended. That is the check that catches an agent that
quietly runs fewer metrics than it should.
