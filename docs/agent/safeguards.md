# Safeguards

The properties that make an audit trustworthy are enforced by the surrounding code, not
left to the agent LLM. This page says what the harness guarantees and, just as
importantly, what it does not.

## The confirm-before-run gate

Running an evaluation can be slow, costly, or download a model, so the agent cannot start
one on its own.

The **tool dispatcher**, not the agent LLM, decides whether `run_suite` may execute:

1. Every call to `plan_suite` records the plan: the backend, the set of metrics, the axis
   and the language, together with the index of the user turn in which it was made.
2. `confirm_plan` succeeds only for a recorded plan, and only in a **strictly later user
   turn**. A plan and its confirmation cannot happen inside the same turn.
3. Before `run_suite` is invoked, the dispatcher requires that a confirmed plan covers the
   call's backend, axis and language, and that the requested metrics are a non-empty subset
   of that plan's metrics.
4. If any of that fails, the dispatcher returns an error to the agent LLM and the tool is
   never called.

A **subset** is allowed because a multi-dataset evaluation runs one narrower call per
prepared dataset, each covering only its own metrics. Running fewer metrics than were
approved is not an escalation. Running a metric that was never approved is refused, and so
is an empty set.

The check holds whether or not the system prompt is followed. The test suite proves it
with a spy that asserts the real function is never called when the gate rejects a call.

### What the gate does not do

The gate is deliberately **structural**. It guarantees that a plan was produced, that a
user message followed it, and that the agent explicitly confirmed. It cannot verify that
the plan was displayed faithfully, or that your reply was actually an agreement. That
judgment stays with the agent LLM, whose instructions treat a hedge, a question or silence
as a refusal to confirm. A phrase filter would be a coarser and easier-to-fool signal, so
the harness does not attempt one.

In [autonomous mode](running.md) the confirmation is sent on your behalf. The gate is
unchanged, but you have chosen to pre-approve.

## No invented inputs

A metric whose stimuli were not supplied is **skipped with a stated reason**, never run on
made-up data. The agent is told to ask you for missing data through `request_missing_inputs`,
which ends its turn. For the metrics a dataset covers, the agent does not ask at all: the
[harness loads the authors' files](datasets.md) and the items never pass through the
model.

A call that would produce an empty report is rejected, and a metric that declines to score
reports its own reason. A score is never reported for a metric that did not run.

## Recommendations follow the backend

Access is derived from the backend, not from what the agent believes. A causal LM is not
offered masked-LM metrics, and an encoder whose checkpoint has no masked-LM head is not
offered them either. That check reads the loaded weights and not only the config: a
checkpoint whose config claims a masked-LM architecture but ships no head weights (the
sentence-transformers `all-mpnet-base-v2` is one) is treated as having no head, because
scoring with a randomly initialized head would still return a number.

## Credentials

Everything the agent LLM sees or emits becomes part of the transcript that is sent back to
it on later turns, so a credential typed into the conversation is exposed to the provider.

- The `construct_backend` tool has **no `api_key` argument**, on purpose. The system prompt
  tells the agent never to ask you for a key: export the provider's standard variable
  (for example `OPENAI_API_KEY`) yourself and just name the model.
- The keys for the agent LLM itself are read from the environment and never enter the
  conversation.
- `inspect_model` still accepts optional `api_key` and `api_base` arguments, so that it can
  probe an API endpoint. That is the one place a key could enter a tool call, and the only
  protection there is the instruction not to ask for one. **Do not type a key into the
  conversation.** Use the environment variable.

## What is recorded

Every score the agent reports comes from the library's deterministic code, with its
fidelity label. A recorded run keeps every turn, every tool call with its arguments, the
library's own summary, and a check that lists any figure in the agent's final message that
appears in no tool result. See [running the agent](running.md#scripted-and-recorded-runs).

The agent's interpretive prose is not covered by any of this and is sometimes wrong where
its numbers are right. Trust the tables, and read the prose as commentary.
