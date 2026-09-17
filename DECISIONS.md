# DECISIONS

Design choices that shape the codebase, with the reason. Narrower than
`REVIEW_LATER.md`: this file records what the code *is*, that file records what
the maintainer should *check*. Newest at the bottom.

---

## 2026-08-22 · New dependencies: `pyyaml` and `pypdf`, dev extra only

Section 1 forbids new runtime dependencies without a note here.

Phase 0 introduces three YAML manifests (`sources/SOURCES.yaml`,
`validation/registry.yaml`, `verification/ledger.yaml`) and a source fetcher
that extracts PDF text. `pyyaml` and `pypdf` were added to
`[project.optional-dependencies] dev` **only**.

The installed library's runtime dependencies are unchanged: `numpy` and
`requests`. Nothing under `src/bias_scope/` imports either package; only
`scripts/sources/`, `scripts/verification/`, and CI do. JSON was considered and
rejected — the manifests are hand-edited, and every example in `PLAN.md` is
YAML. See `REVIEW_LATER.md` RL-003.

## 2026-08-22 · `ruff` gate scope: `src` and `tests`

The lint gate in CI is `ruff check src tests`, as `PLAN.md` Section 2 specifies.
`scripts/experiments/` is not linted; it holds the v0.1 reproduction drivers and
has 53 violations. The Phase 0 scripts under `scripts/sources/`,
`scripts/verification/`, and `scripts/validation/` are ruff-clean and should
stay that way. See `REVIEW_LATER.md` RL-009.

## 2026-08-22 · Evidence in `ledger.yaml` is verified, not trusted

`scripts/verification/render_ledger.py` checks every non-empty evidence value
before rendering it: a `path::test` must exist *and* be collected by pytest, a
bare path must exist, a `FILE.md#anchor` must contain the anchor, and
`exempt: reason` requires the reason. Anything else renders as `BROKEN` and the
script exits non-zero.

Section 1 makes the ledger the definition of "done", so a ledger that repeats
claims it has not checked would be worse than none at all.

## 2026-08-22 · Goldens regenerate only through a script that demands a reason

`scripts/verification/regen_golden.py` requires `--reason`, writes it into the
JSON as `regenerated_because`, prints the old and new score, and tells the
caller to add a `CHANGELOG` entry. `tests/golden/test_golden.py` fails if any
golden lacks that field, so hand-editing a golden to make a test pass is itself
a test failure.

## 2026-08-23 · `fairlearn` in the `dev` extra only

**Decision.** `fairlearn>=0.10.0` is added to the `dev` extra. It is **not** a
runtime dependency, and `bias_scope` never imports it.

**Why.** `DecodingTrust`'s own scorer calls
`fairlearn.metrics.demographic_parity_difference` and
`equalized_odds_difference` directly, so fairlearn *is* the reference
implementation for those two numbers. PLAN.md Section 1's light-core rule keeps
it out of the install, which means BiasScope reimplements them — and a
reimplementation has to be *shown* equal, not assumed equal.
`tests/oracles/test_fairlearn_oracle.py` compares the two across 2000 seeded
random draws and asserts exact agreement to 1e-12. Putting fairlearn in `dev`
is what makes that check run in CI rather than silently skipping.

**Consequence if removed.** The oracle test `importorskip`s and passes
vacuously; the equivalence claim in `docs/fidelity/decodingtrust.md` would then
rest on nothing a reader could re-run.

**Same pattern as** the existing scipy usage: a dev-only oracle for a formula
the library implements itself.

## 2026-09-12 · `anthropic` and `huggingface_hub` in a new `agent` extra

**Decision.** `anthropic>=0.40.0` and `huggingface_hub>=0.24.0` are added to a
new `[project.optional-dependencies] agent` extra (folded into `all`, like
every other extra). `src/bias_scope/` still depends on nothing but `numpy` and
`requests` — this extra is consumed only by the new `src/bias_scope_agent/`
package.

**Why.** `bias_scope_agent` is a single-LLM tool-calling agent (the `anthropic`
SDK drives its tool-use loop) that can optionally live-inspect a target model
identifier by fetching `config.json` from the Hugging Face Hub
(`huggingface_hub.hf_hub_download`, behind the `BIASSCOPE_AGENT_INSPECT_LIVE`
flag). Both imports are deferred to inside the specific functions that need
them (`bias_scope_agent.loop.AgentLoop.__init__`,
`bias_scope_agent.introspection._download_hf_config`), so importing
`bias_scope_agent` itself does not require either package unless a real
Anthropic client or a live Hub lookup is actually used — `AgentSession`,
`tools.py`, and the test suite all run with a `StubBackend` and no client at
all.

**Consequence if removed.** `bias_scope_agent.loop.AgentLoop()` (with no
`client=` override, provider left at the default `"anthropic"`) and
`bias_scope_agent.introspection.inspect_model(..., live=True)` on a Hub
identifier would raise `ImportError` naming the missing package. Every unit
test in `tests/test_bias_scope_agent/` passes a fake client or mocks the Hub
call, so the fast suite does not depend on this extra being installed;
`tests/integration/test_bias_scope_agent_tiny_model.py` does not need it
either (it exercises `HuggingFaceBackend`, not `huggingface_hub`, and uses
`WEAT` with raw embedding arrays rather than a live model call — see
`REVIEW_LATER.md` RL-041).

**Not litellm, for the agent LLM specifically.** `bias_scope` already depends
on `litellm` (the `llm` extra) for target-model backends, and
`bias_scope_agent.introspection` also uses it opportunistically to probe an
API-endpoint identifier. The agent LLM itself (the model running the
tool-calling loop, as opposed to the model being bias-tested) initially used
the `anthropic` SDK directly rather than going through litellm — see the
2026-09-14 entry below for how that decision was later revisited for
multi-provider support, and why litellm was still not the chosen path even
then.

*(2026-09-14 update: the anthropic-client construction logic referenced above
now lives in `bias_scope_agent/providers.py`'s `_build_anthropic_client`, not
directly in `AgentLoop.__init__` — see the entry below.)*

## 2026-09-14 · `openai` and `google-genai` in new provider-specific extras; a translation layer, not litellm

**Decision.** Two new optional extras: `agent-openai` (`openai>=1.0.0`) and
`agent-gemini` (`google-genai>=0.3.0`), both folded into `all` like every
other extra, neither added to the base `agent` extra (which stays
`anthropic` + `huggingface_hub` only, so installing `bias-scope[agent]` alone
does not pull in two more SDKs a Claude-only user never asked for).
`AgentConfig` gained a `provider: str = "anthropic"` field
(`BIASSCOPE_AGENT_PROVIDER` env var), and a new module,
`bias_scope_agent/providers.py`, holds one adapter class per provider
(`AnthropicProvider`, `OpenAIProvider`, `GeminiProvider`), each translating
`schemas.TOOLS` (Anthropic tool-use JSON shape, this package's one internal
representation) to that provider's own function-calling wire format, and
normalizing that provider's response back to a shared `NormalizedResponse`
shape. `AgentLoop` (`loop.py`) calls only `self.client.create(system=,
messages=)` and never branches on provider — every adapter presents the same
interface, proven identical in `tests/test_bias_scope_agent/test_providers.py`
via a parametrized scripted conversation run through all three real adapters.

**Why not litellm here either.** litellm already gives one interface across
providers, so it looks like the obvious tool for exactly this job. It was not
used for two reasons. First, litellm's tool-use response shape (through its
`completion()` unification) is still shaped like OpenAI's function-calling
convention regardless of the underlying provider, so using it would not avoid
writing a translation layer — it would only move where a Gemini-specific
formatting mismatch could surface, from bias_scope_agent's own code (visible
and tested here) into litellm's coverage of Gemini's tool-calling support
(external, and not audited by this project). Second, `bias_scope`'s own use
of litellm is for the model *being tested*, whose only contract with the rest
of the library is `Backend.generate(prompts) -> List[str]` — plain text in,
plain text out, no tool schemas involved. The agent LLM's contract is a full
tool-calling protocol (schemas in, structured tool calls out), a materially
different integration than the one litellm is already trusted for in this
codebase. Reusing "generate some text from a model" plumbing for "run a
tool-calling loop" would have been the wrong abstraction to reach for even if
it had been zero-cost.

**Consequence if removed.** Only the corresponding provider stops working:
`build_provider(AgentConfig(provider="openai"))` /
`(provider="gemini")` raise a `RuntimeError` naming the missing package and
the extra to install (mirroring the existing missing-`ANTHROPIC_API_KEY`
error), same pattern as the `anthropic` case above. The default
(`provider="anthropic"`) path and every existing test are unaffected —
`agent-openai`/`agent-gemini` are consumed only inside
`providers.py`'s `_build_openai_client`/`_build_gemini_client`, both lazily
imported.

**Known limitation, not yet exercised against the real APIs.** These two
adapters' wire-format translations (message shapes, tool-result correlation,
JSON-schema field stripping for Gemini) are grounded in each provider's
publicly documented function-calling contract and verified with hand-built
fakes matching that documented shape (`test_providers.py`), but have not been
run against a real OpenAI or Gemini API call in this session — no live
conversation test was added for either provider, matching the same
live-testing gap already logged for the Anthropic path. Treat both as
implemented-and-unit-tested, not yet field-verified.

## 2026-09-14 (later) · `local` agent-LLM provider reuses `agent-openai`, no new dependency

**Decision.** A fourth `AgentConfig.provider` value, `"local"`, was added for
running the agent LLM itself against a locally-served model rather than a
cloud API. `LocalProvider` (`providers.py`) subclasses `OpenAIProvider` and
overrides only client construction — it inherits `create()` (the message/tool
translation) unchanged, because Ollama, llama.cpp's server, LM Studio, and
vLLM all converged on the same OpenAI-compatible chat-completions wire
format. No new package extra: `local` reuses `agent-openai`'s `openai`
dependency purely as a generic HTTP client pointed at a different
`base_url`, never at `api.openai.com`.

**Why no required API key.** `BIASSCOPE_AGENT_LOCAL_API_KEY` defaults to a
placeholder string (`"local"`) rather than raising like the three cloud
providers do — requiring a real key for a request that never leaves
localhost would be pure friction. `BIASSCOPE_AGENT_LOCAL_BASE_URL` defaults
to Ollama's standard endpoint (`http://localhost:11434/v1`) as the single
most common self-hosted runner, fully overridable for llama.cpp/LM
Studio/vLLM/anything else exposing the same API shape.

**Consequence if removed.** `build_provider(AgentConfig(provider="local"))`
raises `RuntimeError` naming the missing `openai` package and the
`agent-openai` extra, same pattern as every other provider's missing-
dependency error.

**Known limitation.** `_DEFAULT_MODELS["local"]` guesses `"llama3.1"` as a
plausible default — this is a guess, not a discovery mechanism; the actual
correct value depends entirely on what model the user has pulled locally, so
`BIASSCOPE_AGENT_MODEL` will need overriding in most real setups. Not run
against a real local server in this session (no Ollama/llama.cpp instance
available) — same unverified-translation caveat as RL-042 for OpenAI/Gemini,
though the risk is lower here since the wire format is byte-identical to the
already-implemented `OpenAIProvider` path.

## 2026-09-14 (later still) · `construct_backend`'s schema drops `api_key` on purpose

**Decision.** The LLM-facing tool schema for `construct_backend`
(`schemas.py`) no longer has an `api_key` property, even though
`tools.construct_backend`'s Python signature still accepts one.
`api_base` stays in the schema.

**Why.** Everything the agent LLM can put in a tool call becomes part of
`self.messages` — the running transcript sent back to the agent LLM itself
as context on every later turn (see `providers.py`'s module docstring). If
`api_key` stayed a fillable argument, the natural failure mode is the agent
asking the user to paste their target model's API key into chat, which
would then sit in plaintext in that transcript for the rest of the
conversation and flow through to whichever cloud provider is acting as the
agent's own brain. `LiteLLMBackend.generate()` already calls
`litellm.completion(..., api_key=self.api_key, ...)`, and litellm itself
falls back to the provider's standard environment variable
(`OPENAI_API_KEY`, etc.) when `api_key` is `None` — confirmed by reading
`src/bias_scope/backends.py` directly. So the fix costs nothing
functionally: omitting `api_key` from the schema means the agent always
passes `None`, and litellm's own env-var fallback does the rest.
`system_prompt.py` was updated to state this explicitly, so the agent tells
the user to export the variable themselves rather than trying to route
around the missing schema field by asking for it in prose.

**Same pattern as** the confirm-before-run gate: made structurally
impossible rather than merely discouraged. The agent LLM cannot fill a
field that isn't in the schema it was given, the same way it cannot call
`run_suite` before `check_run_gate` allows it.

**Consequence if removed** (i.e. `api_key` added back to the schema): no
immediate breakage — `tools.construct_backend` already accepts the
parameter — but the structural guarantee is gone and enforcement reverts to
"the system prompt asks nicely," which is exactly the category of risk this
project has consistently avoided elsewhere (see RL-040).

## 2026-09-17 · Two more agent-LLM providers: `openrouter` (direct) and `litellm` (general), by request

**Decision.** Two more `AgentConfig.provider` values, requested directly
rather than found in review like the earlier round: `"openrouter"` and
`"litellm"`. Both are thin subclasses of `OpenAIProvider`, same pattern as
`LocalProvider` — no new translation logic, only client construction
differs.

- `OpenRouterProvider` points the existing OpenAI-shaped client at
  OpenRouter's own endpoint (`https://openrouter.ai/api/v1`), which is
  itself a plain OpenAI-compatible chat completions API — requires
  `OPENROUTER_API_KEY`, same fail-fast-with-a-clear-message pattern as
  every other cloud provider here.
- `LiteLLMProvider` routes through `litellm.completion()` directly instead
  of a hand-built per-provider adapter, via a small shim
  (`_wrap_litellm_client`) that makes `litellm.completion` answer to the
  same `.chat.completions.create(...)` shape `OpenAIProvider.create()`
  already calls — litellm's own response shape for `completion()` is
  already OpenAI's (that is the point of litellm), so no separate
  normalization was needed, just the shim. This gives access to whichever
  of litellm's 100+ supported providers the model string names, OpenRouter
  included via litellm's own `"openrouter/<slug>"` routing prefix — not
  just OpenRouter, since litellm's whole value is being provider-agnostic.

**Why two, not one.** They serve different needs: `openrouter` is the
simpler, single-purpose path (one key, no extra dependency beyond `openai`,
OpenRouter's own slug format) for someone who only wants OpenRouter.
`litellm` is the general escape hatch — anything litellm itself reaches,
using litellm's own conventions (routing-prefixed model strings, its own
per-provider environment-variable resolution) — useful for a provider not
worth a dedicated adapter, or for someone who already has litellm
infrastructure (e.g. a self-hosted litellm proxy) they want this to go
through.

**This revisits, not reverses, the earlier "why not litellm" decision**
(2026-09-14 entry above, made before multi-provider support existed at
all). That entry reasoned litellm wouldn't avoid writing a translation
layer for the *hand-built, first-class* providers (Anthropic/OpenAI/Gemini)
and that reusing "generate text" plumbing for "run a tool-calling loop"
would be the wrong abstraction for those. Neither point is actually
contradicted here: `LiteLLMProvider` is an *additional*, optional path
alongside the first-class adapters, not a replacement for them — and it
still needed its own real (if small) translation shim, confirming litellm's
tool-calling response shape needed adapting to this package's internal
representation just like every other provider's does, exactly as that
entry predicted.

**Why `LiteLLMProvider` has no eager API-key check, unlike every other
provider here.** The other four each check one specific, known environment
variable before doing anything else. litellm has no such single variable —
which one it needs depends on the model string's provider prefix
(`OPENROUTER_API_KEY` for `"openrouter/..."`, `ANTHROPIC_API_KEY` for
`"anthropic/..."`, and so on for the rest of its 100+ providers), and
replicating litellm's own prefix-to-variable resolution here would be
either incomplete or a maintenance burden tracking litellm's own provider
list. A missing/wrong key surfaces as litellm's own authentication error on
the first real call instead — later than the other providers' checks, but
not silently.

**Consequence if removed:** `build_provider(AgentConfig(provider=
"openrouter"))` / `(provider="litellm")` raise `RuntimeError` naming the
missing package/extra, same as every other provider. No effect on the
other four.

13 new tests (`test_providers.py`, `test_config.py`), ruff clean. Not run
against a real OpenRouter/litellm call in this session — same
unverified-against-a-real-API caveat as RL-042 for OpenAI/Gemini, logged
as RL-046.

