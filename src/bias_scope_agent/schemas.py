"""Anthropic tool-use JSON schemas, one per callable tool. These are the
LLM-facing contract - `session` never appears here, since the harness (see
loop.py) supplies it when dispatching.
"""

from __future__ import annotations

from typing import Any, Dict, List

INSPECT_MODEL: Dict[str, Any] = {
    "name": "inspect_model",
    "description": (
        "Best-effort guess about a target model identifier (Hugging Face Hub id, "
        "local path, or API endpoint): whether it looks causal or encoder, whether "
        "it has an LM head, whether an endpoint is chat-formatted. Never a final "
        "decision - confirm with the user when confidence is low."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "identifier": {"type": "string", "description": "HF Hub id, local path, or API URL."},
            "api_key": {"type": "string"},
            "api_base": {"type": "string"},
        },
        "required": ["identifier"],
    },
}

CONSTRUCT_BACKEND: Dict[str, Any] = {
    "name": "construct_backend",
    "description": (
        "Build a backend for the user's target model and return an opaque handle. "
        "backend_kind is required when kind='huggingface' and must be 'causal' or "
        "'encoder' - ask the user if inspect_model was not confident. Never ask the "
        "user for an API key here - tell them to export the provider's standard "
        "environment variable (e.g. OPENAI_API_KEY) before this call; there is no "
        "api_key argument on purpose."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["huggingface", "litellm"]},
            "model_id": {"type": "string"},
            "backend_kind": {"type": "string", "enum": ["causal", "encoder"]},
            "dtype": {"type": "string", "enum": ["bf16", "fp32", "fp16"], "default": "bf16"},
            "device": {"type": "string"},
            "api_base": {"type": "string"},
        },
        "required": ["kind", "model_id"],
    },
}

RECOMMEND_METRICS_TOOL: Dict[str, Any] = {
    "name": "recommend_metrics_tool",
    "description": (
        "List bias metrics legal for this backend's access. Always call "
        "explain_exclusions_tool alongside this and show both to the user."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "backend_handle": {"type": "string"},
            "axis": {"type": "string"},
            "language": {"type": "string", "default": "en"},
            "family": {"type": "string"},
        },
        "required": ["backend_handle"],
    },
}

EXPLAIN_EXCLUSIONS_TOOL: Dict[str, Any] = {
    "name": "explain_exclusions_tool",
    "description": "Explain, per excluded metric, why it was excluded for this backend's access.",
    "input_schema": {
        "type": "object",
        "properties": {
            "backend_handle": {"type": "string"},
            "language": {"type": "string", "default": "en"},
        },
        "required": ["backend_handle"],
    },
}

PLAN_SUITE: Dict[str, Any] = {
    "name": "plan_suite",
    "description": (
        "Dry-run a plan for the requested metrics (no execution). Reports which "
        "metrics will need caller-supplied data - call request_missing_inputs for "
        "those before calling run_suite. Must be shown to the user and confirmed "
        "(via confirm_plan) before run_suite will be allowed to execute."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "backend_handle": {"type": "string"},
            "metric_names": {"type": "array", "items": {"type": "string"}},
            "axis": {"type": "string", "default": "gender"},
            "language": {"type": "string", "default": "en"},
        },
        "required": ["backend_handle", "metric_names"],
    },
}

REQUEST_MISSING_INPUTS: Dict[str, Any] = {
    "name": "request_missing_inputs",
    "description": (
        "Tell the user what data is needed for these metrics and end your turn to "
        "wait for their reply. Never fabricate or guess this data yourself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"metric_names": {"type": "array", "items": {"type": "string"}}},
        "required": ["metric_names"],
    },
}

CONFIRM_PLAN: Dict[str, Any] = {
    "name": "confirm_plan",
    "description": (
        "Confirm a plan_suite result after the user has seen it and replied "
        "affirmatively, in a later turn. Required before run_suite is allowed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"plan_id": {"type": "string"}},
        "required": ["plan_id"],
    },
}

RUN_SUITE: Dict[str, Any] = {
    "name": "run_suite",
    "description": (
        "Execute the confirmed plan and return a report handle. Blocked by the "
        "system unless a matching plan_suite result was shown and confirm_plan "
        "was called in a later turn - do not attempt this early."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "backend_handle": {"type": "string"},
            "metric_names": {"type": "array", "items": {"type": "string"}},
            "inputs_handles": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Handles from prepare_inputs - the preferred route: the data "
                    "stays server-side and cannot be altered in transit. Pass EVERY "
                    "handle for the evaluation in ONE run_suite call, so all the "
                    "metrics land in a single report and a single summary. One call "
                    "per dataset would produce one partial report each."
                ),
            },
            "inputs_handle": {
                "type": "string",
                "description": "A single prepared handle; inputs_handles is preferred.",
            },
            "inputs": {
                "type": "object",
                "description": (
                    "Only for data no dataset covers, e.g. items the user typed out "
                    "themselves. Per-metric kwargs keyed by metric name - one entry "
                    "per metric in metric_names. A flat dict of parameters is "
                    'rejected. Constructor arguments go under a nested "__init__" '
                    'key. Example: {"CrowSPairs": {"__init__": {"model_name": '
                    '"bert-base-uncased"}, "sentence_pairs": [["he is a doctor", '
                    '"she is a doctor"]]}}. Pass this or inputs_handle, never both.'
                ),
            },
            "axis": {"type": "string", "default": "gender"},
            "language": {"type": "string", "default": "en"},
            "seed": {"type": "integer", "default": 42},
        },
        "required": ["backend_handle", "metric_names"],
    },
}

SUMMARIZE_REPORT: Dict[str, Any] = {
    "name": "summarize_report",
    "description": (
        "Render a completed report. 'chat' is a concise fidelity-labelled summary "
        "for the conversation; 'markdown'/'html' produce a shareable document."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "report_handle": {"type": "string"},
            "format": {"type": "string", "enum": ["chat", "markdown", "html"], "default": "chat"},
        },
        "required": ["report_handle"],
    },
}

RECORD_FACT: Dict[str, Any] = {
    "name": "record_fact",
    "description": (
        "Remember something the user already told you (e.g. 'model_kind: causal') "
        "so it is not re-asked later in the conversation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
        "required": ["key", "value"],
    },
}

LIST_DATASETS: Dict[str, Any] = {
    "name": "list_datasets",
    "description": (
        "Evaluation datasets this harness can load by itself, and which metrics "
        "each one feeds. Always check this before asking the user for data: if a "
        "dataset covers the metric, use prepare_inputs instead of having anyone "
        "type items into the conversation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "metric_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional filter - only datasets feeding these metrics.",
            }
        },
    },
}

PREPARE_INPUTS: Dict[str, Any] = {
    "name": "prepare_inputs",
    "description": (
        "Load a named dataset server-side and get back an inputs_handle to pass "
        "to run_suite, plus provenance (source file, sha256, item counts). "
        "The data itself is never returned and must never be requested: passing "
        "evaluation items through this conversation has been observed to alter "
        "and drop them, which silently changes the score. This is the preferred "
        "way to supply metric data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "backend_handle": {"type": "string"},
            "dataset": {"type": "string", "description": "A name from list_datasets."},
            "metric_names": {"type": "array", "items": {"type": "string"}},
            "axis": {"type": "string", "default": "gender"},
            "limit": {
                "type": "integer",
                "description": "Optional cap on items. Omit to use the whole subset.",
            },
        },
        "required": ["backend_handle", "dataset", "metric_names"],
    },
}


TOOLS: List[Dict[str, Any]] = [
    INSPECT_MODEL,
    CONSTRUCT_BACKEND,
    RECOMMEND_METRICS_TOOL,
    EXPLAIN_EXCLUSIONS_TOOL,
    LIST_DATASETS,
    PREPARE_INPUTS,
    PLAN_SUITE,
    REQUEST_MISSING_INPUTS,
    CONFIRM_PLAN,
    RUN_SUITE,
    SUMMARIZE_REPORT,
    RECORD_FACT,
]
