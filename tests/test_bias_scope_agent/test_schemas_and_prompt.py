from bias_scope_agent import tools
from bias_scope_agent.schemas import TOOLS
from bias_scope_agent.session import AgentSession
from bias_scope_agent.system_prompt import render_system_prompt

_TOOL_FUNCTION_NAMES = {
    "inspect_model",
    "construct_backend",
    "recommend_metrics_tool",
    "explain_exclusions_tool",
    "plan_suite",
    "request_missing_inputs",
    "confirm_plan",
    "run_suite",
    "summarize_report",
    "record_fact",
    # Data by reference, so evaluation items never cross the tool boundary
    # (datasets.py, REVIEW_LATER RL-053).
    "list_datasets",
    "prepare_inputs",
}


class TestToolSchemas:
    def test_every_tool_function_has_a_schema(self):
        schema_names = {schema["name"] for schema in TOOLS}
        assert schema_names == _TOOL_FUNCTION_NAMES

    def test_every_schema_has_the_anthropic_tool_use_shape(self):
        for schema in TOOLS:
            assert schema["name"]
            assert schema["description"]
            assert schema["input_schema"]["type"] == "object"
            assert "properties" in schema["input_schema"]

    def test_no_schema_mentions_session(self):
        for schema in TOOLS:
            assert "session" not in schema["input_schema"]["properties"]

    def test_construct_backend_requires_kind_and_model_id(self):
        schema = next(s for s in TOOLS if s["name"] == "construct_backend")
        assert set(schema["input_schema"]["required"]) == {"kind", "model_id"}

    def test_construct_backend_schema_never_exposes_api_key(self):
        # The target model's API key must never be an LLM-fillable tool
        # argument - it would then sit in the conversation transcript and be
        # sent to the agent LLM as context. tools.construct_backend still
        # accepts api_key for direct/programmatic callers; only the schema
        # the agent LLM sees is restricted. api_base stays exposed - it is
        # an endpoint URL, not a secret.
        schema = next(s for s in TOOLS if s["name"] == "construct_backend")
        assert "api_key" not in schema["input_schema"]["properties"]
        assert "api_base" in schema["input_schema"]["properties"]

    def test_run_suite_schema_names_match_tools_function_signature(self):
        import inspect

        schema = next(s for s in TOOLS if s["name"] == "run_suite")
        params = set(inspect.signature(tools.run_suite).parameters) - {"session"}
        assert set(schema["input_schema"]["properties"]) <= params


class TestRenderSystemPrompt:
    def test_mentions_the_hard_gate_and_fidelity_rules(self):
        session = AgentSession()
        prompt = render_system_prompt(session)
        assert "confirm_plan" in prompt
        assert "fidelity" in prompt.lower()
        assert "explain_exclusions_tool" in prompt

    def test_renders_recorded_facts_so_they_are_not_re_asked(self):
        session = AgentSession()
        session.facts["model_kind"] = "causal"
        prompt = render_system_prompt(session)
        assert "model_kind" in prompt
        assert "causal" in prompt

    def test_no_facts_yet_still_renders_a_valid_prompt(self):
        session = AgentSession()
        prompt = render_system_prompt(session)
        assert isinstance(prompt, str)
        assert prompt

    def test_states_that_only_unambiguous_confirmation_may_trigger_confirm_plan(self):
        # RL-040 Option A: the gate is structural only, so the prompt is the
        # one place that can ask the agent to actually judge the reply.
        session = AgentSession()
        prompt = render_system_prompt(session)
        assert "unambiguous" in prompt.lower()
        assert "hedge" in prompt.lower()

    def test_tells_the_agent_never_to_collect_the_target_models_api_key_in_chat(self):
        session = AgentSession()
        prompt = render_system_prompt(session).lower()
        assert "api key" in prompt
        assert "environment variable" in prompt
