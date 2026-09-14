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
