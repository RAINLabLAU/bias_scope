import pytest

from bias_scope_agent.registry import HandleRegistry


class TestHandleRegistry:
    def test_register_returns_a_handle_that_resolves_back_to_the_item(self):
        registry: HandleRegistry[str] = HandleRegistry()
        handle = registry.register("some-item")
        assert registry.get(handle) == "some-item"

    def test_distinct_items_get_distinct_handles(self):
        registry: HandleRegistry[str] = HandleRegistry()
        first = registry.register("a")
        second = registry.register("b")
        assert first != second
        assert registry.get(first) == "a"
        assert registry.get(second) == "b"

    def test_unknown_handle_raises_key_error_naming_the_handle(self):
        registry: HandleRegistry[str] = HandleRegistry()
        with pytest.raises(KeyError, match="does-not-exist"):
            registry.get("does-not-exist")

    def test_two_registries_are_independent(self):
        first_registry: HandleRegistry[str] = HandleRegistry()
        second_registry: HandleRegistry[str] = HandleRegistry()
        handle = first_registry.register("only-in-first")
        with pytest.raises(KeyError):
            second_registry.get(handle)
