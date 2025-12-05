"""Tests for core plugin system."""
import pytest
from unittest.mock import MagicMock, patch
from token_copilot.core.plugin import Plugin, PluginManager
from token_copilot.core.copilot import TokenCoPilot


class TestPlugin:
    """Tests for Plugin base class."""

    def test_plugin_is_abstract(self):
        """Test that Plugin cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Plugin()

    def test_plugin_attach_calls_on_attach(self):
        """Test that attach() calls on_attach()."""

        class MockPlugin(Plugin):
            def __init__(self):
                self.on_attach_called = False

            def on_attach(self):
                self.on_attach_called = True

        plugin = MockPlugin()
        copilot = MagicMock(spec=TokenCoPilot)

        plugin.attach(copilot)

        assert plugin.on_attach_called
        assert plugin.copilot == copilot

    def test_plugin_detach_calls_on_detach(self):
        """Test that detach() calls on_detach()."""

        class MockPlugin(Plugin):
            def __init__(self):
                self.on_detach_called = False

            def on_detach(self):
                self.on_detach_called = True

        plugin = MockPlugin()
        copilot = MagicMock(spec=TokenCoPilot)
        plugin.attach(copilot)

        plugin.detach()

        assert plugin.on_detach_called
        assert plugin.copilot is None

    def test_plugin_lifecycle_hooks_have_defaults(self):
        """Test that lifecycle hooks have default implementations."""

        class MinimalPlugin(Plugin):
            pass

        plugin = MinimalPlugin()

        # Should not raise
        plugin.on_attach()
        plugin.on_detach()
        plugin.on_llm_start({}, ["prompt"], "run-123")
        plugin.on_llm_end(MagicMock(), "run-123")
        plugin.on_cost_tracked("gpt-4", 100, 50, 0.01, {})


class TestPluginManager:
    """Tests for PluginManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.copilot = MagicMock(spec=TokenCoPilot)
        self.manager = PluginManager(self.copilot)

    def test_add_plugin(self):
        """Test adding a plugin."""

        class MockPlugin(Plugin):
            pass

        plugin = MockPlugin()
        self.manager.add_plugin(plugin)

        assert len(self.manager._plugins) == 1
        assert plugin.copilot == self.copilot

    def test_add_multiple_plugins(self):
        """Test adding multiple plugins."""

        class MockPlugin1(Plugin):
            pass

        class MockPlugin2(Plugin):
            pass

        plugin1 = MockPlugin1()
        plugin2 = MockPlugin2()

        self.manager.add_plugin(plugin1)
        self.manager.add_plugin(plugin2)

        assert len(self.manager._plugins) == 2

    def test_remove_plugin(self):
        """Test removing a plugin."""

        class MockPlugin(Plugin):
            pass

        plugin = MockPlugin()
        self.manager.add_plugin(plugin)
        self.manager.remove_plugin(plugin)

        assert len(self.manager._plugins) == 0
        assert plugin.copilot is None

    def test_trigger_on_llm_start(self):
        """Test triggering on_llm_start on all plugins."""

        class MockPlugin(Plugin):
            def __init__(self):
                self.on_llm_start_called = False

            def on_llm_start(self, serialized, prompts, run_id, **kwargs):
                self.on_llm_start_called = True

        plugin = MockPlugin()
        self.manager.add_plugin(plugin)

        self.manager.trigger("on_llm_start", {}, ["prompt"], "run-123")

        assert plugin.on_llm_start_called

    def test_trigger_on_llm_end(self):
        """Test triggering on_llm_end on all plugins."""

        class MockPlugin(Plugin):
            def __init__(self):
                self.on_llm_end_called = False

            def on_llm_end(self, response, run_id, **kwargs):
                self.on_llm_end_called = True

        plugin = MockPlugin()
        self.manager.add_plugin(plugin)

        self.manager.trigger("on_llm_end", MagicMock(), "run-123")

        assert plugin.on_llm_end_called

    def test_trigger_on_cost_tracked(self):
        """Test triggering on_cost_tracked on all plugins."""

        class MockPlugin(Plugin):
            def __init__(self):
                self.on_cost_tracked_called = False

            def on_cost_tracked(self, model, input_tokens, output_tokens, cost, metadata):
                self.on_cost_tracked_called = True

        plugin = MockPlugin()
        self.manager.add_plugin(plugin)

        self.manager.trigger("on_cost_tracked", "gpt-4", 100, 50, 0.01, {})

        assert plugin.on_cost_tracked_called

    def test_trigger_multiple_plugins(self):
        """Test that trigger calls all plugins."""

        class MockPlugin(Plugin):
            def __init__(self):
                self.on_llm_start_called = False

            def on_llm_start(self, serialized, prompts, run_id, **kwargs):
                self.on_llm_start_called = True

        plugin1 = MockPlugin()
        plugin2 = MockPlugin()

        self.manager.add_plugin(plugin1)
        self.manager.add_plugin(plugin2)

        self.manager.trigger("on_llm_start", {}, ["prompt"], "run-123")

        assert plugin1.on_llm_start_called
        assert plugin2.on_llm_start_called

    def test_get_all_plugins(self):
        """Test getting all plugins."""

        class MockPlugin(Plugin):
            pass

        plugin1 = MockPlugin()
        plugin2 = MockPlugin()

        self.manager.add_plugin(plugin1)
        self.manager.add_plugin(plugin2)

        plugins = self.manager.get_all_plugins()

        assert len(plugins) == 2
        assert plugin1 in plugins
        assert plugin2 in plugins
