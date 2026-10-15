from mini_llm.tools.builtin.apply_patch import create_apply_patch_tool
from mini_llm.tools.builtin.git_diff import create_git_diff_tool
from mini_llm.tools.builtin.git_log import create_git_log_tool
from mini_llm.tools.builtin.git_status import create_git_status_tool
from mini_llm.tools.builtin.read_file import create_read_file_tool
from mini_llm.tools.builtin.run_test import create_run_test_tool
from mini_llm.tools.builtin.search_code import create_search_code_tool
from mini_llm.tools.builtin.write_file import create_write_file_tool
from mini_llm.tools.registry import ToolRegistry
from mini_llm.workspace import Workspace


def create_default_registry(
    workspace: Workspace,
) -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(create_read_file_tool(workspace))
    registry.register(create_write_file_tool(workspace))
    registry.register(create_apply_patch_tool(workspace))
    registry.register(create_run_test_tool(workspace))
    registry.register(create_git_diff_tool(workspace))
    registry.register(create_git_status_tool(workspace))
    registry.register(create_search_code_tool(workspace))
    registry.register(create_git_log_tool(workspace))

    return registry
