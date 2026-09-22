from mini_llm.tools.builtin.git_log import GIT_LOG_TOOL
from mini_llm.tools.builtin.read_file import READ_FILE_TOOL
from mini_llm.tools.builtin.search_code import SEARCH_CODE_TOOL
from mini_llm.tools.registry import ToolRegistry


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(READ_FILE_TOOL)
    registry.register(SEARCH_CODE_TOOL)
    registry.register(GIT_LOG_TOOL)

    return registry
