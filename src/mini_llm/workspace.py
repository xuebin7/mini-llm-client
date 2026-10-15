from pathlib import Path


class Workspace:
    def __init__(self, root: str | Path):
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def resolve_path(self, path: str | Path) -> Path:
        resolved_path = (self._root / Path(path)).resolve()

        if not resolved_path.is_relative_to(self._root):
            raise ValueError(
                f"the path: {path} is not relative to the root: {self._root}"
            )

        return resolved_path
