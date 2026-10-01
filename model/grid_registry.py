"""Repository-backed registry for numbered LTM and lunar polar grids."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


class GridFamily(str, Enum):
    """Explicit lunar grid families supported by the repository metadata."""

    LTM = "ltm"
    LPS_N = "lps_n"
    LPS_S = "lps_s"


@dataclass(frozen=True)
class GeographicCoverage:
    """A non-wrapping geographic coverage envelope in degrees."""

    south: float
    west: float
    north: float
    east: float

    def __post_init__(self) -> None:
        values = (self.south, self.west, self.north, self.east)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Grid geographic coverage must be finite.")
        if not -90 <= self.south < self.north <= 90:
            raise ValueError(
                "Grid geographic coverage latitude bounds are invalid."
            )
        if not -180 <= self.west < self.east <= 180:
            raise ValueError(
                "Grid geographic coverage longitude bounds are invalid."
            )


@dataclass(frozen=True)
class TileMatrixDefinition:
    """One zoom matrix from a repository TMS definition."""

    zoom_level: int
    cell_size: float
    origin_x: float
    origin_y: float
    tile_width: int
    tile_height: int
    matrix_width: int
    matrix_height: int

    def __post_init__(self) -> None:
        if self.zoom_level < 0:
            raise ValueError("Tile-matrix zoom level must be nonnegative.")
        if not math.isfinite(self.cell_size) or self.cell_size <= 0:
            raise ValueError("Tile-matrix cell size must be positive and finite.")
        if not all(math.isfinite(value) for value in (self.origin_x, self.origin_y)):
            raise ValueError("Tile-matrix origin must be finite.")
        if self.tile_width <= 0 or self.tile_height <= 0:
            raise ValueError("Tile dimensions must be positive.")
        if self.matrix_width <= 0 or self.matrix_height <= 0:
            raise ValueError("Tile-matrix dimensions must be positive.")


@dataclass(frozen=True)
class GridDefinition:
    """Grid-neutral metadata loaded from one repository TMS JSON file."""

    grid_id: str
    family: GridFamily
    definition_path: Path
    title: str
    crs_wkt: str
    geographic_coverage: GeographicCoverage
    tile_matrices: tuple[TileMatrixDefinition, ...]

    def __post_init__(self) -> None:
        if not self.grid_id:
            raise ValueError("Grid ID must not be empty.")
        if not self.crs_wkt.strip():
            raise ValueError(f"Grid {self.grid_id!r} has no CRS WKT.")
        if not self.tile_matrices:
            raise ValueError(f"Grid {self.grid_id!r} has no tile matrices.")
        zooms = tuple(matrix.zoom_level for matrix in self.tile_matrices)
        if len(set(zooms)) != len(zooms):
            raise ValueError(f"Grid {self.grid_id!r} has duplicate zoom levels.")

    @property
    def zoom_levels(self) -> tuple[int, ...]:
        return tuple(matrix.zoom_level for matrix in self.tile_matrices)

    def matrix(self, zoom_level: int) -> TileMatrixDefinition:
        for matrix in self.tile_matrices:
            if matrix.zoom_level == zoom_level:
                return matrix
        raise KeyError(
            f"Grid {self.grid_id!r} does not define zoom {zoom_level}."
        )


_LTM_GRID_ID = re.compile(
    r"^tms_LTM_(?P<grid_id>(?:[1-9]|[1-3][0-9]|4[0-5])[NS])RG[.]json$"
)
_BBOX = re.compile(
    r"BBOX\[\s*(?P<south>[-+0-9.eE]+)\s*,\s*"
    r"(?P<west>[-+0-9.eE]+)\s*,\s*"
    r"(?P<north>[-+0-9.eE]+)\s*,\s*"
    r"(?P<east>[-+0-9.eE]+)\s*\]"
)


def _grid_identity(path: Path) -> tuple[str, GridFamily]:
    match = _LTM_GRID_ID.fullmatch(path.name)
    if match:
        return match.group("grid_id"), GridFamily.LTM
    if path.name == "tms_LPS_NRG.json":
        return "LPS_N", GridFamily.LPS_N
    if path.name == "tms_LPS_SRG.json":
        return "LPS_S", GridFamily.LPS_S
    raise ValueError(f"Unrecognized repository TMS filename: {path.name!r}.")


def _coverage_from_wkt(crs_wkt: str, *, grid_id: str) -> GeographicCoverage:
    matches = tuple(_BBOX.finditer(crs_wkt))
    if not matches:
        raise ValueError(f"Grid {grid_id!r} CRS has no geographic BBOX.")
    values = {
        name: float(matches[-1].group(name))
        for name in ("south", "west", "north", "east")
    }
    return GeographicCoverage(**values)


def _tile_matrix(raw: Mapping[str, object], *, grid_id: str) -> TileMatrixDefinition:
    try:
        origin = raw["pointOfOrigin"]
        if not isinstance(origin, list) or len(origin) != 2:
            raise TypeError("pointOfOrigin must contain two values")
        return TileMatrixDefinition(
            zoom_level=int(raw["id"]),
            cell_size=float(raw["cellSize"]),
            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
            tile_width=int(raw["tileWidth"]),
            tile_height=int(raw["tileHeight"]),
            matrix_width=int(raw["matrixWidth"]),
            matrix_height=int(raw["matrixHeight"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Grid {grid_id!r} contains an invalid tile matrix: {exc}."
        ) from exc


def load_grid_definition(path: str | Path) -> GridDefinition:
    """Load and validate one repository TMS JSON definition."""
    definition_path = Path(path).resolve()
    grid_id, family = _grid_identity(definition_path)
    try:
        raw = json.loads(definition_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not load grid definition {definition_path}: {exc}."
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Grid definition {definition_path} must be an object.")
    try:
        crs_wkt = str(raw["crs"])
        raw_matrices = raw["tileMatrices"]
        if not isinstance(raw_matrices, list):
            raise TypeError("tileMatrices must be a list")
        matrices = tuple(
            sorted(
                (_tile_matrix(item, grid_id=grid_id) for item in raw_matrices),
                key=lambda matrix: matrix.zoom_level,
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Grid definition {definition_path} is invalid: {exc}."
        ) from exc
    return GridDefinition(
        grid_id=grid_id,
        family=family,
        definition_path=definition_path,
        title=str(raw.get("title", grid_id)),
        crs_wkt=crs_wkt,
        geographic_coverage=_coverage_from_wkt(crs_wkt, grid_id=grid_id),
        tile_matrices=matrices,
    )


def _definition_sort_key(definition: GridDefinition) -> tuple[int, int, int]:
    if definition.family is GridFamily.LTM:
        zone = int(definition.grid_id[:-1])
        hemisphere = 0 if definition.grid_id.endswith("N") else 1
        return 0, hemisphere, zone
    if definition.family is GridFamily.LPS_N:
        return 1, 0, 0
    return 2, 0, 0


class GridRegistry:
    """Immutable collection of canonical repository grid definitions."""

    DEFAULT_DIRECTORY = Path(__file__).resolve().parent.parent / "TMS" / "RG"

    def __init__(self, definitions: Iterable[GridDefinition]) -> None:
        ordered = tuple(sorted(definitions, key=_definition_sort_key))
        by_id: dict[str, GridDefinition] = {}
        for definition in ordered:
            if definition.grid_id in by_id:
                raise ValueError(f"Duplicate grid ID: {definition.grid_id!r}.")
            by_id[definition.grid_id] = definition
        if not by_id:
            raise ValueError("Grid registry must contain at least one definition.")
        self._definitions = ordered
        self._by_id = MappingProxyType(by_id)

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        require_repository_inventory: bool = True,
    ) -> GridRegistry:
        """Load all JSON definitions from a repository TMS directory."""
        root = Path(directory).resolve()
        paths = tuple(sorted(root.glob("*.json")))
        definitions = tuple(load_grid_definition(path) for path in paths)
        registry = cls(definitions)
        if require_repository_inventory:
            ltm_count = sum(
                item.family is GridFamily.LTM for item in registry.definitions
            )
            polar_ids = {
                item.grid_id
                for item in registry.definitions
                if item.family is not GridFamily.LTM
            }
            if ltm_count != 90 or polar_ids != {"LPS_N", "LPS_S"}:
                raise ValueError(
                    "Repository grid inventory must contain 90 numbered LTM "
                    "definitions plus LPS_N and LPS_S; found "
                    f"{ltm_count} LTM and {sorted(polar_ids)} polar grids."
                )
        return registry

    @property
    def definitions(self) -> tuple[GridDefinition, ...]:
        return self._definitions

    @property
    def grid_ids(self) -> tuple[str, ...]:
        return tuple(definition.grid_id for definition in self._definitions)

    def by_family(self, family: GridFamily) -> tuple[GridDefinition, ...]:
        return tuple(
            definition
            for definition in self._definitions
            if definition.family is family
        )

    def __contains__(self, grid_id: object) -> bool:
        return grid_id in self._by_id

    def __getitem__(self, grid_id: str) -> GridDefinition:
        try:
            return self._by_id[grid_id]
        except KeyError as exc:
            raise KeyError(f"Unknown lunar grid ID: {grid_id!r}.") from exc

    def __iter__(self) -> Iterator[GridDefinition]:
        return iter(self._definitions)

    def __len__(self) -> int:
        return len(self._definitions)


@lru_cache(maxsize=1)
def default_grid_registry() -> GridRegistry:
    """Return the validated registry for the cloned repository metadata."""
    return GridRegistry.from_directory(GridRegistry.DEFAULT_DIRECTORY)


__all__ = [
    "GeographicCoverage",
    "GridDefinition",
    "GridFamily",
    "GridRegistry",
    "TileMatrixDefinition",
    "default_grid_registry",
    "load_grid_definition",
]
