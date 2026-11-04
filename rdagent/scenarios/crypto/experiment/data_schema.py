"""The data the scenario tells the agent about, read from the schema files under the data root.

The data pipeline exports one JSON file per dataset (market.crypto.bar.v1.json): a slug, a version,
the columns with their types and the annotations that describe them. The files are the fact; this
module only reads and renders them, so adding a dataset is a file, not a code change.

Only the datasets in LAYOUTS are rendered from a full directory scan, because each also needs a
statement of how its rows are laid out on disk and how it meets the bars. A caller that passes
``available_schemas`` instead names the schema files to open, and no other file in the directory is read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rdagent.log import rdagent_logger as logger


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class DataSchema:
    slug: str
    version: int
    name: str
    description: str
    columns: tuple[Column, ...]


@dataclass(frozen=True)
class Layout:
    """Where a schema is stored and what its time column says."""

    directory: str
    """The dataset directory, relative to the data root."""

    kind: str
    """The kind of data, as the prompt names it."""

    partition: str
    """What one sub-directory of the dataset holds."""

    time: str
    """What the time column means, in words."""

    duckdb: str | None = None
    """A single DuckDB file holding the same rows, relative to the data root.

    The data root may keep a dataset in one DuckDB file instead of the csv.gz partitions; either
    location counts as "the rows are on disk".
    """


LAYOUTS: Mapping[str, Layout] = {
    "market.crypto.bar": Layout(
        directory="crypto/data/crypto_bar",
        kind="bar",
        partition="<instrument>-<bar type>",
        time=(
            "`ts_open` is the bar's open time in nanoseconds since the Unix epoch (UTC); a bar's "
            "timestamp is `ts_open`, not `ts_event`."
        ),
        duckdb="crypto/crypto_bar.duckdb",
    ),
    "market.crypto_funding_rate": Layout(
        directory="crypto/data/crypto_funding_rate",
        kind="funding rate",
        partition="<instrument>",
        time=(
            "`ts_event` is the settlement time in nanoseconds since the Unix epoch (UTC); a bar sees "
            "the latest settlement at or before its close."
        ),
        duckdb="crypto/crypto_funding_rate.duckdb",
    ),
}


def _rows_on_disk(data_root: Path, layout: Layout) -> bool:
    """Whether the dataset's rows are there, in either the csv.gz layout or a DuckDB file."""
    if (data_root / layout.directory).is_dir():
        return True
    return layout.duckdb is not None and (data_root / layout.duckdb).is_file()


def load_schemas(schema_dir: Path, available_schemas: Sequence[str] | None = None) -> dict[str, DataSchema]:
    """The latest version of each schema in schema_dir, by slug.

    ``available_schemas`` names the files to open, in order. Without it, every ``*.json`` in the
    directory is read. A file that does not parse or lacks a slug, a version or columns is skipped
    with a warning: the data pipeline may be writing the directory while the scenario renders.
    """
    found: dict[str, DataSchema] = {}
    if not schema_dir.is_dir():
        return found
    for path in _schema_paths(schema_dir, available_schemas):
        try:
            schema = _parse(path)
        except (OSError, KeyError, TypeError, ValueError) as error:
            logger.warning(f"Skipping the data schema {path}: {error}")
            continue
        if schema.slug not in found or schema.version > found[schema.slug].version:
            found[schema.slug] = schema
    return found


def render_data_schemas(data_root: Path, available_schemas: Sequence[str] | None = None) -> str:
    """The prompt section of the datasets under data_root, one block each.

    With ``available_schemas``, only those files are opened, in the order given, and a file whose
    slug has no layout is still described. Without it, every schema file is scanned and only the
    datasets in :data:`LAYOUTS` are rendered. A known layout is included only when its rows are on
    disk, so a schema the pipeline wrote before the rows does not reach the prompt.
    """
    schema_dir = data_root / "schemas"
    if not schema_dir.is_dir():
        # The prompt falls back to "no data schema is available", which reads like a missing setting;
        # say which directory was looked at so the two causes are distinguishable.
        logger.warning(f"No data schemas under {schema_dir}: the prompt will not describe any dataset.")
    schemas = load_schemas(schema_dir, available_schemas)
    blocks = []
    chosen = (
        [(schema, LAYOUTS.get(schema.slug)) for schema in schemas.values()]
        if available_schemas is not None
        else [(schemas[slug], LAYOUTS[slug]) for slug in LAYOUTS if slug in schemas]
    )
    for schema, layout in chosen:
        if layout is not None and not _rows_on_disk(data_root, layout):
            logger.warning(
                f"Skipping the data schema {schema.slug}: neither {data_root / layout.directory} "
                f"nor {data_root / layout.duckdb} is on disk"
            )
            continue
        blocks.append(_render(schema, layout))
    return "\n\n".join(blocks)


def _schema_paths(schema_dir: Path, available_schemas: Sequence[str] | None) -> list[Path]:
    """The schema files to open. A name has to be one file in ``schema_dir``, not a path."""
    if available_schemas is None:
        return sorted(schema_dir.glob("*.json"))
    paths = []
    for name in available_schemas:
        filename = name.strip()
        if not filename or Path(filename).name != filename:
            logger.warning(f"Skipping the data schema {name!r}: it is not a file name")
            continue
        path = schema_dir / filename
        if not path.is_file():
            logger.warning(f"Skipping the data schema {filename}: {path} is not on disk")
            continue
        paths.append(path)
    return paths


def _render(schema: DataSchema, layout: Layout | None) -> str:
    # The schema's own description is the pipeline's boilerplate ("Jarvis CryptoBar history."); the
    # prompt only keeps what helps write a factor, the columns and where the rows are.
    heading = f"{schema.name}, {layout.kind}" if layout is not None else schema.name
    lines = [
        f"### `{schema.slug}` ({heading})",
        "",
        "| Column | Type | Description |",
        "|---|---|---|",
    ]
    lines += [f"| `{column.name}` | {column.type} | {column.description} |" for column in schema.columns]
    # lines += [
    #     "",
    #     f"Rows: `<data root>/{layout.directory}/{layout.partition}/<start>_<end>.csv.gz`, "
    #     "a gzip CSV with these columns.",
    #     layout.time,
    # ]
    return "\n".join(lines)


def _parse(path: Path) -> DataSchema:
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns = tuple(
        Column(
            name=str(column["name"]),
            type=_type_of(column),
            description=_described(payload, str(column["name"])),
        )
        for column in payload["columns"]
    )
    return DataSchema(
        slug=str(payload["slug"]),
        version=int(payload["version"]),
        name=str(payload.get("name") or payload["slug"]),
        description=str(payload.get("description") or ""),
        columns=columns,
    )


def _described(payload: Mapping[str, object], name: str) -> str:
    annotations = payload.get("annotations") or {}
    fields = annotations.get("fields") if isinstance(annotations, Mapping) else None
    annotation = fields.get(name) if isinstance(fields, Mapping) else None
    if isinstance(annotation, Mapping):
        return str(annotation.get("description") or "")
    return ""


def _type_of(column: Mapping[str, object]) -> str:
    if column.get("type") == "decimal" and column.get("precision") is not None:
        return f"decimal({column['precision']},{column.get('scale', 0)})"
    return str(column.get("type") or "")


__all__ = ["Column", "DataSchema", "LAYOUTS", "Layout", "load_schemas", "render_data_schemas"]
