from __future__ import annotations

import dataclasses as dataclasses
from typing import Callable


@dataclasses.dataclass(slots=True, frozen=True)
class SqlDialect:
    name: str
    opening_escape: str
    closing_escape: str
    datetime_type: str
    boolean_type: str
    maximum_varchar_length: int | None
    identity_fragment_function: Callable[[str], str]
    primary_key_fragment_function: Callable[[str, str], str]
    unique_key_fragment_function: Callable[[str, str], str]
    placeholder: str
    comment: str

    # Helper so Loader does not have to know quoting rules
    def escape(self, identifier: str) -> str:               # noqa: D401
        """Return correctly escaped identifier for this dialect."""
        return f"{self.opening_escape}{identifier}{self.closing_escape}"




mssql = SqlDialect(
    name="mssql",
    opening_escape="[",
    closing_escape="]",
    datetime_type="datetime2",
    boolean_type="bit",
    maximum_varchar_length=None,  # None means nvarchar(max) is allowed
    identity_fragment_function=lambda table: (
        f"id int identity constraint pk_{table}_id primary key"
    ),
    primary_key_fragment_function=lambda table, column: (
        f" constraint pk_{table}_{column} primary key"
    ),
    unique_key_fragment_function=lambda table, column: (
        f" constraint ak_{table}_{column} unique"
    ),
    placeholder="?",
    comment="--"
)

mariadb = SqlDialect(
    name="mariadb",
    opening_escape="`",
    closing_escape="`",
    datetime_type="datetime",
    boolean_type="bit",
    maximum_varchar_length=21844,
    identity_fragment_function=lambda table: (
        f"id int auto_increment, constraint pk_{table}_id primary key (id)"
    ),
    primary_key_fragment_function=lambda table, column: (
        f"constraint pk_{table}_{column} primary key ({column})"
    ),
    unique_key_fragment_function=lambda table, column: (
        f"constraint ak_{table}_{column} unique ({column})"
    ),
    placeholder="%s",
    comment="#"
)


postgres = SqlDialect(
    name='postgres',
    opening_escape='"',
    closing_escape='"',
    datetime_type='timestamptz',
    boolean_type='boolean',
    maximum_varchar_length=10485760,
    identity_fragment_function=lambda table: (
        f"id serial constraint pk_{table}_id primary key"
    ),
    primary_key_fragment_function=lambda table, column: (
        f" constraint pk_{table}_{column} primary key"
    ),
    unique_key_fragment_function=lambda table, column: (
        f" constraint ak_{table}_{column} unique"
    ),
    placeholder="%s",
    comment="--"
)