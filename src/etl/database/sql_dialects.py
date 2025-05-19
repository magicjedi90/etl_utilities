from __future__ import annotations

import dataclasses as dataclasses
from typing import Callable


@dataclasses.dataclass(slots=True, frozen=True)
class SqlDialect:
    name: str
    opening_quote: str
    closing_quote: str
    datetime_type: str
    boolean_type: str
    maximum_varchar_length: int | None            # None == “unlimited”
    identity_fragment_function: Callable[[str], str]
    primary_key_fragment_function: Callable[[str, str], str]
    unique_key_fragment_function: Callable[[str, str], str]


mssql = SqlDialect(
    name="mssql",
    opening_quote="[",
    closing_quote="]",
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
)

mariadb = SqlDialect(
    name="mariadb",
    opening_quote="`",
    closing_quote="`",
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
)


postgres = SqlDialect(
    name='postgres',
    opening_quote='"',
    closing_quote='"',
    datetime_type='timestamptz',
    boolean_type='boolean',
    maximum_varchar_length=10485760,
    identity_fragment_function=lambda table: (
        f"id serial constraint pk_{table}_id primary key"
    ),
    primary_key_fragment_function=lambda table, column: (
        f" constraint pk_{table}_{column} primary key"
    ),
)