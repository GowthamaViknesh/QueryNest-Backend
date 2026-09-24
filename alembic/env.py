"""Alembic environment: connects as the API's DB login and manages schema "app" only."""

from alembic import context
from sqlalchemy import create_engine

from querynest.appdb import SCHEMA, Base
from querynest.config import settings

target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to):
    # Only our own schema; never touch the sales tables
    if type_ == "table":
        return obj.schema == SCHEMA
    return True


def run_migrations_online() -> None:
    engine = create_engine(settings.app_db_url())
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            include_object=include_object,
            version_table_schema=SCHEMA,  # alembic's own bookkeeping table lives in "app" too
        )
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
