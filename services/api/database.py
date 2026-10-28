"""Database engine and session management for the API.

The engine is built lazily from `settings` on first use, so tests can point
`settings` at another database and call `get_engine.cache_clear()`.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from libs.config import settings


class Base(DeclarativeBase):
    pass


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(settings.database_url, pool_pre_ping=True)


def get_db() -> Iterator[Session]:
    db = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)()
    try:
        yield db
    finally:
        db.close()
