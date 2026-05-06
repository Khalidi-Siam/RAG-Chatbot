from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config.settings import settings

engine = create_engine(settings.database_url, echo=False)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def get_db():
    """
    Dependency for FastAPI.
    Yields a DB session and closes it automatically.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()