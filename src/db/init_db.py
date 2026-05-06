from db.database import engine, Base
from db import models  # noqa: F401


def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully.")


if __name__ == "__main__":
    init_db()