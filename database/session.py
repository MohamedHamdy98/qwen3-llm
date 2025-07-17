# db/session.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use your DB URI — can be sqlite:///gpu_logs.db or postgres://...
# Path to the SQLite DB inside ./database folder
DB_PATH = os.path.join("database", "gpu_db.sqlite")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
