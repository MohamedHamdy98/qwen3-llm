# initialize_tables.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gpu_log import Base
from database.session import engine

Base.metadata.create_all(bind=engine)
