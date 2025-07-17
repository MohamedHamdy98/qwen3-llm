# models/gpu_log.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class GPULog(Base):
    __tablename__ = "gpu_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    endpoint = Column(String)
    request_id = Column(String)
    gpu_index = Column(Integer)
    mem_used_MB = Column(Float)
    mem_total_MB = Column(Float)
    gpu_util_percent = Column(Float)
    mem_util_percent = Column(Float)
    stage = Column(String)  # "before" or "after"
    timestamp = Column(DateTime, default=datetime.utcnow)
