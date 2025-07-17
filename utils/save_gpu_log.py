# utils/save_gpu_log.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.session import SessionLocal
from models.gpu_log import GPULog

def save_gpu_log(endpoint: str, request_id: str, gpu_data: dict, stage: str):
    db = SessionLocal()
    try:
        log = GPULog(
            endpoint=endpoint,
            request_id=request_id,
            gpu_index=gpu_data.get("gpu_index"),
            mem_used_MB=gpu_data.get("mem_used_MB"),
            mem_total_MB=gpu_data.get("mem_total_MB"),
            gpu_util_percent=gpu_data.get("gpu_util_percent"),
            mem_util_percent=gpu_data.get("mem_util_percent"),
            stage=stage
        )
        db.add(log)
        db.commit()
    except Exception as e:
        print(f"❌ Failed to save GPU log: {e}")
    finally:
        db.close()
