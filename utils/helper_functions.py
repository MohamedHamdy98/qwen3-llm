import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates)

def get_gpu_info(index=0):
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(index)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        utilization = nvmlDeviceGetUtilizationRates(handle)

        gpu_data = {
            "gpu_index": index,
            "mem_used_MB": mem_info.used // 1024**2,
            "mem_total_MB": mem_info.total // 1024**2,
            "gpu_util_percent": utilization.gpu,
            "mem_util_percent": utilization.memory
        }
        return gpu_data
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            nvmlShutdown()
        except:
            pass
