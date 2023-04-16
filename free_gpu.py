import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cashe():
    print("Initial GPU usage")
    gpu_usage()

    print(torch.cuda.get_device_name(0))

    torch.cuda.empty_cache()
    
    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

free_gpu_cashe()