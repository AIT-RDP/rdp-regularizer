"""Drop files the CPU Chronos image does not need at runtime.

Keeps ``torch/bin/torch_shm_manager``: importing torch fails without it.
"""
import os
import shutil
import sys

torch_root = os.path.join(
    sys.prefix,
    "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages",
    "torch",
)

for relative in ("include", "share", "test", "utils/hipify", "onnx"):
    shutil.rmtree(os.path.join(torch_root, relative), ignore_errors=True)

bin_dir = os.path.join(torch_root, "bin")
if os.path.isdir(bin_dir):
    for name in os.listdir(bin_dir):
        if name == "torch_shm_manager":
            continue
        path = os.path.join(bin_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

caches = []
for dirpath, dirnames, _filenames in os.walk(sys.prefix):
    if "__pycache__" in dirnames:
        caches.append(os.path.join(dirpath, "__pycache__"))
for path in caches:
    shutil.rmtree(path, ignore_errors=True)
