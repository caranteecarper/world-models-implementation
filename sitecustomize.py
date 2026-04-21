import os

# Windows Conda environments can load Intel OpenMP from both PyTorch and
# Conda's Library/bin. Without this, Jupyter kernels may abort with:
# "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
