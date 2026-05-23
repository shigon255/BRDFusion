LD_DEBUG=libs python - <<'PY' 2>&1 | egrep 'libcuda|libnvoptix|ptxjit|rtcore|found'
import ctypes
for lib in ("libcuda.so.1","libnvoptix.so.1","libnvidia-ptxjitcompiler.so.1","libnvidia-rtcore.so.1"):
    try:
        ctypes.CDLL(lib)
        print("loaded", lib)
    except OSError as e:
        print("FAILED", lib, "->", e)
PY
