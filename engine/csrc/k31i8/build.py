"""Build (and import) the k31i8 extension: python3 build.py [BUILD_DIR]. JIT through torch.utils.cpp_extension,
sm_103a, sources in this directory."""
import os
import sys

from torch.utils.cpp_extension import load

here = os.path.dirname(os.path.abspath(__file__))
build = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "build")
os.makedirs(build, exist_ok=True)
m = load(name="k31i8", sources=[os.path.join(here, f) for f in ("bindings.cpp", "dense_i8.cu")],
         extra_include_paths=[here, os.environ.get("K31I8_INC", "/build/inc")],
         build_directory=build, verbose=True, extra_cflags=["-O3"],
         extra_cuda_cflags=["-O3", "-gencode", "arch=compute_103a,code=sm_103a", "-lineinfo"])
print("k31i8 built:", [n for n in dir(m) if not n.startswith("_")])
