import torch, sys
sys.path.insert(0, "/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y/python")
from y_lang.torch_interop import TorchKernel
out = torch.full((32,), -7, dtype=torch.int32, device="cuda")
k = TorchKernel.__new__(TorchKernel)
k.source, k.kernel_name, k.target_sm, k.ptx = None, "shfl_i32", "sm_89", open("shfl_i32.ptx").read()
k._module = k._function = None
k._ensure_loaded()
k.launch((1, 1, 1), (32, 1, 1), [out])
torch.cuda.synchronize()
print("lane:     ", list(range(8)))
print("got:      ", out[:8].tolist())
print("expected: ", [(t ^ 1) + 1000 for t in range(8)])
