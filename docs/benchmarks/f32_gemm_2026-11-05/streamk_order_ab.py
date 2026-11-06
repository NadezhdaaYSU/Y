# 4096^3, W unused: the 3- and 4-parameter kernels from the SAME ptxas (cubins), alternating which runs first.
import torch, cupy as cp, numpy as np, subprocess, re, os
torch.backends.cuda.matmul.allow_tf32=False
Y="/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y/target/release/Y"
m=n=k=4096
fs={}
for v in (3,4):
    name=f"ab{v}"; params="A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>"+(", W: GlobalMemory<U64>" if v==4 else "")
    open(name+".ysu","w").write(f"@tile({m}, {n}, {k})\nkernel {name}({params}) {{\n}}\n\nfn main() {{\n}}\n")
    subprocess.run([Y,name+".ysu","--emit-ptx","-o",name+".ptx"],check=True,capture_output=True)
    subprocess.run(["ptxas","-arch=sm_89","-o",name+".cubin",name+".ptx"],check=True)
    sass=subprocess.run(["cuobjdump","-sass",name+".cubin"],capture_output=True,text=True).stdout
    ptx=open(name+".ptx").read(); smem=int(re.search(r"required: (\d+) bytes",ptx).group(1)); blk=int(re.search(r"block \((\d+),1,1\)",ptx).group(1))
    f=cp.RawModule(path=os.path.abspath(name+".cubin")).get_function(name); f.max_dynamic_shared_size_bytes=smem
    fs[v]=(f,smem,blk,[l.split("/* 0x")[0].strip() for l in sass.splitlines() if l.strip().startswith("/*") and ";" in l])
print("SASS identical:", fs[3][3]==fs[4][3], len(fs[3][3]))
a=torch.randn(m,k,device="cuda"); b=torch.randn(k,n,device="cuda"); c=torch.empty(m,n,device="cuda"); w=torch.zeros(1024,dtype=torch.int64,device="cuda")
A_,B_,C_,W_=cp.asarray(a),cp.asarray(b),cp.asarray(c),cp.asarray(w)
grid=(512,1,1)
def run(v):
    f,smem,blk,_=fs[v]; args=(A_,B_,C_) if v==3 else (A_,B_,C_,W_); f(grid,(blk,1,1),args,shared_mem=smem)
fs[33]=fs[3]   # the same kernel again: the noise floor
def run(v):
    f,smem,blk,_=fs[v]; args=(A_,B_,C_) if v in (3,33) else (A_,B_,C_,W_); f(grid,(blk,1,1),args,shared_mem=smem)
import itertools, random
random.seed(0)
best={3:1e9,4:1e9,33:1e9}; allt={3:[],4:[],33:[]}
for r in range(30):
    order=[3,4,33]; random.shuffle(order)
    for v in order:
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(3): run(v)
        e.record(); e.synchronize(); t=s.elapsed_time(e)/3; best[v]=min(best[v],t); allt[v].append(t)
import statistics as st
for v,nm in [(3,"3-param"),(33,"3-param again"),(4,"4-param")]:
    print(f"{nm:14s} min {best[v]*1e3:7.1f} us  median {st.median(allt[v])*1e3:7.1f} us")
print(f"4/3 min {best[4]/best[3]:.4f} median {st.median(allt[4])/st.median(allt[3]):.4f} | 3again/3 min {best[33]/best[3]:.4f} median {st.median(allt[33])/st.median(allt[3]):.4f}")
