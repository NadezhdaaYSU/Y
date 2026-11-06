# Y's FP32 GEMM with the workspace W (stream-K where the plan takes it) against the 3-parameter
# kernel and cuBLAS (TF32 off). Bits against an in-order fmaf oracle on several grids, W full of garbage.
import torch, cupy as cp, numpy as np, subprocess, sys, re, os
torch.backends.cuda.matmul.allow_tf32=False
torch.set_float32_matmul_precision("highest")
Y=os.environ.get("YBIN","/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y/target/release/Y")
ORACLE=cp.RawKernel(r'''
extern "C" __global__ void oracle(const float* A,const float* B,float* C,int M,int N,int K){
  int j=blockIdx.x*blockDim.x+threadIdx.x, i=blockIdx.y; if(j>=N) return;
  float acc=0.f; for(int k=0;k<K;k++) acc=fmaf(A[(size_t)i*K+k],B[(size_t)k*N+j],acc);
  C[(size_t)i*N+j]=acc; }''',"oracle")
def build(m,n,k,with_w):
    name=f"y{4 if with_w else 3}_{m}_{n}_{k}"
    params="A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>"+(", W: GlobalMemory<U64>" if with_w else "")
    open(name+".ysu","w").write(f"@tile({m}, {n}, {k})\nkernel {name}({params}) {{\n}}\n\nfn main() {{\n}}\n")
    r=subprocess.run([Y,name+".ysu","--emit-ptx","-o",name+".ptx"],capture_output=True,text=True)
    assert r.returncode==0, r.stdout[-2000:]+r.stderr[-2000:]
    ptx=open(name+".ptx").read()
    tile=re.search(r"tile `(\w+)`",ptx).group(1)
    blk=int(re.search(r"Launch: block \((\d+),1,1\)",ptx).group(1))
    g=re.search(r"Grid \((\d+), 1, 1\)",ptx) or re.search(r"\((\d+), 1, 1\) is one tile per CTA",ptx)
    grid=int(g.group(1)); sk="Stream-K" in ptx
    smem=int(re.search(r"Dynamic shared memory required: (\d+) bytes",ptx).group(1))
    f=cp.RawModule(path=os.path.abspath(name+".ptx")).get_function(name); f.max_dynamic_shared_size_bytes=smem
    return f,tile,blk,grid,smem,sk
def ev(): return torch.cuda.Event(enable_timing=True)
shapes=[tuple(map(int,s.split("x"))) for s in sys.argv[1:]]
torch.manual_seed(0)
import random
RNG=random.Random(0)
for (m,n,k) in shapes:
    f3,t3,b3,g3,s3,_=build(m,n,k,False); f4,t4,b4,g4,s4,sk=build(m,n,k,True)
    a=torch.randn(m,k,device="cuda"); b=torch.randn(k,n,device="cuda")
    oc=torch.empty(m,n,device="cuda"); ORACLE(((n+127)//128,m),(128,),(cp.asarray(a),cp.asarray(b),cp.asarray(oc),np.int32(m),np.int32(n),np.int32(k)))
    tiles=(m+127)//64*((n+255)//64)+64
    W=torch.randint(-2**62,2**62,(tiles,),dtype=torch.int64,device="cuda")   # garbage: the launch tags make clearing it unnecessary
    A_,B_,W_=cp.asarray(a),cp.asarray(b),cp.asarray(W)
    def y4(c,grid): f4((grid,1,1),(b4,1,1),(A_,B_,cp.asarray(c),W_),shared_mem=s4)
    def y3(c,grid=g3): f3((grid,1,1),(b3,1,1),(A_,B_,cp.asarray(c)),shared_mem=s3)
    exact={}
    for grid in sorted({g4,1,3,max(1,g4//2)}):
        c=torch.full((m,n),float("nan"),device="cuda"); y4(c,grid); torch.cuda.synchronize()
        exact[grid]=torch.equal(c.view(torch.int32),oc.view(torch.int32))
    c4=torch.full((m,n),float("nan"),device="cuda"); c3=torch.empty(m,n,device="cuda"); cb=torch.mm(a,b)
    runs={"cuBLAS":lambda: torch.mm(a,b,out=cb),"Y3":lambda: y3(c3),"Y4":lambda: y4(c4,g4)}
    reps=min(200,max(3,int(1e11/(2*m*n*k)))); best={v:1e9 for v in runs}
    # The order is shuffled every round: at the card's power cap a fixed order
    # handicaps whichever kernel runs last (4096^3, identical SASS: 2-5% in a
    # fixed order, 0.01% shuffled).
    for r in range(15):
        order=list(runs.items()); RNG.shuffle(order)
        for v,fn in order:
            s,e=ev(),ev(); s.record()
            for _ in range(reps): fn()
            e.record(); e.synchronize(); best[v]=min(best[v],s.elapsed_time(e)/reps)
    after=torch.equal(c4.view(torch.int32),oc.view(torch.int32))   # after hundreds of launches on one W
    print(f"{m}x{n}x{k} {t4:5s} {'stream-K grid '+str(g4) if sk else 'plain (W unused)':22s} bits {exact} after-timing {after} | "
          f"cuBLAS {best['cuBLAS']*1e3:8.1f} us  Y3 {best['cuBLAS']/best['Y3']:.3f}x  Y4 {best['cuBLAS']/best['Y4']:.3f}x  (Y4/Y3 time {best['Y4']/best['Y3']:.3f})", flush=True)
print(subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,power.draw,temperature.gpu","--format=csv,noheader"],capture_output=True,text=True).stdout.strip())
