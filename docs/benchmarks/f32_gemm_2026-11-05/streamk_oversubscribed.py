# Grids larger than the CTAs that fit at once: correctness then rests on CTAs starting in index order.
import torch, cupy as cp, numpy as np, sys, time
sys.argv=[sys.argv[0]]; exec(open("streamk_bench.py").read().split("def ev():")[0])
torch.manual_seed(1)
for (m,n,k) in [(768,768,768),(3072,3072,3072),(1500,1500,1500),(512,768,768)]:
    f4,t4,b4,g4,s4,sk=build(m,n,k,True)
    a=torch.randn(m,k,device="cuda"); b=torch.randn(k,n,device="cuda")
    oc=torch.empty(m,n,device="cuda"); ORACLE(((n+127)//128,m),(128,),(cp.asarray(a),cp.asarray(b),cp.asarray(oc),np.int32(m),np.int32(n),np.int32(k)))
    W=torch.randint(-2**62,2**62,(4096,),dtype=torch.int64,device="cuda")
    res={}
    for grid in [2*g4+5, 4*g4, 1000]:
        c=torch.full((m,n),float("nan"),device="cuda"); t0=time.time()
        f4((grid,1,1),(b4,1,1),(cp.asarray(a),cp.asarray(b),cp.asarray(c),cp.asarray(W)),shared_mem=s4); torch.cuda.synchronize()
        res[grid]=(torch.equal(c.view(torch.int32),oc.view(torch.int32)), round((time.time()-t0)*1e3,1))
    print(f"{m}x{n}x{k} {t4} header grid {g4}: {res}", flush=True)
