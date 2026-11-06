# In-order stream-K probe: a persistent grid of P CTAs splits the (tile, k-tile) units evenly.
# A tile cut between two CTAs is CONTINUED, not reduced: the first CTA stores its exact f32
# accumulators into C and raises a flag; the next loads them and carries the same fma chain on.
# So every output is still one in-order fma chain over k = 0..K-1 (checked bit for bit).
import torch, cupy as cp, numpy as np, subprocess, sys
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
HDR = r'''
__device__ __forceinline__ void cp4(float* s, const float* g){ unsigned a=(unsigned)__cvta_generic_to_shared(s); asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(a), "l"(g)); }
__device__ __forceinline__ void cp16(float* s, const float* g){ unsigned a=(unsigned)__cvta_generic_to_shared(s); asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(a), "l"(g)); }
__device__ __forceinline__ void commit(){ asm volatile("cp.async.commit_group;\n" ::); }
template<int W> __device__ __forceinline__ void waitg(){ asm volatile("cp.async.wait_group %0;\n" :: "n"(W)); }
__device__ __forceinline__ unsigned long long ld_acq(const unsigned long long* p){ unsigned long long v; asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel(unsigned long long* p, unsigned long long v){ asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory"); }
__device__ __forceinline__ unsigned long long gridtag(){ unsigned long long g; asm volatile("mov.u64 %0, %%gridid;" : "=l"(g)); return g<<32; }

template<int M,int N,int K,int BM,int BN,int BK,int WM,int WN,int TM,int TN,int S,int PADA,int G>
__device__ __forceinline__ void seg(const float* __restrict__ A,const float* __restrict__ B,float* __restrict__ C,
                                    int tile,int kt0,int kt1,bool loadacc,float* smem){
    constexpr int NT=(BM/WM)*(BN/WN)*32, LM=WM/TM, LN=WN/TN, SA=BM+PADA;
    constexpr int AN=BM*BK/NT, BNC=BK*BN/4/NT, NC=AN+BNC;
    constexpr int SZA=BK*SA, SZB=BK*BN, NBX=N/BN, NBY=M/BM;
    static_assert(LM*LN==32 && NT%BK==0 && NT%(BN/4)==0 && NC<=BK-1 && K%BK==0 && M%BM==0 && N%BN==0, "cfg");
    float* As=smem; float* Bs=smem+S*SZA;
    const int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
    const int wm=warp%(BM/WM), wn=warp/(BM/WM), lm=lane/LN, ln=lane%LN;
    int bx,by; { const int grp=tile/(G*NBX), first=grp*G, gm=min(NBY-first,G), r=tile-grp*(G*NBX); by=first+r%gm; bx=r/gm; }
    const int KT=kt1-kt0;
    const float* ga = A + (size_t)(by*BM + tid/BK)*K + kt0*BK + tid%BK;
    const float* gb = B + (size_t)(kt0*BK + tid/(BN/4))*N + bx*BN + (tid%(BN/4))*4;
    const int sa_off=(tid%BK)*SA + tid/BK, sb_off=(tid/(BN/4))*BN + (tid%(BN/4))*4;
    auto copy=[&](int c,int stage){
        if(c<AN) cp4(As+stage*SZA+sa_off+c*(NT/BK), ga+(size_t)c*(NT/BK)*K);
        else { int i=c-AN; cp16(Bs+stage*SZB+sb_off+i*(NT/(BN/4))*BN, gb+(size_t)i*(NT/(BN/4))*N); }
    };
    float acc[TM][TN];
    if(loadacc){
        #pragma unroll
        for(int ci=0;ci<TM/4;ci++)
            #pragma unroll
            for(int r=0;r<4;r++){ const int row=by*BM+wm*WM+ci*(LM*4)+lm*4+r;
                #pragma unroll
                for(int cj=0;cj<TN/4;cj++){ const int col=bx*BN+wn*WN+cj*(LN*4)+ln*4;
                    float4 v=__ldcg(reinterpret_cast<const float4*>(C+(size_t)row*N+col));
                    acc[ci*4+r][cj*4]=v.x; acc[ci*4+r][cj*4+1]=v.y; acc[ci*4+r][cj*4+2]=v.z; acc[ci*4+r][cj*4+3]=v.w; } }
    } else {
        #pragma unroll
        for(int i=0;i<TM;i++)
            #pragma unroll
            for(int j=0;j<TN;j++) acc[i][j]=0.f;
    }
    float a[2][TM], b[2][TN];
    const int fa=wm*WM+lm*4, fb=wn*WN+ln*4;
    auto frag=[&](int f,const float* as,const float* bs,int k){
        #pragma unroll
        for(int c=0;c<TM/4;c++){ float4 v=*reinterpret_cast<const float4*>(&as[k*SA+fa+c*(LM*4)]);
            a[f][c*4]=v.x;a[f][c*4+1]=v.y;a[f][c*4+2]=v.z;a[f][c*4+3]=v.w; }
        #pragma unroll
        for(int c=0;c<TN/4;c++){ float4 v=*reinterpret_cast<const float4*>(&bs[k*BN+fb+c*(LN*4)]);
            b[f][c*4]=v.x;b[f][c*4+1]=v.y;b[f][c*4+2]=v.z;b[f][c*4+3]=v.w; }
    };
    #pragma unroll
    for(int s=0;s<S-1;s++){
        if(s<KT){
            #pragma unroll
            for(int c=0;c<NC;c++) copy(c,s);
            ga+=BK; gb+=(size_t)BK*N;
        }
        commit();
    }
    int rs=0, ws=S-1;
    for(int kt=0;kt<KT;kt++){
        waitg<S-2>(); __syncthreads();
        const float* as=As+rs*SZA; const float* bs=Bs+rs*SZB;
        const bool more = kt+S-1<KT;
        frag(0,as,bs,0);
        #pragma unroll
        for(int k=0;k<BK;k++){
            if(k<BK-1) frag((k+1)&1,as,bs,k+1);
            #pragma unroll
            for(int c=0;c<NC;c++) if(c==k && more) copy(c,ws);
            if(k==BK-2){ if(more){ ga+=BK; gb+=(size_t)BK*N; } commit(); }
            #pragma unroll
            for(int i=0;i<TM;i++)
                #pragma unroll
                for(int j=0;j<TN;j++) acc[i][j]=fmaf(a[k&1][i],b[k&1][j],acc[i][j]);
        }
        rs = rs+1==S ? 0 : rs+1; ws = ws+1==S ? 0 : ws+1;
    }
    waitg<0>();
    #pragma unroll
    for(int ci=0;ci<TM/4;ci++)
        #pragma unroll
        for(int r=0;r<4;r++){ const int row=by*BM+wm*WM+ci*(LM*4)+lm*4+r;
            #pragma unroll
            for(int cj=0;cj<TN/4;cj++){ const int col=bx*BN+wn*WN+cj*(LN*4)+ln*4;
                *reinterpret_cast<float4*>(C+(size_t)row*N+col)=make_float4(acc[ci*4+r][cj*4],acc[ci*4+r][cj*4+1],acc[ci*4+r][cj*4+2],acc[ci*4+r][cj*4+3]); } }
}

// One full tile per CTA, grid-strided: the shape of Y's kernel today.
template<int M,int N,int K,int BM,int BN,int BK,int WM,int WN,int TM,int TN,int S,int PADA,int G>
__device__ __forceinline__ void plain(const float* A,const float* B,float* C){
    extern __shared__ __align__(16) float smem[];
    constexpr int T=(M/BM)*(N/BN), KT=K/BK;
    for(int t=blockIdx.x;t<T;t+=gridDim.x){ seg<M,N,K,BM,BN,BK,WM,WN,TM,TN,S,PADA,G>(A,B,C,t,0,KT,false,smem); __syncthreads(); }
}

// In-order stream-K over P = gridDim.x CTAs.
template<int M,int N,int K,int BM,int BN,int BK,int WM,int WN,int TM,int TN,int S,int PADA,int G>
__device__ __forceinline__ void streamk(const float* A,const float* B,float* C,unsigned long long* flags){
    const unsigned long long tag=gridtag();
    extern __shared__ __align__(16) float smem[];
    constexpr int T=(M/BM)*(N/BN), KT=K/BK;
    const long U=(long)T*KT, P=gridDim.x, i=blockIdx.x;
    const long s=i*U/P, e=(i+1)*U/P;
    if(s>=e) return;
    const int tf=(int)(s/KT), tl=(int)((e-1)/KT);
    const int k0f=(int)(s-(long)tf*KT), k1l=(int)(e-(long)tl*KT);
    auto run=[&](int t,int a,int b){
        const bool wait=a>0, pub=b<KT;
        // The flag holds the k-tile the tile's chain has reached, so a tile cut in three is unambiguous:
        // each continuation waits for exactly its own start. The last one returns the flag to 0.
        if(wait){ if(threadIdx.x==0){ unsigned it=0; while(ld_acq(&flags[t])!=(tag|(unsigned)a)){ if(++it>(1u<<27)) __trap(); } } __syncthreads(); __threadfence(); }
        seg<M,N,K,BM,BN,BK,WM,WN,TM,TN,S,PADA,G>(A,B,C,t,a,b,wait,smem);
        if(pub){ __threadfence(); __syncthreads(); if(threadIdx.x==0) st_rel(&flags[t],tag|(unsigned)b); }
        __syncthreads();
    };
    // segments with no predecessor first (the head of the last tile, then whole tiles), the continuation last
    if(tl!=tf || k0f==0){
        const int a_l = (tl==tf)? k0f : 0;
        if(a_l==0 && k1l<KT) run(tl,0,k1l);
    }
    for(int t=tf;t<=tl;t++){
        const int a=(t==tf)?k0f:0, b=(t==tl)?k1l:KT;
        if(a==0 && b==KT) run(t,0,KT);
    }
    if(k0f>0) run(tf,k0f,(tf==tl)?k1l:KT);
}
'''
CFG = {  # BM,BN,BK,WM,WN,TM,TN,S,PADA,G
 "mid":   (64,128,16,32,64,8,8,3,4,8),
 "small": (64,64,16,32,32,4,8,3,4,8),
 "large": (128,256,16,64,64,8,16,3,4,8),
}
# (shape, tile, [P values for stream-K])
RUNS=[]
for a in sys.argv[1:]:
    shp,tile,ps=a.split(":"); RUNS.append((tuple(map(int,shp.split("x"))),tile,[int(p) for p in ps.split(",")]))
src=HDR; KN={}
for (m,n,k),tile,ps in RUNS:
    BM,BN,BK,WM,WN,TM,TN,S,PADA,G=CFG[tile]; NT=(BM//WM)*(BN//WN)*32; tpl=f"<{m},{n},{k},{BM},{BN},{BK},{WM},{WN},{TM},{TN},{S},{PADA},{G}>"
    nm=f"plain_{tile}_{m}_{n}_{k}"; src+=f'extern "C" __global__ void __launch_bounds__({NT},1) {nm}(const float* A,const float* B,float* C){{ plain{tpl}(A,B,C); }}\n'
    nm2=f"sk_{tile}_{m}_{n}_{k}"; src+=f'extern "C" __global__ void __launch_bounds__({NT},1) {nm2}(const float* A,const float* B,float* C,unsigned long long* F){{ streamk{tpl}(A,B,C,F); }}\n'
    KN[(m,n,k,tile)]=(nm,nm2,NT,S*BK*(BM+PADA+BN)*4,(m//BM)*(n//BN))
mod=cp.RawModule(code=src,options=("-std=c++17",),backend="nvrtc")
ORACLE=cp.RawKernel(r'''
extern "C" __global__ void oracle(const float* A,const float* B,float* C,int M,int N,int K){
  int j=blockIdx.x*blockDim.x+threadIdx.x, i=blockIdx.y; if(j>=N) return;
  float acc=0.f; for(int k=0;k<K;k++) acc=fmaf(A[(size_t)i*K+k],B[(size_t)k*N+j],acc);
  C[(size_t)i*N+j]=acc; }''',"oracle")
def ev(): return torch.cuda.Event(enable_timing=True)
torch.manual_seed(0)
flags=torch.randint(-2**62,2**62,(1<<16,),dtype=torch.int64,device="cuda")  # garbage: tags make zeroing unnecessary
for (m,n,k),tile,ps in RUNS:
    nm,nm2,NT,smem,T=KN[(m,n,k,tile)]
    fp=mod.get_function(nm); fp.max_dynamic_shared_size_bytes=max(smem,64*1024)
    fs=mod.get_function(nm2); fs.max_dynamic_shared_size_bytes=max(smem,64*1024)
    a=torch.randn(m,k,device="cuda"); b=torch.randn(k,n,device="cuda")
    oc=torch.empty(m,n,device="cuda"); ORACLE(((n+127)//128,m),(128,),(cp.asarray(a),cp.asarray(b),cp.asarray(oc),np.int32(m),np.int32(n),np.int32(k)))
    A_,B_,F_=cp.asarray(a),cp.asarray(b),cp.asarray(flags)
    runs={"cuBLAS":(lambda c: torch.mm(a,b,out=c)), f"plain({T})":(lambda c: fp((T,1,1),(NT,1,1),(A_,B_,cp.asarray(c)),shared_mem=smem))}
    for P in ps:
        for one in ([False,True] if smem<50*1024 else [False]):
            sm=60*1024 if one else smem
            runs[f"sk P={P}{' 1/SM' if one else ''}"]=(lambda c,P=P,sm=sm: fs((P,1,1),(NT,1,1),(A_,B_,cp.asarray(c),F_),shared_mem=sm))
    outs={v:torch.full((m,n),float("nan"),device="cuda") for v in runs}
    for v,fn in runs.items(): fn(outs[v])
    torch.cuda.synchronize()
    exact={v:torch.equal(outs[v].view(torch.int32),oc.view(torch.int32)) for v in runs if v!="cuBLAS"}
    reps=min(200,max(3,int(1e11/(2*m*n*k)))); best={v:1e9 for v in runs}
    for r in range(11):
        for v,fn in runs.items():
            s,e=ev(),ev(); s.record()
            for _ in range(reps): fn(outs[v])
            e.record(); e.synchronize(); best[v]=min(best[v],s.elapsed_time(e)/reps)
    again={v:torch.equal(outs[v].view(torch.int32),oc.view(torch.int32)) for v in runs if v!="cuBLAS"}
    print(f"{m}x{n}x{k} {tile} ({T} tiles):", flush=True)
    for v in runs:
        print(f"   {v:16s} {best[v]*1e3:8.1f} us  {best['cuBLAS']/best[v]:.3f}x cuBLAS" + ("" if v=="cuBLAS" else f"  bits {exact[v]} {again[v]}"), flush=True)
print(subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,power.draw,temperature.gpu","--format=csv,noheader"],capture_output=True,text=True).stdout.strip())
