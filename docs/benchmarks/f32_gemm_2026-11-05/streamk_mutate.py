import subprocess, shutil, os, time, sys
REPO="/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y"
SRC=REPO+"/src/ptx_emitter/f32_simt_gemm.rs"
BAK=os.path.abspath("f32_simt_gemm.rs.orig")
shutil.copy2(SRC,BAK)
M=[
 ("M1 continuation loads start at K tile 0", 'w(self, format!("mul.wide.u32 {}, {}, {};", off64, ka, t.bk * 4));', 'w(self, format!("mul.wide.u32 {}, {}, {};", off64, ka, 0));', False),
 ("M2 a continuation starts from zero", 'w(self, format!("@{} bra {};", pz, fresh));', 'w(self, format!("bra.uni {};", fresh));', False),
 ("M3 nfull ignores the K tail", 'w(self, format!("sub.u32 {}, {}, {};", nfull, nk, nfull));', 'w(self, format!("mov.u32 {}, {};", nfull, nk));', False),
 ("M4 split tiles not offset by the whole waves", 'w(self, format!("add.u32 {}, {}, {};", tf, tf, tdp));', '', False),
 ("M5 loop K limit ignores ka", 'w(self, format!("add.u32 {}, {}, {};", klim, ka, kt));', 'w(self, format!("mov.u32 {}, {};", klim, kt));', True),
 ("M6 no launch tag", 'w(self, format!("shl.b64 {}, {}, 32;", tag, tag));', 'w(self, format!("mov.u64 {}, 0;", tag));', False),
 ("M7 no barrier after the wait", 'self.ptx_buffer.push_str(&format!("{}:\\n", waited));\n        w(self, "bar.sync 0;".to_string());', 'self.ptx_buffer.push_str(&format!("{}:\\n", waited));', False),
 ("M8 loop loads past the segment", 'w(self, format!("sub.s32 {}, {}, {};", last, nk, s - 1));', 'w(self, format!("mov.s32 {}, {};", last, nk));', True),
 ("M9 prologue loads past the segment", 'w(self, format!("setp.le.u32 {}, {}, {};", pz, nk, st));', 'w(self, format!("setp.lt.u32 {}, {}, {};", pz, nk, st));', True),
 ("M10 a wait no producer satisfies (the 30 s trap)", 'w(self, format!("cvt.u64.u32 {}, {};", want, ska));', 'w(self, format!("mov.u64 {}, 0;", want));', False),
 ("M11 plan threshold 0.5", '(plain >= sk * 1.04).then_some(p as u32)', '(plain >= sk * 0.5).then_some(p as u32)', False),
 ("M12 whole tiles not grid-strided", 'w(self, format!("mad.lo.u32 {}, {}, {}, {};", tmp, np, j, pid));', 'w(self, format!("add.u32 {}, {}, {};", tmp, j, pid));', False),
 ("M13 publish the start, not the end", 'w(self, format!("cvt.u64.u32 {}, {};", v64, skb));', 'w(self, format!("cvt.u64.u32 {}, {};", v64, ska));', False),
]
only=sys.argv[1:]
def run(cmd,t):
    try:
        r=subprocess.run(cmd,cwd=REPO,capture_output=True,text=True,timeout=t); return r.returncode, r.stdout+r.stderr
    except subprocess.TimeoutExpired as e: return "timeout", ""
for name,old,new,san in M:
    if only and name.split()[0] not in only: continue
    s=open(BAK).read(); assert s.count(old)==1, (name, s.count(old))
    open(SRC,"w").write(s.replace(old,new)); os.utime(SRC,None)
    rc,out=run(["cargo","build","--release","--tests"],900)
    if rc!=0: print(f"{name}: BUILD FAILED", flush=True); continue
    t0=time.time()
    rc,out=run(["cargo","test","--release","--test","f32_simt_gemm","workspace","--","--test-threads=1"],900)
    verdict="KILLED" if rc!=0 else "survived"
    why=[l for l in out.splitlines() if "panicked" in l or "FAILED" in l or "failed" in l][:2]
    if rc==0 and san:
        rc2,out2=run(["cargo","test","--release","--test","f32_simt_gemm","compute_sanitizer","--","--test-threads=1"],1200)
        if rc2!=0: verdict="KILLED (sanitizer)"; why=[l for l in out2.splitlines() if "found something" in l or "panicked" in l][:2]
    print(f"{name}: {verdict} ({time.time()-t0:.0f}s) {why}", flush=True)
shutil.copy2(BAK,SRC); os.utime(SRC,None)
print("restored", flush=True)
