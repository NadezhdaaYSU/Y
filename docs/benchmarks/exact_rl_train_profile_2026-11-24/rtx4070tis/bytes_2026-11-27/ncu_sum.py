import csv, sys, collections
rows = [l for l in open(sys.argv[1]) if l.startswith('"')]
r = list(csv.DictReader(rows))
def num(s):  # Turkish locale: '.' thousands, ',' decimal
    return float(s.replace(".", "").replace(",", "."))
scale_t = {"ns": 1e-3, "nsecond": 1e-3, "us": 1, "usecond": 1, "ms": 1e3, "msecond": 1e3}
scale_b = {"byte": 1e-6, "Kbyte": 1e-3, "Mbyte": 1, "Gbyte": 1e3}
ids = collections.defaultdict(dict); names = {}
for x in r:
    try: v = num(x["Metric Value"])
    except ValueError: continue
    m, u = x["Metric Name"], x["Metric Unit"]
    if m == "gpu__time_duration.sum": v *= scale_t[u]
    elif m.startswith("dram__bytes"): v *= scale_b[u]
    ids[x["ID"]][m] = v; names[x["ID"]] = (x["Kernel Name"], x["Grid Size"], x["Block Size"])
agg = collections.defaultdict(list)
for i, d in ids.items(): agg[names[i][0]].append(d)
out = []
for k, ds in agg.items():
    n = len(ds); t = sum(d["gpu__time_duration.sum"] for d in ds)
    mb = sum(d["dram__bytes_read.sum"] + d["dram__bytes_write.sum"] for d in ds)
    dr = sum(d["dram__throughput.avg.pct_of_peak_sustained_elapsed"] * d["gpu__time_duration.sum"] for d in ds) / t
    sm = sum(d["sm__throughput.avg.pct_of_peak_sustained_elapsed"] * d["gpu__time_duration.sum"] for d in ds) / t
    out.append((t, n, k, mb / n, dr, sm, t / n, mb / t * 1e-3))
out.sort(reverse=True)
print(f"{'sampled us':>10} {'n':>4} {'us/call':>8} {'MB/call':>8} {'GB/s':>6} {'DRAM%':>6} {'SM%':>5}  kernel")
for t, n, k, mb, dr, sm, per, gbs in out[:int(sys.argv[2]) if len(sys.argv) > 2 else 25]:
    print(f"{t:10.0f} {n:4d} {per:8.1f} {mb:8.2f} {gbs:6.0f} {dr:6.1f} {sm:5.1f}  {k[:90]}")
T = sum(o[0] for o in out)
print(f"all: {T/1000:.2f} ms over {sum(o[1] for o in out)} launches; time-weighted DRAM {sum(o[0]*o[4] for o in out)/T:.1f}%, SM {sum(o[0]*o[5] for o in out)/T:.1f}%")
