"""repro.py OLD.json NEW.json...: for every case in both, whether the inputs (SHA-256 of A and B) and
the outputs of the columns both have (Y_mma, Y_f32, cuBLAS, SGEMM, RN32) are the same bits."""
import json, sys
old = {(c["dist"], c["shape"]): c for c in json.load(open(sys.argv[1]))["cases"]}
n = same_in = 0
diff = []
for f in sys.argv[2:]:
    for c in json.load(open(f))["cases"]:
        o = old.get((c["dist"], c["shape"]))
        if not o:
            continue
        n += 1
        same_in += o["sha256_A"] == c["sha256_A"] and o["sha256_B"] == c["sha256_B"]
        for col in ("Y_mma", "Y_f32", "cuBLAS", "SGEMM", "RN32"):
            if col in c["columns"] and col in o["columns"] and o["columns"][col]["sha256_C"] != c["columns"][col]["sha256_C"]:
                diff.append((c["dist"], c["shape"], col))
print(f"{n} cases in both; inputs identical in {same_in}; outputs differing: {diff if diff else 'none'}")
