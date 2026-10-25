"""versions.py OLD.json NEW.json: for every case in both (same dist and shape), whether the inputs are the
same bits and each column both ran gives the same output bits (SHA-256 of C)."""
import json, sys
old = {(c["dist"], c["shape"]): c for c in json.load(open(sys.argv[1]))["cases"]}
n = cols = 0
diff = []
for c in json.load(open(sys.argv[2]))["cases"]:
    o = old.get((c["dist"], c["shape"]))
    if not o:
        continue
    n += 1
    assert o["sha256_A"] == c["sha256_A"] and o["sha256_B"] == c["sha256_B"], (c["dist"], c["shape"])
    for col, v in c["columns"].items():
        if col in o["columns"]:
            cols += 1
            if v["sha256_C"] != o["columns"][col]["sha256_C"]:
                diff.append((c["dist"], c["shape"], col))
print(f"{n} cases, {cols} columns compared; inputs identical; outputs differing: {diff if diff else 'none'}")
