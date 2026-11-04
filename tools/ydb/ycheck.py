"""compute-sanitizer on one Y kernel launch, reported in Y's terms.

    ydb prog.ysu --gpu 'k<<<1, 32>>>(arange(32), zeros(32), 32)' --check memcheck

runs `ylaunch.py --plain` - the build WITHOUT -g, the optimised code with its
line table, which is what the sanitizer can attribute to Y lines (with the
debug build it cannot even name the kernel) - under `compute-sanitizer --tool
TOOL`, and prints what it found once per Y line: the line's source, how many
times and in which threads, and for a memory error the buffer element it
touched, `Out[32]`, rather than a device address.

    memcheck    out-of-bounds and misaligned accesses to global, shared and
                local memory (an index into a `GlobalMemory` is not checked
                by the compiler or at run time)
    racecheck   hazards on shared memory between threads no barrier orders
    initcheck   reads of global memory nothing wrote
    synccheck   barriers some threads of a block do not reach

A record this cannot read is printed as the sanitizer wrote it, never dropped.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
PREFIX = "========= "


def find_sanitizer():
    import shutil
    found = shutil.which("compute-sanitizer")
    if found:
        return found
    for d in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/opt/cuda", "/usr/local/cuda"):
        if d and os.path.isfile(os.path.join(d, "bin", "compute-sanitizer")):
            return os.path.join(d, "bin", "compute-sanitizer")
    return None


def records(text):
    """(records, other lines): each record is one unindented line of the
    sanitizer's and the indented lines after it, prefix removed - its blank
    `=========` lines do not separate the banner from the first report, so
    they cannot be what splits; other lines are the program's own output."""
    recs, cur, other = [], [], []
    for line in text.splitlines():
        if line.rstrip() == PREFIX.rstrip():
            if cur:
                recs.append(cur)
                cur = []
            continue
        if line.startswith(PREFIX):
            body = line[len(PREFIX):].rstrip()
            if cur and not body.startswith(" "):
                recs.append(cur)
                cur = []
            cur.append(body)
        else:
            if cur:
                recs.append(cur)
                cur = []
            other.append(line)
    if cur:
        recs.append(cur)
    return recs, other


_LOC = r"at (?:(?P<fn>[\w$.]+)\+0x[0-9a-f]+ in (?P<file>[^\s:]+):(?P<line>\d+)|0x[0-9a-f]+)"


def parse(rec):
    """A record as a dict with 'kind' and its fields, or None if it is not
    one this reads."""
    head = rec[0]
    m = re.match(r"Invalid __(global|shared|local)__ (read|write) of size (\d+) bytes?$", head) or \
        re.match(r"Uninitialized __(global)__ memory (read) of size (\d+) bytes?$", head)
    if m:
        r = {"kind": "uninit" if head.startswith("Uninit") else "invalid", "space": m.group(1),
             "access": m.group(2), "size": int(m.group(3))}
        for line in rec[1:]:
            if re.match(r"\s*at ", line):
                lm = re.search(_LOC, line)
                r["loc"] = (lm.group("file"), int(lm.group("line"))) if lm and lm.group("file") else None
            tm = re.match(r"\s*by thread \((\d+),(\d+),(\d+)\) in block \((\d+),(\d+),(\d+)\)", line)
            if tm:
                v = tuple(int(x) for x in tm.groups())
                r["thread"], r["block"] = v[:3], v[3:]
            am = re.match(r"\s*(?:Access to|Address) (0x[0-9a-f]+) (?:is (out of bounds|misaligned))?", line)
            if am:
                r["address"] = int(am.group(1), 16)
                r["why"] = am.group(2) or ""
            nm = re.match(r"\s*and is (\d+) bytes (after|before) the nearest allocation at (0x[0-9a-f]+) of size (\d+)",
                          line)
            if nm:
                r["nearest"] = (int(nm.group(3), 16), int(nm.group(4)))
        if "loc" not in r or "thread" not in r:
            return None
        return r
    m = re.match(r"(Error|Warning): Race reported between (Write|Read) access " + _LOC, head)
    if m and len(rec) >= 2:
        m2 = re.match(r"\s*and (Write|Read) access " + _LOC + r"(?: \[(?P<n>\d+) hazards?\])?", rec[1])
        if m2 and m.group("file") and m2.group("file"):
            return {"kind": "race", "level": m.group(1),
                    "first": (m.group(2), m.group("file"), int(m.group("line"))),
                    "second": (m2.group(1), m2.group("file"), int(m2.group("line"))),
                    "hazards": int(m2.group("n") or 1)}
        return None
    m = re.match(r"Barrier error detected\.(.*)$", head)
    if m:
        r = {"kind": "barrier", "what": m.group(1).strip()}
        for line in rec[1:]:
            lm = re.search(_LOC, line)
            if lm and lm.group("file"):
                r["loc"] = (lm.group("file"), int(lm.group("line")))
            tm = re.match(r"\s*by thread \((\d+),(\d+),(\d+)\) in block \((\d+),(\d+),(\d+)\)", line)
            if tm:
                v = tuple(int(x) for x in tm.groups())
                r["thread"], r["block"] = v[:3], v[3:]
        return r if "loc" in r and "thread" in r else None
    m = re.match(r"Program hit (\S+) \(error (\d+)\) due to \"(.*)\" on CUDA API call to (\w+)", head)
    if m:
        return {"kind": "api", "error": m.group(1), "why": m.group(3), "call": m.group(4)}
    if re.match(r"(ERROR|RACECHECK) SUMMARY:|COMPUTE-SANITIZER$|Target application returned an error$", head):
        return {"kind": "summary", "text": head}
    return None


def _source(files, name, line):
    path = files.get(name)
    if not path:
        return ""
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return ""
    return lines[line - 1].strip() if 0 < line <= len(lines) else ""


def _threads(ts):
    """A list of (x, y, z) threads, compactly: a run of a 1-D block as a range."""
    ts = sorted(set(ts), key=lambda t: (t[2], t[1], t[0]))
    if all(t[1] == 0 and t[2] == 0 for t in ts) and ts and ts[-1][0] - ts[0][0] == len(ts) - 1:
        if len(ts) == 1:
            return "thread (%d,0,0)" % ts[0][0]
        return "threads (%d,0,0)-(%d,0,0)" % (ts[0][0], ts[-1][0])
    shown = ", ".join("(%d,%d,%d)" % t for t in ts[:6])
    return "threads %s%s" % (shown, ", ... (%d in all)" % len(ts) if len(ts) > 6 else "")


def _element(addr, buffers, nearest):
    """`Name[i]` for a device address, and where it is relative to the buffer."""
    for b in buffers:
        lo, hi = b["address"], b["address"] + b["count"] * b["size"]
        if lo <= addr < hi:
            off = addr - lo
            return b, off // b["size"], off % b["size"], "inside"
    if nearest:
        base, _ = nearest
        for b in buffers:
            if b["address"] == base:
                off = addr - base
                return b, off // b["size"], off % b["size"], "past the end" if off >= 0 else "before the start"
    return None, None, None, None


def report(text, buffers, files, tool, launch):
    """The sanitizer's output `text` as a Y report: (report text, errors)."""
    recs, other = records(text)
    groups, order, verbatim, api, summary = {}, [], [], [], []
    for rec in recs:
        r = parse(rec)
        if r is None:
            verbatim.append(rec)
            continue
        if r["kind"] == "summary":
            if "SUMMARY" in r["text"]:
                summary.append(r["text"])
            continue
        if r["kind"] == "api":
            api.append(r)
            continue
        if r["kind"] == "race":
            key = ("race", r["level"], r["first"], r["second"])
        elif r["kind"] == "barrier":
            key = ("barrier", r["what"], r["loc"])
        else:
            key = (r["kind"], r["space"], r["access"], r["size"], r.get("why"), r["loc"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    out = ["compute-sanitizer %s on %s, the build without -g:" % (tool, launch)]
    if not order and not verbatim and not api:
        out.append("")
        out.append("  nothing found")
    for key in order:
        rs = groups[key]
        out.append("")
        if key[0] == "race":
            _, level, (a1, f1, l1), (a2, f2, l2) = key
            n = sum(r["hazards"] for r in rs)
            out.append("%s:%d  %s" % (f1, l1, _source(files, f1, l1)))
            if (f2, l2) != (f1, l1):
                out.append("%s:%d  %s" % (f2, l2, _source(files, f2, l2)))
            out.append("  %s: a race on shared memory between the %s on line %d and the %s on line %d, "
                       "%d hazard%s: no barrier orders them" % (level.lower(), a1.lower(), l1, a2.lower(), l2,
                                                                 n, "" if n == 1 else "s"))
            continue
        if key[0] == "barrier":
            f, l = key[2]
            out.append("%s:%d  %s" % (f, l, _source(files, f, l)))
            blocks = sorted(set(r["block"] for r in rs))
            out.append("  barrier error%s: %s, %s of block%s %s" % (
                "" if len(rs) == 1 else "s", key[1], _threads([r["thread"] for r in rs]),
                "" if len(blocks) == 1 else "s", ", ".join("(%d,%d,%d)" % b for b in blocks)))
            continue
        kind, space, access, size, why, (f, l) = key
        out.append("%s:%d  %s" % (f, l, _source(files, f, l)))
        blocks = sorted(set(r["block"] for r in rs))
        what = "read of memory nothing wrote" if kind == "uninit" else \
            "%s %s %s" % ("misaligned" if why == "misaligned" else "out-of-bounds", space, access)
        out.append("  %s, %d byte%s, %d time%s: %s of block%s %s" % (
            what, size, "" if size == 1 else "s", len(rs), "" if len(rs) == 1 else "s",
            _threads([r["thread"] for r in rs if r["block"] == blocks[0]]) +
            (" (and %d other blocks)" % (len(blocks) - 1) if len(blocks) > 1 else ""),
            "" if len(blocks) == 1 else "s", "(%d,%d,%d)" % blocks[0]))
        where = {}
        for r in rs:
            if "address" not in r:
                continue
            b, i, rem, rel = _element(r["address"], buffers, r.get("nearest"))
            if b is None:
                where.setdefault(("?", None), []).append(r["address"])
            else:
                where.setdefault((b["name"], rel), []).append((i, rem, b))
        for (name, rel), hits in where.items():
            if name == "?":
                addrs = sorted(hits)
                out.append("  at %s: in no buffer this launch passed" % (
                    "%#x" % addrs[0] if len(addrs) == 1 else "%#x .. %#x" % (addrs[0], addrs[-1])))
                continue
            idx = sorted(set(i for i, _, _ in hits))
            b = hits[0][2]
            span = "%s[%d]" % (name, idx[0]) if len(idx) == 1 else "%s[%d] .. %s[%d]" % (name, idx[0], name, idx[-1])
            odd = " (not on an element boundary)" if any(rem for _, rem, _ in hits) else ""
            if rel == "inside":
                out.append("  at %s%s, inside %s (%d x %s)" % (span, odd, name, b["count"], b["type"]))
            else:
                out.append("  at %s%s: %s of %s (%d x %s)" % (span, odd, rel, name, b["count"], b["type"]))
    for r in api:
        out.append("")
        out.append("The kernel then failed: %s (%s) on %s." % (r["error"], r["why"], r["call"]))
    for rec in verbatim:
        out.append("")
        out.append("Reported as the sanitizer wrote it (not one this report reads):")
        out.extend("  " + l for l in rec)
    if summary:
        out.append("")
        out.extend(summary)
    # The sanitizer's own count, held against what this read: a report that
    # missed one must not read as clean.
    counted = None
    for s in summary:
        m = re.match(r"ERROR SUMMARY: (\d+) errors?", s) or \
            re.match(r"RACECHECK SUMMARY: \d+ hazards? displayed \((\d+) errors?, (\d+) warnings?\)", s)
        if m:
            counted = sum(int(g) for g in m.groups())
    read = sum(len(v) for v in groups.values()) + len(api) + len(verbatim)
    if counted is None or counted != read:
        out.append("")
        out.append("The sanitizer %s, and this report read %d: what it wrote, in full:" % (
            "counted %d" % counted if counted is not None else "printed no summary", read))
        out.extend("  " + l for l in text.splitlines())
    program = [l for l in other if l.strip()]
    if program:
        out.append("")
        out.append("The launch's own output:")
        out.extend(program)
    errors = len(recs) - sum(1 for rec in recs if (parse(rec) or {}).get("kind") == "summary")
    return "\n".join(out), errors


def source_files(ptx):
    """basename -> path for each `.file` the module names."""
    files = {}
    for m in re.finditer(r'^\.file\s+\d+\s+"((?:[^"\\]|\\.)*)"', ptx, re.M):
        path = m.group(1).replace('\\"', '"').replace("\\\\", "\\")
        files.setdefault(os.path.basename(path), path)
    return files


def run(sanitizer, tool, ptx_path, launch, work):
    """Run the launch under the sanitizer; (report, exit status)."""
    buffers_path = os.path.join(work, "buffers.json")
    cmd = [sanitizer, "--tool", tool, "--show-backtrace", "no", "--print-limit", "100000",
           "--error-exitcode", "3"]
    cmd += [sys.executable, os.path.join(HERE, "ylaunch.py"), "--plain", "--buffers", buffers_path,
            ptx_path, launch]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        with open(buffers_path) as f:
            buffers = json.load(f)
    except (OSError, ValueError):
        buffers = []
    with open(ptx_path) as f:
        files = source_files(f.read())
    text, errors = report(p.stdout, buffers, files, tool, launch)
    return text, (1 if errors or p.returncode != 0 else 0)
