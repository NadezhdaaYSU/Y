"""The debug information in a PTX module, read back.

`Y --emit-ptx -g` writes DWARF into `.section .debug_abbrev` / `.debug_info`
as `.b8` bytes and `.b32` / `.b64` references to labels and sections
(`src/ptx_debug_info.rs`). This reads it the way a debugger would: the abbrev
table first, then the DIE tree it describes. It reads nvcc `-G`'s output too.

    python3 tools/ydb/ydwarf.py prog.ptx          # every kernel, parameter and binding
    python3 tools/ydb/ydwarf.py --check prog.ptx  # hold the description to the code

A reference to a label is kept as its name; a location `DW_OP_regx` operand is
turned back into the PTX register it names.
"""
import json
import re
import sys

TAGS = {
    0x11: "compile_unit", 0x2e: "subprogram", 0x05: "formal_parameter", 0x34: "variable",
    0x24: "base_type", 0x0b: "lexical_block", 0x0f: "pointer_type", 0x3b: "unspecified_type",
    0x16: "typedef", 0x13: "structure_type", 0x0d: "member", 0x01: "array_type", 0x21: "subrange_type",
    0x26: "const_type", 0x35: "volatile_type", 0x10: "reference_type",
}
ATTRS = {
    0x03: "name", 0x25: "producer", 0x13: "language", 0x10: "stmt_list", 0x1b: "comp_dir",
    0x11: "low_pc", 0x12: "high_pc", 0x40: "frame_base", 0x2007: "linkage_name", 0x6e: "linkage_name",
    0x3a: "decl_file", 0x3b: "decl_line", 0x49: "type", 0x3f: "external", 0x02: "location",
    0x3e: "encoding", 0x0b: "byte_size", 0x33: "address_class", 0x87: "noreturn", 0x27: "prototyped",
    0x34: "artificial", 0x38: "data_member_location", 0x20: "inline",
}


class DwarfError(Exception):
    pass


def sections(ptx):
    """name -> list of items: an int byte, or (symbol, width) for a reference,
    followed by width - 1 None placeholders so offsets stay byte offsets."""
    out = {}
    for name, body in re.findall(r"\.section\s+(\.[\w.]+)\s*\{(.*?)\}", ptx, re.S):
        items = []
        for line in body.splitlines():
            m = re.match(r"\s*\.b(8|16|32|64)\s+(.*?)\s*$", line)
            if not m:
                continue
            width = int(m.group(1)) // 8
            for v in m.group(2).split(","):
                v = v.strip()
                if re.fullmatch(r"-?\d+", v):
                    n = int(v)
                    items += list(n.to_bytes(width, "little", signed=n < 0))
                elif re.fullmatch(r"0x[0-9a-fA-F]+", v):
                    items += list(int(v, 16).to_bytes(width, "little"))
                else:
                    items += [(v, width)] + [None] * (width - 1)
        out[name] = items
    return out


def _uleb(b, i):
    r = s = 0
    while True:
        x = b[i]
        if not isinstance(x, int):
            raise DwarfError("a reference inside a LEB128 number at %d" % i)
        i += 1
        r |= (x & 0x7F) << s
        s += 7
        if not x & 0x80:
            return r, i


def register_name(code):
    """The PTX register a `DW_OP_regx` operand names: its ASCII bytes."""
    raw = code.to_bytes(8, "big").lstrip(b"\0")
    try:
        name = raw.decode("ascii")
    except UnicodeDecodeError:
        return None
    return name if re.fullmatch(r"%[a-z]+\d*", name) else None


def _location(block):
    """A location expression as ('reg', name), ('pieces', [(name, bytes)]) -
    a value whose parts are in registers of their own - ('addr', symbol) or
    ('expr', bytes)."""
    if block and block[0] == 0x90 and all(isinstance(x, int) for x in block):
        code, end = _uleb(block, 1)
        if end == len(block):
            return ("reg", register_name(code))
    if block and block[0] in (0x90, 0x93) and all(isinstance(x, int) for x in block):
        # Each piece: DW_OP_regx REG DW_OP_piece SIZE, or DW_OP_piece SIZE
        # alone for a piece no register holds (None).
        pieces, i = [], 0
        while i < len(block):
            reg = None
            if block[i] == 0x90:
                code, i = _uleb(block, i + 1)
                reg = register_name(code)
            if i >= len(block) or block[i] != 0x93:
                break
            size, i = _uleb(block, i + 1)
            pieces.append((reg, size))
        if pieces and i == len(block):
            return ("pieces", pieces)
    if block and block[0] == 0x03 and isinstance(block[1], tuple):
        return ("addr", block[1][0])
    return ("expr", [x if not isinstance(x, tuple) else x[0] for x in block if x is not None])


def parse(ptx):
    """The DIE tree: a list of compile units, each {'tag', 'offset', 'attrs',
    'children'}; references (`type`) are DIE offsets within the unit."""
    secs = sections(ptx)
    if ".debug_info" not in secs or ".debug_abbrev" not in secs:
        raise DwarfError("the module has no .debug_info / .debug_abbrev: it was not compiled with -g")
    ab = secs[".debug_abbrev"]
    abbrevs, i = {}, 0
    while i < len(ab):
        code, i = _uleb(ab, i)
        if code == 0:
            break
        tag, i = _uleb(ab, i)
        children = ab[i]
        i += 1
        attrs = []
        while True:
            a, i = _uleb(ab, i)
            f, i = _uleb(ab, i)
            if a == 0 and f == 0:
                break
            attrs.append((a, f))
        abbrevs[code] = (tag, children, attrs)

    b = secs[".debug_info"]
    units, i = [], 0
    while i < len(b):
        start = i

        def fixed(n):
            nonlocal i
            x = b[i]
            if isinstance(x, tuple):
                v = x[0]
            else:
                if any(not isinstance(y, int) for y in b[i:i + n]):
                    raise DwarfError("a reference that does not start where a %d-byte value does, at %d" % (n, i))
                v = int.from_bytes(bytes(b[i:i + n]), "little")
            i += n
            return v

        length = fixed(4)
        end = i + length
        version = fixed(2)
        fixed(4)  # the abbrev table's offset: a reference to .debug_abbrev
        addr = fixed(1)
        if version not in (2, 3, 4) or addr != 8:
            raise DwarfError("DWARF version %s, address size %s: not what PTX carries" % (version, addr))
        stack, roots = [], []
        while i < end:
            off = i - start
            code, i = _uleb(b, i)
            if code == 0:
                if stack:
                    stack.pop()
                continue
            if code not in abbrevs:
                raise DwarfError("DIE at %d uses abbrev %d, which the table does not define" % (off, code))
            tag, children, attrs = abbrevs[code]
            die = {"tag": TAGS.get(tag, hex(tag)), "offset": off, "attrs": {}, "children": []}
            for a, f in attrs:
                if f == 0x08:  # string
                    j = i
                    while b[j] != 0:
                        j += 1
                    v = bytes(b[i:j]).decode("utf-8", "replace")
                    i = j + 1
                elif f == 0x01:
                    v = fixed(8)
                elif f in (0x0b, 0x0c, 0x11):
                    v = fixed(1)
                elif f in (0x05, 0x12):
                    v = fixed(2)
                elif f in (0x06, 0x13, 0x10, 0x17):
                    v = fixed(4)
                elif f in (0x07, 0x14):
                    v = fixed(8)
                elif f == 0x0a:
                    n = b[i]
                    i += 1
                    v = _location(b[i:i + n]) if a == 0x02 else b[i:i + n]
                    i += n
                elif f == 0x09:
                    n, i = _uleb(b, i)
                    v = _location(b[i:i + n]) if a == 0x02 else b[i:i + n]
                    i += n
                elif f == 0x0f:
                    v, i = _uleb(b, i)
                elif f == 0x19:
                    v = True
                else:
                    raise DwarfError("form %#x at %d is not one PTX debug information uses" % (f, off))
                die["attrs"][ATTRS.get(a, hex(a))] = v
            (stack[-1]["children"] if stack else roots).append(die)
            if children:
                stack.append(die)
        if stack:
            raise DwarfError("a DIE's children are not terminated")
        units.append(roots)
        i = end
    return units


def _index(die, out):
    out[die["offset"]] = die
    for c in die["children"]:
        _index(c, out)


def type_of(dies, off):
    """A type DIE as {'kind', 'name', ...}."""
    d = dies.get(off)
    if d is None:
        return {"kind": "missing", "name": "?"}
    a = d["attrs"]
    if d["tag"] == "base_type":
        return {"kind": "base", "name": a.get("name"), "encoding": a.get("encoding"), "size": a.get("byte_size")}
    if d["tag"] == "pointer_type":
        target = type_of(dies, a["type"]) if "type" in a else {"kind": "void", "name": "void"}
        return {"kind": "pointer", "name": "*" + (target.get("name") or "?"), "target": target}
    if d["tag"] == "structure_type":
        fields = [{"name": m["attrs"].get("name"), "type": type_of(dies, m["attrs"].get("type"))}
                  for m in d["children"] if m["tag"] == "member"]
        return {"kind": "struct", "name": a.get("name"), "size": a.get("byte_size"), "fields": fields}
    return {"kind": d["tag"], "name": a.get("name", "?")}


def kernels(ptx):
    """kernel name -> {'line', 'params': [...], 'vars': [...]}: each a dict
    with 'name', 'line', 'type' and 'reg' (None: no location), and for a
    binding 'scope' (the begin and end labels of its block) and 'depth'."""
    out = {}
    for unit in parse(ptx):
        dies = {}
        for d in unit:
            _index(d, dies)
        for cu in unit:
            for sp in cu["children"]:
                if sp["tag"] != "subprogram":
                    continue
                a = sp["attrs"]
                k = {"line": a.get("decl_line"), "params": [], "vars": [],
                     "begin": a.get("low_pc"), "end": a.get("high_pc")}

                def var(d, scope, depth):
                    loc = d["attrs"].get("location")
                    regs = [loc[1]] if loc and loc[0] == "reg" else \
                        [r for r, _ in loc[1] if r] if loc and loc[0] == "pieces" else []
                    return {
                        "name": d["attrs"].get("name"),
                        "line": d["attrs"].get("decl_line"),
                        "type": type_of(dies, d["attrs"].get("type")),
                        "reg": loc[1] if loc and loc[0] == "reg" else None,
                        "regs": regs,
                        "scope": scope,
                        "depth": depth,
                    }

                def walk(d, scope, depth):
                    for c in d["children"]:
                        if c["tag"] == "formal_parameter":
                            k["params"].append(var(c, scope, depth))
                        elif c["tag"] == "variable":
                            k["vars"].append(var(c, scope, depth))
                        elif c["tag"] == "lexical_block":
                            walk(c, (c["attrs"].get("low_pc"), c["attrs"].get("high_pc")), depth + 1)

                walk(sp, (k["begin"], k["end"]), 0)
                out[a.get("name")] = k
    return out


def _entries(ptx):
    """kernel name -> its body's lines (comments dropped), in order."""
    out, name, depth = {}, None, 0
    for raw in ptx.splitlines():
        t = raw.split("//", 1)[0].strip()
        if name is None:
            m = re.match(r"(?:\.visible\s+)?\.entry\s+([A-Za-z_$][\w$]*)", t)
            if m:
                name, depth = m.group(1), 0
                out[name] = []
            continue
        out[name].append(t)
        depth += t.count("{") - t.count("}")
        if depth == 0 and "}" in t:
            name = None
    return out


def _written(insn):
    """The registers an instruction writes: its first operand, or the
    braced list a vector load fills. Stores and branches write none."""
    m = re.match(r"(?:@!?%\w+\s+)?([a-z][\w.]*)\s+(.*);$", insn)
    if not m:
        return []
    op, rest = m.groups()
    if op.split(".")[0] in ("st", "bra", "ret", "exit", "bar", "membar", "trap", "red", "cp", "prefetch", "fence"):
        return []
    rest = rest.strip()
    if rest.startswith("{"):
        return re.findall(r"%\w+", rest[:rest.index("}")])
    return re.findall(r"%\w+", rest.split(",")[0])[:1]


def check(ptx):
    """What is wrong with the module's description of its kernels: a list of
    messages, empty when every binding is where the description says.

    * every register a binding names is declared by its kernel and written
      before the binding's scope begins (it holds the binding's value there);
    * two bindings name one register only if nothing writes it while either
      is in scope - a vector and a copy of one of its lanes - so they agree;
    * each scope's labels are in the kernel, the begin before the end, and
      inside the scope that contains it."""
    problems = []
    entries = _entries(ptx)
    for kname, k in kernels(ptx).items():
        body = entries.get(kname)
        if body is None:
            problems.append("%s: described, but the module has no such kernel" % kname)
            continue
        labels = {t[:-1]: i for i, t in enumerate(body) if re.fullmatch(r"\$[\w$]+:", t)}
        declared = set()
        for t in body:
            m = re.match(r"\.reg\s+\.\w+\s+(%[a-z]+)<(\d+)>;", t)
            if m:
                declared |= {"%s%d" % (m.group(1), i) for i in range(int(m.group(2)))}
        writers = {}
        for i, t in enumerate(body):
            for r in _written(t):
                writers.setdefault(r, []).append(i)
        owners = {}
        for v in k["params"] + k["vars"]:
            begin, end = v["scope"]
            where = "%s: `%s` (line %s)" % (kname, v["name"], v["line"])
            if begin not in labels or end not in labels:
                problems.append("%s: its scope's labels %s..%s are not in the kernel" % (where, begin, end))
                continue
            if labels[begin] > labels[end]:
                problems.append("%s: its scope ends before it begins" % where)
            for r in v["regs"]:
                owners.setdefault(r, []).append((v["name"], labels[begin], labels[end]))
                if r not in declared:
                    problems.append("%s: register %s is not declared" % (where, r))
                elif not any(i < labels[begin] for i in writers.get(r, [])):
                    problems.append("%s: register %s is not written before the scope begins" % (where, r))
        for r, named in owners.items():
            if len(named) < 2:
                continue
            for name, lo, hi in named:
                inside = [i for i in writers.get(r, []) if lo < i < hi]
                if inside:
                    problems.append("%s: %s share register %s, and line %d of the kernel writes it while `%s` "
                                    "is in scope" % (kname, ", ".join("`%s`" % n for n, _, _ in named), r,
                                                     inside[0] + 1, name))
                    break
        # Nesting: walk the scope tree again with each block's parent.
        for unit in parse(ptx):
            for cu in unit:
                for sp in cu["children"]:
                    if sp["tag"] != "subprogram" or sp["attrs"].get("name") != kname:
                        continue

                    def nest(d, lo, hi):
                        for c in d["children"]:
                            if c["tag"] != "lexical_block":
                                continue
                            b, e = c["attrs"].get("low_pc"), c["attrs"].get("high_pc")
                            if b in labels and e in labels and not (lo <= labels[b] <= labels[e] <= hi):
                                problems.append("%s: the scope %s..%s is not inside the one containing it"
                                                % (kname, b, e))
                            nest(c, labels.get(b, lo), labels.get(e, hi))

                    a = sp["attrs"]
                    nest(sp, labels.get(a.get("low_pc"), 0), labels.get(a.get("high_pc"), len(body)))
    return problems


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--check"] and len(args) == 2:
        with open(args[1]) as f:
            found = check(f.read())
        for p in found:
            print(p)
        sys.exit(1 if found else 0)
    if len(args) != 1:
        sys.exit("usage: ydwarf.py [--check] PROGRAM.ptx")
    with open(args[0]) as f:
        print(json.dumps(kernels(f.read()), indent=2))
