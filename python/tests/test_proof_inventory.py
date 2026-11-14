"""docs/exact_decode_proof_inventory.md quotes theorem statements from proofs/.

A quoted statement that is no longer in its file, or a proof file the
inventory does not list, fails here. No Coq and no torch are needed.
"""
import pathlib
import re
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "exact_decode_proof_inventory.md"
PROOFS = REPO / "proofs"


def strip_comments(src):
    """Coq source without `(* ... *)` comments (they nest)."""
    out, depth, i = [], 0, 0
    while i < len(src):
        if src.startswith("(*", i):
            depth += 1
            i += 2
        elif depth and src.startswith("*)", i):
            depth -= 1
            i += 2
        else:
            if not depth:
                out.append(src[i])
            i += 1
    return "".join(out)


def squeeze(text):
    return " ".join(text.split())


def sections(doc):
    """`{file name: [quoted coq blocks]}` from the inventory."""
    found, current = {}, None
    for part in re.split(r"(?m)^### `proofs/([A-Za-z0-9_]+\.v)`$", doc)[1:]:
        if part.endswith(".v") and "\n" not in part:
            current = part
            found[current] = []
        else:
            found[current] = re.findall(r"```coq\n(.*?)\n```", part, flags=re.S)
    return found


class ProofInventory(unittest.TestCase):
    def setUp(self):
        self.sections = sections(DOC.read_text())

    def test_every_proof_file_is_listed(self):
        on_disk = {p.name for p in PROOFS.glob("*.v")}
        self.assertEqual(on_disk, set(self.sections))

    def test_every_listed_file_quotes_a_theorem(self):
        for name, blocks in self.sections.items():
            self.assertTrue(blocks, f"{name}: no statement is quoted")

    def test_quoted_statements_are_in_their_files(self):
        checked = 0
        for name, blocks in self.sections.items():
            source = squeeze(strip_comments((PROOFS / name).read_text()))
            for block in blocks:
                self.assertIn(squeeze(block), source,
                              f"{name}: the inventory quotes a statement the file "
                              f"no longer contains:\n{block}")
                checked += 1
        self.assertGreaterEqual(checked, 26)

    def test_a_changed_statement_is_noticed(self):
        """The check above can fail: one changed bound is not in the file."""
        name = "NormGridSum.v"
        source = squeeze(strip_comments((PROOFS / name).read_text()))
        block = next(b for b in self.sections[name] if "8192" in b)
        self.assertIn(squeeze(block), source)
        self.assertNotIn(squeeze(block.replace("8192", "8193")), source)


if __name__ == "__main__":
    unittest.main()
