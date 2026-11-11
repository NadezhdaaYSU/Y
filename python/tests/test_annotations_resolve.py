"""Every annotation in the y_lang package names something that exists.

Python 3.14 evaluates annotations only when asked (PEP 649), so a name used in
one and never imported passes there unnoticed. Python 3.8-3.13 evaluate them
when the function is defined: `y_lang/ops/block.py` used `Tuple` without
importing it, and on Kaggle's Python 3.13 (2026-11-11) `import y_lang.ops`
raised NameError, under a package that declares Python >= 3.8. This asks for
every function's and class's annotations, which fails on any version.
"""

import importlib
import inspect
import pkgutil
import typing
import unittest

import y_lang


def unresolvable():
    bad, checked = [], 0
    for info in pkgutil.walk_packages(y_lang.__path__, "y_lang."):
        if ".tests" in info.name:
            continue
        try:
            mod = importlib.import_module(info.name)
        except ImportError as e:              # an optional dependency, not an annotation
            if isinstance(e, ModuleNotFoundError) and e.name and not e.name.startswith("y_lang"):
                continue
            bad.append(f"{info.name}: import failed: {e!r}")
            continue
        for name, obj in vars(mod).items():
            if getattr(obj, "__module__", None) != mod.__name__:
                continue
            targets = []
            if inspect.isfunction(obj):
                targets.append(obj)
            elif inspect.isclass(obj):
                targets.append(obj)
                targets += [v for v in vars(obj).values() if inspect.isfunction(v)]
            for t in targets:
                checked += 1
                try:
                    typing.get_type_hints(t)
                except Exception as e:        # noqa: BLE001 - the error is the finding
                    bad.append(f"{info.name}.{getattr(t, '__qualname__', name)}: {e!r}")
    return bad, checked


class TestAnnotationsResolve(unittest.TestCase):
    def test_every_annotation_resolves(self):
        bad, checked = unresolvable()
        self.assertGreater(checked, 300, "the walk found too little to mean anything")
        self.assertEqual(bad, [], "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
