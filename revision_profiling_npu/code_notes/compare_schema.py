"""Structural comparison of two result JSONs (key paths; list elements by first item). python3 -B compare_schema.py ref.json new.json"""
import json, sys

def paths(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            if p.endswith("loaded_modules") or p.endswith("files_checked") or p.endswith("modules"):
                continue
            out.update(paths(v, f"{p}.{k}" if p else k))
    elif isinstance(o, list) and o and isinstance(o[0], dict):
        out.update(paths(o[0], p + "[]"))
    else:
        out[p] = type(o).__name__
    return out

a, b = (paths(json.load(open(f))) for f in sys.argv[1:3])
print("ref keys", len(a), "new keys", len(b))
print("missing in new:", sorted(set(a) - set(b)))
print("extra in new:", sorted(set(b) - set(a)))
print("type differs:", sorted(k for k in set(a) & set(b) if a[k] != b[k] and "NoneType" not in (a[k], b[k])))
