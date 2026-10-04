"""make_manifest.py -- write (or verify) MANIFEST.json of a deposit folder: every file with its package-relative path, size in
bytes and SHA-256.

Usage:  python scripts/make_manifest.py <package folder>            write <package folder>/MANIFEST.json
        python scripts/make_manifest.py <package folder> --check    recompute and compare with the stored MANIFEST.json
"""
import hashlib, json, os, sys, time


def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(root):
    rows = []
    for d, _, files in os.walk(root):
        for f in sorted(files):
            p = os.path.join(d, f); r = os.path.relpath(p, root).replace("\\", "/")
            if r == "MANIFEST.json":
                continue
            rows.append(dict(path=r, bytes=os.path.getsize(p), sha256=sha256(p)))
    return sorted(rows, key=lambda x: x["path"])


if __name__ == "__main__":
    root = os.path.abspath(sys.argv[1]); man = os.path.join(root, "MANIFEST.json")
    rows = scan(root)
    if "--check" in sys.argv:
        old = {r["path"]: r for r in json.load(open(man, encoding="utf-8"))["files"]}; new = {r["path"]: r for r in rows}
        bad = [p for p in set(old) | set(new) if old.get(p, {}).get("sha256") != new.get(p, {}).get("sha256")]
        print("OK" if not bad else f"{len(bad)} mismatches: {sorted(bad)[:20]}"); sys.exit(1 if bad else 0)
    json.dump(dict(package=os.path.basename(root), created=time.strftime("%Y-%m-%d"), n_files=len(rows), total_bytes=sum(r["bytes"] for r in rows),
                   hash="sha256", files=rows), open(man, "w", encoding="utf-8"), indent=1)
    print(f"{len(rows)} files, {sum(r['bytes'] for r in rows) / 1e6:.1f} MB -> {man}")
