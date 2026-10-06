#!/usr/bin/env python3
"""Crash-free exact-match measurement for the boilerplate model (no torch needed, plain Python).

1) prep:   make N prompt files from a test folder
       python bp_check.py prep --dir bp3\\test_names --n 60 --out evalset\\names
2) run the model on every prompt with the loop that already works for you (one process per prompt):
       Get-ChildItem evalset\\names\\*.cs | ForEach-Object { "=== $($_.Name)"; python minigpt.py --load models\\bp3.pt --prompt-file $_.FullName --temp 0 --tokens 800 } | Out-File -Encoding utf8 results\\bp3\\names.txt
3) score:  compare the output with the real answers (the text after '// out:' in the test folder)
       python bp_check.py score --dir bp3\\test_names --results results\\bp3\\names.txt
"""
import argparse
import os
import re
import sys

MARK = "// out:\n"


def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-16"):  # PowerShell 5 Tee-Object/Out-File may write UTF-16
        if enc == "utf-16" and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            continue
        try:
            return raw.decode(enc).replace("\r\n", "\n")
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace").replace("\r\n", "\n")


def split_example(text):
    head, tail = text.split(MARK, 1)
    task = head.split("\n", 1)[0].replace("// task:", "").strip()
    return task, head + MARK, tail


def prep(a):
    names = sorted(f for f in os.listdir(a.dir) if f.endswith(".cs"))[:a.n]
    os.makedirs(a.out, exist_ok=True)
    for f in names:
        _, prompt, _ = split_example(read_text(os.path.join(a.dir, f)))
        with open(os.path.join(a.out, f), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(prompt)
    print(f"wrote {len(names)} prompts to {a.out}")


def parse_results(text):
    """{file name: generated text after '// out:'}; blocks without the marker (errors) map to None."""
    parts = re.split(r"(?m)^=== (\S+)[ \t]*$", text)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        block = parts[i + 1].replace("\r\n", "\n")
        out[parts[i]] = block.split(MARK, 1)[1].rstrip() if MARK in block else None
    return out


def first_diff(got, want):
    g, w = got.split("\n"), want.split("\n")
    for k in range(max(len(g), len(w))):
        x = g[k] if k < len(g) else "<missing>"
        y = w[k] if k < len(w) else "<missing>"
        if x != y:
            return k + 1, x.strip(), y.strip()
    return 0, "", ""


def score(a):
    res = parse_results(read_text(a.results))
    per, fails, errors = {}, [], 0
    for name, got in res.items():
        path = os.path.join(a.dir, name)
        if not os.path.exists(path):
            continue
        task, _, want = split_example(read_text(path))
        want = want.rstrip()
        h = per.setdefault(task, [0, 0])
        h[1] += 1
        if got is None:
            errors += 1
        elif got == want:
            h[0] += 1
            continue
        fails.append((name, task, got, want))
    total = sum(v[1] for v in per.values())
    hits = sum(v[0] for v in per.values())
    if not total:
        raise SystemExit("No matching results found (file names in the results must match the test folder).")
    print(f"exact match: {hits}/{total} = %{100 * hits / total:.1f}" + (f"   ({errors} blocks had no output/error)" if errors else ""))
    for t in sorted(per):
        print(f"  {t:12s} {per[t][0]}/{per[t][1]}  %{100 * per[t][0] / per[t][1]:.1f}")
    for name, task, got, want in fails[:a.show]:
        print("-" * 60)
        if got is None:
            print(f"{name} [{task}]: no generated output")
            continue
        line, x, y = first_diff(got, want)
        print(f"{name} [{task}] first difference at line {line}:")
        print("  model:", x)
        print("  real :", y)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--dir", required=True)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--dir", required=True)
    s.add_argument("--results", required=True)
    s.add_argument("--show", type=int, default=5, help="how many failures to print")
    a = ap.parse_args()
    prep(a) if a.cmd == "prep" else score(a)


if __name__ == "__main__":
    sys.exit(main())
