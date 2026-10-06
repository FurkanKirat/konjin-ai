#!/usr/bin/env python3
"""Interactive demo for the boilerplate model (needs minigpt.py and gen_boilerplate.py in the same folder).

    python bp_demo.py --load models\\bp3.pt

You give a class name, a field style and the fields ("string Name, int Id, float Price"); the model writes
the constructor / ToString / Equals+GetHashCode / Clone. The answer is also compared with the rule-based
reference (the program that generated the training data), so you can see whether the model got it exactly right.

One-shot (no questions):
    python bp_demo.py --load models\\bp3.pt --class Item --style prop --fields "string Name, int Id" --task all
"""
import argparse
import os
import re
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _d) for _d in ("model", "datagen", "eval")]
import gen_boilerplate as g  # noqa: E402

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
STYLES = {"1": "field", "field": "field", "2": "prop", "prop": "prop", "3": "priv", "priv": "priv"}
TASKS = g.TASKS
MAX_FIELDS = 8  # trained up to 8 (HashCode.Combine takes at most 8 arguments)


def split_top(s):
    """Split on commas that are not inside <> or []."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<[":
            depth += 1
        elif ch in ">]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def parse_fields(spec):
    fields = []
    for part in split_top(spec):
        try:
            t, n = part.rsplit(None, 1)
        except ValueError:
            raise ValueError(f"'{part}': write each field as 'type name'")
        n = n.lstrip("_")
        if not IDENT.match(n):
            raise ValueError(f"'{n}' is not a valid identifier")
        cam = n[0].lower() + n[1:]
        if cam in g.KEYWORDS:
            raise ValueError(f"'{cam}' is a C# keyword")
        t = re.sub(r"\s*,\s*", ", ", t.strip())
        fields.append((t, n[0].upper() + n[1:], cam))
    if not 1 <= len(fields) <= MAX_FIELDS:
        raise ValueError(f"1-{MAX_FIELDS} fields are supported (the model was trained on 1-8)")
    return fields


def build(cname, style, fields, task):
    """Returns (prompt, reference answer) in exactly the training format."""
    head = f"// task: {task}\n{g.class_text(cname, style, fields)}\n// out:\n"
    return head, g.OUT[task](cname, style, fields) + "\n"


def show(title, gen, ref):
    print(f"\n--- {title} " + "-" * max(4, 56 - len(title)))
    print(gen)
    print(f"[exactly equal to the rule-based reference: {'YES' if gen.rstrip() == ref.rstrip() else 'NO'}]")
    if gen.rstrip() != ref.rstrip():
        print("reference:\n" + ref.rstrip())


def run(generate, cname, style, fields, task):
    for t in (TASKS if task == "all" else [task]):
        head, ref = build(cname, style, fields, t)
        if len((head + ref).encode("utf-8")) > 1000:
            print(f"warning: '{t}' output is longer than anything seen in training; it may be cut or wrong.")
        show(t, generate(head.encode("utf-8")).rstrip(), ref)


def load(path):
    import torch
    sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _d) for _d in ("model", "datagen", "eval")]
    import minigpt as mg
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(path, map_location=device, weights_only=True)
    model = mg.build(ck["cfg"], device)
    model.load_state_dict(ck["model"])
    vocab = ck["vocab"]
    return lambda p: mg.sample_bytes(model, vocab, p, 900, 0.0, device).decode("utf-8", errors="replace")


def ask(msg, default=None):
    s = input(msg).strip()
    return s or default


def interactive(generate):
    print("Boilerplate model demo. Empty class name quits.\n"
          "Fields: 'type name' separated by commas, e.g.  string Name, int Id, List<int> Scores\n")
    while True:
        cname = ask("class name: ")
        if not cname:
            return
        if not IDENT.match(cname) or cname in g.KEYWORDS:
            print("not a valid class name\n")
            continue
        style = STYLES.get(ask("field style  1=public field  2=property  3=private readonly _field [2]: ", "2"))
        if not style:
            print("choose 1, 2 or 3\n")
            continue
        try:
            fields = parse_fields(ask("fields: ", ""))
        except ValueError as e:
            print(f"error: {e}\n")
            continue
        task = ask(f"task ({'/'.join(TASKS)}/all) [all]: ", "all")
        if task not in TASKS + ["all"]:
            print("unknown task\n")
            continue
        run(generate, cname, style, fields, task)
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--class", dest="cname")
    ap.add_argument("--style", default="prop")
    ap.add_argument("--fields")
    ap.add_argument("--task", default="all")
    a = ap.parse_args()
    generate = load(a.load)
    if a.cname and a.fields:
        run(generate, a.cname, STYLES[a.style], parse_fields(a.fields), a.task)
    else:
        interactive(generate)


if __name__ == "__main__":
    main()