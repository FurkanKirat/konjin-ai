#!/usr/bin/env python3
"""Rule-based C# boilerplate dataset for minigpt (one example per .cs file, EOF is appended by minigpt).

Example file:
    // task: equals
    public class Item
    {
        public string Name;
        public int Id;
    }
    // out:
    <generated members>

Tasks: constructor, tostring, equals (Equals + GetHashCode), clone.
Splits (no overlap in identifiers):
    train/       identifiers from the TRAIN word pool, 1-6 fields
    test_names/  identifiers ONLY from the held-out word pool, 1-6 fields
    test_long/   TRAIN identifiers but 7-8 fields (never seen in training)

Usage:  python gen_boilerplate.py --out bp --train 150000 --test 300
"""
import argparse
import collections
import os
import random
import re

WORDS = """
health mana stamina armor damage speed range level score count index size width height depth length
weight price cost value amount total limit max min start end first last next prev current target source
name title label text message id key token code type kind mode state status flag active enabled visible
locked dirty ready alive dead owner parent child leader member player enemy ally boss npc pet hero
item weapon shield potion scroll gem coin gold silver crystal ore wood stone iron steel cloth leather
position rotation scale offset origin center bounds radius angle direction velocity force mass gravity
color alpha red green blue tint shade light shadow glow fade pulse spark flame frost storm wind rain
time delay timer cooldown duration interval frame tick step turn round wave phase stage season day night
tile chunk block cell grid map room door gate wall floor roof stair bridge tower castle camp village
texture sprite mesh model shader material effect sound music voice noise track clip layer channel
input button axis cursor menu panel slot icon tooltip dialog quest reward skill spell perk trait buff
inventory bag chest crate vault shop trade recipe craft upgrade rank tier grade rarity quality chance
seed noise biome region zone area sector layer depth height slope soil water lava sand snow ice
packet buffer stream queue stack list table cache pool batch entry record header footer payload
host port address session ticket version build patch release branch commit author email
""".split()

KEYWORDS = set("""
abstract as base bool break byte case catch char checked class const continue decimal default delegate do double
else enum event explicit extern false finally fixed float for foreach goto if implicit in int interface internal
is lock long namespace new null object operator out override params private protected public readonly ref return
sbyte sealed short sizeof stackalloc static string struct switch this throw true try typeof uint ulong unchecked
unsafe ushort using virtual void volatile while
""".split())

PRIMS = ["int", "float", "double", "bool", "string", "long", "byte", "char", "short", "uint"]
NULLABLE_OK = ["int", "float", "double", "bool", "long", "byte"]
ENGINE = ["Vector2", "Vector3", "Color", "DateTime", "Guid"]
SUFFIXES = ["", "", "", "Data", "Info", "Config", "State", "Stats"]
TASKS = ["constructor", "tostring", "equals", "clone"]
MAX_BYTES = 1000  # whole example (prompt + output) must fit in a 1024 block


PSEUDO = 0.0  # probability that a drawn word is a random pronounceable pseudo-word (set from --pseudo)


def extract_words(dirs, min_count=5, max_words=5000, scan_mb=300):
    """Real words from the identifiers of .cs files under `dirs` (camelCase/PascalCase/snake_case split,
    lower-cased, 3-12 letters). Returns the max_words most frequent words that occur at least min_count times."""
    cnt, scanned = collections.Counter(), 0
    ident = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    part = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+")
    for root in dirs:
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d.lower() not in ("bin", "obj", ".git", "library", "temp", "packages")]
            for f in fn:
                if not f.lower().endswith(".cs"):
                    continue
                try:
                    with open(os.path.join(dp, f), encoding="utf-8-sig") as fh:
                        text = fh.read(500_000)
                except (UnicodeDecodeError, OSError):
                    continue
                scanned += len(text)
                for w in part.findall(" ".join(ident.findall(text))):
                    w = w.lower()
                    if 3 <= len(w) <= 12 and w.isascii():
                        cnt[w] += 1
                if scanned > scan_mb * 1_000_000:
                    break
    words = [w for w, c in cnt.most_common(max_words * 2) if c >= min_count and w not in KEYWORDS]
    return words[:max_words]


def pools(words=None, seed=12345):
    ws = sorted(set(w for w in (words or WORDS) if w not in KEYWORDS))
    random.Random(seed).shuffle(ws)
    k = int(len(ws) * 0.15)
    return ws[k:], ws[:k]  # train, held-out


def pseudo_word(rng):
    return "".join(rng.choice("bcdfghklmnprstvwz") + rng.choice("aeiou") for _ in range(rng.randint(2, 4)))


def rand_word(rng, pool):
    return pseudo_word(rng) if PSEUDO and rng.random() < PSEUDO else rng.choice(pool)


def pascal(w):
    return w[0].upper() + w[1:]


def camel(parts):
    return parts[0] + "".join(pascal(p) for p in parts[1:])


def rand_name(rng, pool, used):
    for _ in range(100):
        parts = [rand_word(rng, pool) for _ in range(rng.choice((1, 1, 2)))]
        pas = "".join(pascal(p) for p in parts)
        if pas not in used and camel(parts) not in KEYWORDS:
            used.add(pas)
            return pas, camel(parts)
    raise RuntimeError("could not draw a unique name")


def rand_type(rng, pool):
    r = rng.random()
    if r < 0.50:
        return rng.choice(PRIMS)
    if r < 0.58:
        return rng.choice(NULLABLE_OK) + "?"
    if r < 0.68:
        return rng.choice(ENGINE)
    base = rng.choice(PRIMS) if rng.random() < 0.5 else pascal(rand_word(rng, pool))
    form = rng.random()
    if form < 0.35:
        return base
    if form < 0.70:
        return f"List<{base}>"
    if form < 0.90:
        return f"{base}[]"
    return f"Dictionary<string, {base}>"


def make_class(rng, pool, nfields):
    cname = pascal(rand_word(rng, pool)) + rng.choice(SUFFIXES)
    style = rng.choice(("field", "prop", "priv"))
    used, fields = {cname}, []  # a member may not share its enclosing type's name
    for _ in range(nfields):
        pas, cam = rand_name(rng, pool, used)
        fields.append((rand_type(rng, pool), pas, cam))
    return cname, style, fields


def member(style, f):
    return "_" + f[2] if style == "priv" else f[1]


def class_text(cname, style, fields):
    lines = [f"public class {cname}", "{"]
    for t, pas, cam in fields:
        if style == "field":
            lines.append(f"    public {t} {pas};")
        elif style == "prop":
            lines.append(f"    public {t} {pas} {{ get; set; }}")
        else:
            lines.append(f"    private readonly {t} _{cam};")
    lines.append("}")
    return "\n".join(lines)


def out_constructor(cname, style, fields):
    args = ", ".join(f"{t} {cam}" for t, _, cam in fields)
    body = "\n".join(f"    {member(style, f)} = {f[2]};" for f in fields)
    return f"public {cname}({args})\n{{\n{body}\n}}"


def out_tostring(cname, style, fields):
    parts = ", ".join(f"{f[1]}={{{member(style, f)}}}" for f in fields)
    return f'public override string ToString()\n{{\n    return $"{cname}({parts})";\n}}'


def out_equals(cname, style, fields):
    cmp = " && ".join(f"{member(style, f)} == other.{member(style, f)}" for f in fields)
    args = ", ".join(member(style, f) for f in fields)
    return (f"public override bool Equals(object obj) => obj is {cname} other && Equals(other);\n\n"
            f"public bool Equals({cname} other)\n{{\n    return other != null && {cmp};\n}}\n\n"
            f"public override int GetHashCode()\n{{\n    return HashCode.Combine({args});\n}}")


def out_clone(cname, style, fields):
    args = ", ".join(member(style, f) for f in fields)
    return f"public {cname} Clone()\n{{\n    return new {cname}({args});\n}}"


OUT = {"constructor": out_constructor, "tostring": out_tostring, "equals": out_equals, "clone": out_clone}


def make_example(rng, pool, lo, hi):
    while True:
        cname, style, fields = make_class(rng, pool, rng.randint(lo, hi))
        task = rng.choice(TASKS)
        text = (f"// task: {task}\n{class_text(cname, style, fields)}\n// out:\n"
                f"{OUT[task](cname, style, fields)}\n")
        if len(text.encode("utf-8")) <= MAX_BYTES:
            return task, text


def write_split(path, n, rng, pool, lo, hi):
    os.makedirs(path, exist_ok=True)
    counts = dict.fromkeys(TASKS, 0)
    width = len(str(n))
    for i in range(n):
        task, text = make_example(rng, pool, lo, hi)
        counts[task] += 1
        with open(os.path.join(path, f"{i:0{width}d}.cs"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    print(f"{path}: {n} files  {counts}")


def write_prompt_pack(out):
    """8 hand-check prompts (+ expected outputs): one per task from test_names (n_) and from test_long (l_)."""
    pdir, edir = os.path.join(out, "prompts", "bp"), os.path.join(out, "expected")
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(edir, exist_ok=True)
    k = 0
    for split, tag in (("test_names", "n"), ("test_long", "l")):
        seen = set()
        for f in sorted(os.listdir(os.path.join(out, split))):
            with open(os.path.join(out, split, f), encoding="utf-8") as fh:
                text = fh.read()
            task = text.split("\n", 1)[0].replace("// task:", "").strip()
            if task in seen:
                continue
            seen.add(task)
            k += 1
            head, tail = text.split("// out:\n", 1)
            name = f"{k:02d}_{tag}_{task}.cs"
            for d, body in ((pdir, head + "// out:\n"), (edir, tail)):
                with open(os.path.join(d, name), "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(body)
            if len(seen) == len(TASKS):
                break
    print(f"hand-check prompts: {pdir}   expected outputs: {edir}")


def main():
    global PSEUDO
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bp", help="output folder (train/, test_names/, test_long/ are created inside)")
    ap.add_argument("--train", type=int, default=150000)
    ap.add_argument("--test", type=int, default=300, help="files per test split")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--words-from", nargs="+", default=None, metavar="DIR",
                    help="take the identifier words from the .cs files under these folders (thousands of real words "
                         "instead of the built-in ~200); the held-out 15%% split is made from them")
    ap.add_argument("--max-words", type=int, default=5000)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--pseudo", type=float, default=0.1,
                    help="train only: probability that a word is a random pseudo-word (teaches pure copying)")
    ap.add_argument("--show", action="store_true", help="print one example per task and exit (writes nothing)")
    a = ap.parse_args()
    words = None
    if a.words_from:
        words = extract_words(a.words_from, a.min_count, a.max_words)
        print(f"words extracted from the corpus: {len(words)} (e.g. {', '.join(words[:8])} ... {', '.join(words[-4:])})")
        if len(words) < 300:
            raise SystemExit("Too few words extracted; check --words-from or lower --min-count.")
    train_pool, held_pool = pools(words)
    print(f"word pools: train {len(train_pool)}, held-out {len(held_pool)}")
    rng = random.Random(a.seed)
    if a.show:
        PSEUDO = a.pseudo
        for t in TASKS:
            while True:
                task, text = make_example(rng, train_pool, 2, 4)
                if task == t:
                    print(text + "=" * 40)
                    break
        return
    PSEUDO = a.pseudo
    write_split(os.path.join(a.out, "train"), a.train, rng, train_pool, 1, 6)
    PSEUDO = 0.0  # tests use real words only
    write_split(os.path.join(a.out, "test_names"), a.test, rng, held_pool, 1, 6)
    write_split(os.path.join(a.out, "test_long"), a.test, rng, train_pool, 7, 8)
    write_prompt_pack(a.out)


if __name__ == "__main__":
    main()