# Word-frequency lists (words.tsv format) from real English prose found on this machine.
import ast, collections, glob, re, sys
WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
def prose_lines_go(path):
    for line in open(path, errors="ignore"):
        s = line.strip()
        if s.startswith("//") and not s.startswith("//go:") and not s.startswith("// +build"):
            yield s[2:]
def prose_py(path):
    try:
        tree = ast.parse(open(path, errors="ignore").read())
    except Exception:
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module, ast.AsyncFunctionDef)):
            d = ast.get_docstring(node)
            if d:
                yield d
def count(texts):
    c = collections.Counter()
    for t in texts:
        # skip code-ish lines: identifiers with _ or camelCase, brackets
        for w in WORD.findall(t):
            if any(ch.isupper() for ch in w[1:]):
                continue
            c[w.lower()] += 1
    return c
def dump(c, out):
    tot = sum(c.values())
    with open(out, "w") as fh:
        for w, n in c.most_common(80000):
            fh.write(f"{w}\t{n / tot:.9g}\n")
    print(out, len(c), "distinct", tot, "tokens")
go = count(t for p in glob.glob("/usr/local/go/src/**/*.go", recursive=True) if not p.endswith("_test.go") and "testdata" not in p and __import__("os").path.isfile(p) for t in prose_lines_go(p))
dump(go, "corp_go/words.tsv")
py = count(t for p in glob.glob("/usr/lib/python3*/**/*.py", recursive=True) if __import__("os").path.isfile(p) and "test" not in p for t in prose_py(p))
dump(py, "corp_py/words.tsv")
top = [l for l in open("words.tsv")][:5000]
open("corp_top5k/words.tsv", "w").writelines(top)
print("corp_top5k/words.tsv 5000")
