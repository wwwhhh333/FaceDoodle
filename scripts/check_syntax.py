import sys
from pathlib import Path

errors = []
for p in list(Path("app").rglob("*.py")) + list(Path("tests").rglob("*.py")):
    try:
        source = p.read_text(encoding="utf-8-sig")
        compile(source, str(p), "exec")
    except SyntaxError as e:
        errors.append(f"{p}: {e}")

if errors:
    for e in errors:
        print(e, file=sys.stderr)
    sys.exit(1)
