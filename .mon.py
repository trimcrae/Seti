import json
import os

prev_path = "/tmp/mon_prev.txt"
state = []
for name, path in (("diagnose", "/tmp/a.json"), ("run", "/tmp/b.json")):
    try:
        with open(path) as fh:
            d = json.load(fh)
        state.append(f"{name}={d.get('status')}/{d.get('conclusion') or '-'}")
    except Exception:
        state.append(f"{name}=unknown")
cur = " ".join(state)
prev = ""
if os.path.exists(prev_path):
    prev = open(prev_path).read().strip()
if cur != prev:
    print(cur, flush=True)
    with open(prev_path, "w") as fh:
        fh.write(cur)
if cur.count("completed") == 2:
    print("BOTH TERMINAL", flush=True)
    open("/tmp/done.flag", "w").write("1")
