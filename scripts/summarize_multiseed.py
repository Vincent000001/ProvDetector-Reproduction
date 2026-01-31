#!/usr/bin/env python3
import json, glob, os, statistics as st, sys

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "results/multiseed"
    paths = sorted(glob.glob(os.path.join(base, "seed*/test_metrics.json")))
    if not paths:
        print("No test_metrics.json found under", base)
        return 1

    rows=[]
    for p in paths:
        with open(p,'r') as f:
            m=json.load(f)
        f1 = m.get("f1_score", m.get("f1"))
        prec = m.get("precision")
        rec = m.get("recall")
        auc = m.get("roc_auc")
        run = os.path.basename(os.path.dirname(p))
        rows.append((run, prec, rec, f1, auc))

    print("dir\tprec\trec\tf1\tauc")
    for r in rows:
        print("%s\t%.4f\t%.4f\t%.4f\t%s" % (r[0], r[1], r[2], r[3], ("nan" if r[4] is None else f"{r[4]:.4f}")))

    f1s=[r[3] for r in rows if r[3] is not None]
    print("")
    print("F1 mean=%.4f std=%.4f min=%.4f max=%.4f" % (st.mean(f1s), st.pstdev(f1s), min(f1s), max(f1s)))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
