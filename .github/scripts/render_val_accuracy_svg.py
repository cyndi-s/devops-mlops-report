#!/usr/bin/env python3
import os, csv, statistics as stats
from pathlib import Path

CSV_FILE = Path("commitHistory.csv")
SVG_OUT = Path("val_accuracy.svg")
MAX_ROWS = 50  # visualize the most recent N rows

def read_vals():
    if not CSV_FILE.exists():
        return []
    vals = []
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                v = float(row.get("val_accuracy", "") or "")
                vals.append(v)
            except Exception:
                continue
    return vals[-MAX_ROWS:]

def normalize(vs):
    if not vs: return []
    lo, hi = min(vs), max(vs)
    if hi == lo: return [0.5 for _ in vs]
    return [(v - lo) / (hi - lo) for v in vs]

def moving_avg(vs, k=5):
    if not vs: return []
    out=[]
    for i in range(len(vs)):
        s = vs[max(0, i-k+1):i+1]
        out.append(sum(s)/len(s))
    return out

def points_to_svg(points, w, h, pad=20):
    if not points: 
        return f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg"><text x="10" y="20">No data</text></svg>'
    n = len(points)
    xs = [pad + (i*(w-2*pad)/(n-1)) for i in range(n)]
    ys = [pad + (1-p)*(h-2*pad) for p in points]
    path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x,y in zip(xs, ys))
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg"><path d="{path}" fill="none" stroke="black" stroke-width="2"/><text x="{pad}" y="{h-pad/2}" font-size="10">val_accuracy (last {n})</text></svg>'

vals = read_vals()
norm = normalize(vals)
avg = moving_avg(norm)

svg = points_to_svg(avg if avg else norm, 640, 240)
SVG_OUT.write_text(svg, encoding="utf-8")
print(f"Wrote {SVG_OUT} (n={len(vals)})")
