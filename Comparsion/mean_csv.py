import sys
import csv
import os
import glob
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python mean_csv.py <prefix>")
    print("  e.g. python mean_csv.py FBP")
    print("  Finds all Comparsion/<prefix>_*/metrics.csv files")
    sys.exit(1)

prefix = sys.argv[1]
base_dir = os.path.dirname(os.path.abspath(__file__))
pattern = os.path.join(base_dir, f"{prefix}_*", "metrics.csv")
csv_files = sorted(glob.glob(pattern))

if not csv_files:
    print(f"No metrics.csv found matching: {pattern}")
    sys.exit(1)

results = []
for csv_path in csv_files:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        continue

    folder = os.path.basename(os.path.dirname(csv_path))
    numeric_cols = [k for k in rows[0] if k != 'slice']
    entry = {'folder': folder, 'n_slices': len(rows)}
    for col in numeric_cols:
        vals = [float(r[col]) for r in rows]
        entry[f"{col}_mean"] = np.mean(vals)
        entry[f"{col}_std"] = np.std(vals)
    results.append(entry)

# Write summary CSV
out_path = os.path.join(base_dir, f"{prefix}_summary.csv")
fieldnames = list(results[0].keys())
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

# Print to console too
print(f"Wrote {out_path} ({len(results)} settings)")
for r in results:
    print(f"\n{r['folder']} ({r['n_slices']} slices)")
    for k, v in r.items():
        if k in ('folder', 'n_slices'):
            continue
        print(f"  {k:<25} {v:.4f}")
