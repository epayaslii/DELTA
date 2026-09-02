"""Generate transcripts/<vid>.txt (ordered action list) from groundTruth/<vid>.txt
by run-length-collapsing consecutive labels.

ATBA and HAL both read transcripts/ from disk; the dinggd/50salads bundle does
not ship them.

    python scripts/make_transcripts.py data/50salads
"""
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/50salads")
gt_dir = root / "groundTruth"
tr_dir = root / "transcripts"
tr_dir.mkdir(exist_ok=True)

n = 0
for gt in sorted(gt_dir.glob("*.txt")):
    lines = [ln.strip() for ln in gt.read_text().splitlines() if ln.strip()]
    if not lines:
        continue
    tr = [lines[0]]
    for x in lines[1:]:
        if x != tr[-1]:
            tr.append(x)
    (tr_dir / gt.name).write_text("\n".join(tr) + "\n")
    n += 1

print(f"wrote {n} transcripts -> {tr_dir}")
