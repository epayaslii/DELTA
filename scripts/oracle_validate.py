"""Full-pipeline oracle validation: ASOT + local boundary search, fed a
near-perfect similarity, across all 5 splits, 19-class protocol (canonical).
"""
import numpy as np
from delta.data import ActionSegDataset
from delta.align import segmentation_report, align_asot, refine_boundaries, segments

ds = ActionSegDataset('50salads', 'data/50salads')


def oracle_sim(rec, jitter, seed):
    T, N = rec.num_label_frames, len(rec.transcript)
    rng = np.random.default_rng(seed)
    s = rng.normal(0, jitter, size=(N, T)) if jitter else np.zeros((N, T))
    for n, (_, a, b) in enumerate(segments(rec.frame_labels)):
        s[n, a:b] += 1.0
    return s


def run(jitter, use_refine, alpha=0.1, rho=0.5, seed=0, radius=90, window=40):
    rows = []
    for sp in (1, 2, 3, 4, 5):
        for v in ds.split(sp, 'test'):
            rec = ds.record(v)
            s = oracle_sim(rec, jitter, seed)           # (N, T)
            C = (1.0 - s).T                              # (T, N) cost
            ar = align_asot(C, rec.transcript, alpha=alpha, rho=rho)
            y = ar.y_star
            if use_refine:
                e = np.cumsum(np.r_[0, np.diff(y) != 0])
                r = refine_boundaries(s, e, radius=radius, window=window, w_visual=0.0)
                y = np.asarray(rec.transcript, int)[r.entry_of_frame]
            rows.append(segmentation_report(y, rec.frame_labels, ignore=set()))
    m = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    return m


for jitter in (0.0, 0.3, 1.0, 2.0):
    m = run(jitter, use_refine=False)
    print(f"ASOT only,   jitter={jitter:<4} MoF {m['MoF']:.3f}  MoC {m['MoC']:.3f}  F1@50 {m['F1@50']:.1f}")
for jitter in (0.0, 0.3, 1.0, 2.0):
    m = run(jitter, use_refine=True)
    print(f"ASOT+refine, jitter={jitter:<4} MoF {m['MoF']:.3f}  MoC {m['MoC']:.3f}  F1@50 {m['F1@50']:.1f}")
