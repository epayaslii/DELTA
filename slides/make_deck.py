"""Generate the progress deck for Mariella. python-pptx, 16:9."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
ACCENT = RGBColor(0x1F, 0x6F, 0x4D)      # herb green
FLAG = RGBColor(0xB4, 0x53, 0x09)        # amber
INK = RGBColor(0x22, 0x26, 0x2B)
MUTED = RGBColor(0x5B, 0x64, 0x70)
LIGHT = RGBColor(0xEC, 0xEE, 0xF1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

FONT = "Calibri"


def _tf(box):
    tf = box.text_frame
    tf.word_wrap = True
    return tf


def bg(slide, color=WHITE):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def title_bar(slide, text, sub=None):
    box = slide.shapes.add_textbox(Inches(0.55), Inches(0.35), Inches(12.2), Inches(0.95))
    tf = _tf(box)
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.size = Pt(30); r.font.bold = True; r.font.name = FONT; r.font.color.rgb = NAVY
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run(); r2.text = sub
        r2.font.size = Pt(14); r2.font.name = FONT; r2.font.color.rgb = MUTED
    # rule
    ln = slide.shapes.add_shape(1, Inches(0.55), Inches(1.45), Inches(12.2), Pt(2))
    ln.fill.solid(); ln.fill.fore_color.rgb = ACCENT; ln.line.fill.background()
    return slide


def bullets(title, items, sub=None, body_top=1.75, size=17):
    """items: list of (level, text) or plain str (level 0). '' text -> spacer."""
    s = prs.slides.add_slide(BLANK); bg(s)
    title_bar(s, title, sub)
    box = s.shapes.add_textbox(Inches(0.7), Inches(body_top), Inches(12.0), Inches(5.4))
    tf = _tf(box)
    first = True
    for it in items:
        lvl, txt = it if isinstance(it, tuple) else (0, it)
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = lvl
        p.space_after = Pt(6)
        if txt == "":
            p.add_run().text = " "; p.space_after = Pt(4); continue
        bullet = "▸ " if lvl == 0 else ("– " if lvl == 1 else "· ")
        r = p.add_run(); r.text = bullet + txt
        r.font.name = FONT
        r.font.size = Pt(size if lvl == 0 else size - 2 if lvl == 1 else size - 3)
        r.font.color.rgb = INK if lvl == 0 else MUTED
        if lvl == 0:
            r.font.bold = False
    return s


def table_slide(title, headers, rows, sub=None, col_widths=None, font=12):
    s = prs.slides.add_slide(BLANK); bg(s)
    title_bar(s, title, sub)
    nr, nc = len(rows) + 1, len(headers)
    gt = s.shapes.add_table(nr, nc, Inches(0.6), Inches(1.8), Inches(12.1), Inches(0.4 * nr)).table
    if col_widths:
        for i, w in enumerate(col_widths):
            gt.columns[i].width = Inches(w)
    for j, h in enumerate(headers):
        c = gt.cell(0, j); c.text = h
        c.fill.solid(); c.fill.fore_color.rgb = NAVY
        pr = c.text_frame.paragraphs[0]; pr.runs[0].font.size = Pt(font + 1)
        pr.runs[0].font.bold = True; pr.runs[0].font.color.rgb = WHITE; pr.runs[0].font.name = FONT
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = gt.cell(i, j); c.text = str(val)
            c.fill.solid(); c.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
            for pr in c.text_frame.paragraphs:
                for rn in pr.runs:
                    rn.font.size = Pt(font); rn.font.name = FONT; rn.font.color.rgb = INK
    return s


def section(title, kicker=None):
    s = prs.slides.add_slide(BLANK); bg(s, NAVY)
    box = s.shapes.add_textbox(Inches(0.8), Inches(2.9), Inches(11.7), Inches(1.6))
    tf = _tf(box)
    if kicker:
        p = tf.paragraphs[0]; r = p.add_run(); r.text = kicker.upper()
        r.font.size = Pt(14); r.font.name = FONT; r.font.color.rgb = RGBColor(0x9D, 0xD3, 0xBE)
        r.font.bold = True
        p2 = tf.add_paragraph()
    else:
        p2 = tf.paragraphs[0]
    r2 = p2.add_run(); r2.text = title
    r2.font.size = Pt(34); r2.font.bold = True; r2.font.name = FONT; r2.font.color.rgb = WHITE
    return s


# ------------------------------------------------------------------ TITLE
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
band = s.shapes.add_shape(1, 0, 0, Inches(13.333), Inches(2.5))
band.fill.solid(); band.fill.fore_color.rgb = NAVY; band.line.fill.background()
box = s.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(1.5))
tf = _tf(box)
r = tf.paragraphs[0].add_run()
r.text = "Temporal Alignment for DELTA"
r.font.size = Pt(40); r.font.bold = True; r.font.name = FONT; r.font.color.rgb = WHITE
p = tf.add_paragraph(); r = p.add_run()
r.text = "Progress, findings, and directions for a VLM-based approach"
r.font.size = Pt(18); r.font.name = FONT; r.font.color.rgb = RGBColor(0xC9, 0xD6, 0xE3)
box = s.shapes.add_textbox(Inches(0.8), Inches(3.1), Inches(11.7), Inches(2.5))
tf = _tf(box)
for i, line in enumerate([
    "Eliz Payasli — research internship, IRI / UPC",
    "Advisor: Mariella Dimiccoli",
    "Base work: DELTA — Dense Long-Term Action Anticipation from Procedural Transcripts",
    "Repository: github.com/epayaslii/DELTA",
]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    rn = p.add_run(); rn.text = line
    rn.font.size = Pt(15 if i < 2 else 13); rn.font.name = FONT
    rn.font.color.rgb = INK if i < 2 else MUTED
    if i == 0:
        rn.font.bold = True

# ------------------------------------------------------------------ 1. WHERE WE ARE
bullets("Where we are — one slide", [
    (0, "Goal: transcript-only Dense Long-Term Action Anticipation (DLTA); focus = the "
        "Temporal Alignment (TA) component, first on 50Salads."),
    (0, "Direction (per your guidance): a fundamentally different TA — align via a "
        "vision-language model, not via segmentation the way ATBA does."),
    (0, "Done: research repo + infrastructure, 50Salads analysis, a VLM-direct aligner "
        "prototype (21 tests); analysed the DELTA code (received) and the two CVPR'26 "
        "baselines HAL and CVA."),
    (0, "Workstream split: Eliz → HAL (integrate + measure on DLTA); co-intern → CVA (the VLM alignment ideas)."),
    (0, "Key blocker: raw 50Salads video is unavailable (official host down) — needed for VLM features."),
    (0, "This deck: what's built, what we learned, the phased TA plan, and questions — for your input."),
], sub="Internship progress review")

# ------------------------------------------------------------------ 2. PROBLEM RECAP
bullets("The problem, briefly", [
    (0, "DLTA: from an observed video prefix, predict the future actions, their order, and durations — densely."),
    (0, "DELTA learns this from transcripts only (ordered action list, no timestamps, no frame labels)."),
    (0, "Its Temporal Alignment module turns the transcript into dense pseudo-labels Y* that supervise "
        "segmentation and the anticipation decoder."),
    (0, "DELTA's TA follows ATBA (CVPR'24). On 50Salads DELTA trails fully-supervised methods:"),
    (1, "avg MoC 20.9 (DELTA)  vs  ~25.9 (FUTR)  vs  ~28.4 (ActFusion)."),
    (0, "If the pseudo-labels are poor, everything downstream is poor. So TA quality is the lever."),
], sub="DLTA · transcript-only · why TA matters")

# ------------------------------------------------------------------ 3. FRAMING
bullets("The research framing", [
    (0, "ATBA / HAL — \"alignment THROUGH segmentation\":"),
    (1, "frozen I3D  →  trained frame classifier  →  posteriors P  →  boundary + transition scores  →  DP  →  Y*"),
    (1, "the alignment can be no better than that classifier at telling the actions apart."),
    (0, "On 50Salads that classifier is a near-worst case: fixed overhead camera, near-duplicate "
        "fine-grained actions (cut_tomato / cut_cheese, add_oil / add_vinegar)."),
    (0, "Our direction — \"alignment THROUGH semantic matching\" (and CVA proves it works):"),
    (1, "frozen VLM:  s(n, t) = sim( text(action_n), frame_t )  →  monotonic / OT alignment  →  Y*"),
    (1, "no frame classifier in the alignment path; the fine-grained vocabulary is disambiguated by "
        "the noun in the label, from step 0."),
    (1, "CVA (CVPR'26) does exactly this for video temporal grounding and is SOTA (+5 R@1)."),
], sub="Two ways to recover temporal structure from a transcript")

# ------------------------------------------------------------------ 4. THE REPO
table_slide("What's built — the repository",
    ["Module", "What it does", "Status"],
    [
        ["delta.data", "50Salads / Breakfast conventions, transcripts, splits, dataset stats", "done + tests"],
        ["delta.features", "Frozen VLM feature extraction — VideoLLaMA3 / SigLIP2 / DINOv2 backbones, text encoder, extraction CLI", "done (untested on video)"],
        ["delta.align", "similarity matrix · order-preserving DP + soft alignment · segmentation/alignment metrics", "done + tests"],
        ["delta.viz", "segmentation-timeline plotting (GT vs pseudo-labels vs prediction)", "done"],
        ["docs/", "TA reference · 50Salads analysis · DELTA-code deep-read (loss assembly, VLM plug-in points) · HAL & CVA baselines · approach + lit review", "8 documents"],
        ["tests/", "CPU-only unit tests — synthetic + real-transcript validation", "21 passing"],
    ],
    sub="github.com/epayaslii/DELTA  ·  Python 3.11 / uv",
    col_widths=[2.1, 8.0, 2.0], font=11)

# ------------------------------------------------------------------ 5. PROGRESS: INFRA + DATA
bullets("Progress 1 — infrastructure & data", [
    (0, "Feature-extraction pipeline: pluggable frozen backbones (VL3-SigLIP = VideoLLaMA3 vision tower, "
        "SigLIP2, DINOv2), outputs (D,T) features aligned to the ground-truth frame grid — drop-in for I3D."),
    (0, "50Salads benchmark bundle downloaded & verified (HuggingFace dinggd/50salads):"),
    (1, "50 videos, 19 classes (17 actions + start/end), I3D 2048-d, confirmed 30 fps, ~20 segments/video."),
    (0, "Raw 50Salads videos: the official host (Dundee) is DOWN (NXDOMAIN everywhere); no public mirror."),
    (1, "Blocks VLM feature extraction. Need a copy from the lab — see questions at the end."),
    (0, "Environment: local Mac has no GPU/torch → model runs go to the UPC cluster."),
], sub="pipeline · dataset · the video blocker")

# ------------------------------------------------------------------ 6. PROGRESS: 50S ANALYSIS
table_slide("Progress 2 — why 50Salads is hard (measured)",
    ["Finding", "Number", "Implication"],
    [
        ["Naive uniform alignment (zero visual evidence)", "MoC 0.34", "the floor any TA must beat"],
        ["I3D consecutive-frame distance at GT boundaries\nvs at random frames", "1.11×", "class-agnostic boundary signal is\nnearly absent → ATBA's candidate\nstep misses true transitions"],
        ["Next-action entropy (transcript order)", "~1.9 bits\n(~3–4 successors)", "order alone doesn't pin timing —\nvisual evidence must carry it"],
        ["Per-class segment-duration variation", "CV 0.5–0.8", "duration head has weak signal\n(paper: +0.2 MoC on 50S vs +3.3 BF)"],
    ],
    sub="Stage 1 dataset analysis  ·  notebook + docs/50salads-notes.md",
    col_widths=[4.6, 2.3, 5.2], font=11)

# ------------------------------------------------------------------ 7. PROGRESS: VLM ALIGNER
bullets("Progress 3 — VLM-direct aligner prototype", [
    (0, "delta.align.similarity — transcript × frame cosine similarity from frozen VLM embeddings "
        "(model-agnostic; takes pre-extracted embeddings)."),
    (0, "delta.align.ta — two solvers:"),
    (1, "align_dp: hard order-preserving DP (monotone, every transcript action covered, transition penalty)."),
    (1, "align_soft: entropic forward–backward → per-frame posteriors AND P(boundary_r = t) distributions "
        "(for uncertainty-aware targets later)."),
    (0, "delta.align.evaluate — swap-in similarity providers (naive / oracle / VLM), score Y* vs held-out GT."),
    (0, "Validated align_dp on real 50Salads transcripts (up to 26 segments, repeated classes): "
        "near-perfect recovery from clean/noisy blocky similarity."),
    (0, "Missing piece: the real VLM similarity matrix → needs frame features → needs raw video."),
], sub="the method, minus the VLM  ·  21 tests")

# ------------------------------------------------------------------ 8. PROGRESS: DELTA CODE
bullets("Progress 4 — the DELTA code (\"WLTA\")", [
    (0, "Received from the group; analysed in docs/delta-code.md. WLTA = Weakly-supervised Long-Term Anticipation."),
    (0, "Built on the CLOT / ASOT codebase — so it already contains BOTH alignment mechanisms:"),
    (1, "--model_type atba  → the ATBA boundary detector (src/atba_loss.py)"),
    (1, "--model_type wclot → ASOT optimal transport (src/asot.py, gsw.py)"),
    (1, "plus CTC, TSM smoothing, cross-modal attention, linear-chain CRF, DistilBERT, full LTA eval."),
    (0, "The \"situation\": README is CLOT's; run scripts call a missing train_edit_elena9sept.py; "
        "hardcoded paths; needs Linux + CUDA + conda + wandb."),
    (0, "Implication: our contribution narrows cleanly — swap the frame-classifier posteriors / OT cost "
        "matrix for a frozen-VLM similarity; reuse everything downstream. And we can run the real baseline."),
], sub="what it is · how it runs · what it means for us")

# ------------------------------------------------------------------ 9. PROGRESS: HAL
bullets("Progress 5 — HAL (CVPR'26), my focus", [
    (0, "HAL = Hierarchical Action Learning. Verdict (confirmed from the CVPR PDF): "
        "HAL is NOT a different TA — it IS ATBA plus a small variational regulariser."),
    (1, "models/model.py: \"adapted from CVPR24_ATBA\". Boundary detector = byte-for-byte ATBA."),
    (1, "L_total = L_y^ATBA  −  α·ELBO  +  β·L_s"),
    (1, "ELBO = VAE reconstruction + KL on a two-scale latent (slow \"action\" z1 / fast \"visual\" z2); "
        "L_s = a change-rate penalty forcing z1 to evolve slower than z2."),
    (0, "Gains over ATBA: +2.4 MoF Breakfast, +3.3 Hollywood, +3.0 GTEA — modest, some within std."),
    (0, "Never evaluated 50Salads (loader rejects it). Reports MoF/IoU only — no edit, F1, boundary metric."),
    (0, "Open question nobody has answered: does HAL's hierarchy + smoothness help "
        "dense anticipation (MoC), or only segmentation (MoF)?  ← my workstream."),
], sub="ATBA + a hierarchy/smoothness prior  ·  same task as us")

# ------------------------------------------------------------------ 10. CVA
bullets("Progress 6 — CVA (CVPR'26), the co-intern's focus", [
    (0, "CVA = Context-aware Video-text Alignment. Task: video temporal grounding "
        "(NL query → one time span), fully supervised, SlowFast + CLIP features. SOTA on "
        "QVHighlights / Charades-STA / TACoS (+5 R@1)."),
    (0, "Different task, but it IS \"align through a VLM, done right\". Three components:"),
    (1, "QCD — query-aware background-mix augmentation (context robustness)."),
    (1, "CTE — hierarchical encoder: windowed self-attn + learnable global queries + bidirectional cross-attn."),
    (1, "CBD loss — boundary-focused contrastive: boundary-frame reps invariant to context, "
        "contrasted vs adjacent + most-similar background."),
    (0, "Not a drop-in baseline (wrong task/supervision) — the methodological reference. "
        "CBD and CTE are directly transferable to our aligned pseudo-boundaries."),
], sub="the strongest VLM video-text aligner  ·  adjacent task")

# ------------------------------------------------------------------ 11. THE TWO AXES + SPLIT
bullets("Scoping — two axes, split between us", [
    (0, "Axis 1  —  transcript → dense labels  (our task's TA stage):"),
    (1, "classifier-based alignment:  ATBA (CVPR'24)  →  HAL (CVPR'26)"),
    (0, "Axis 2  —  NL query → span  (adjacent, supervised):"),
    (1, "VLM semantic alignment:  CVA (CVPR'26, SOTA)"),
    (0, ""),
    (0, "Eliz → HAL:  reproduce, understand, integrate into DELTA's TA, measure the effect on DLTA MoC."),
    (0, "Co-intern → CVA:  the VLM video-text-alignment angle (CBD, CTE, semantic similarity)."),
    (0, "Both feed the DELTA VLM-direct method: HAL brings structure/smoothness, CVA brings the VLM alignment."),
], sub="Eliz → HAL   ·   co-intern → CVA")

# ------------------------------------------------------------------ 12. TAKEAWAYS
bullets("Key takeaways", [
    (0, "The bottleneck is \"alignment through segmentation\" — a trained classifier that 50Salads breaks."),
    (0, "HAL confirms it: the CVPR'26 SOTA is still ATBA-based, +2–3 MoF, never touches 50Salads."),
    (0, "DELTA already has optimal-transport alignment (ASOT, --model_type wclot) — the VLM swap is "
        "≈ one line in the cost matrix; everything downstream (CTC, CRF, LTA decoder, eval) is reused."),
    (0, "CVA shows VLM video-text alignment + a boundary-contrastive loss is SOTA on the adjacent task."),
    (0, "Our contribution: bring that into transcript-supervised DLTA, where ATBA/HAL currently win."),
], sub="what the analysis established")

# ------------------------------------------------------------------ 13. BLOCKERS
table_slide("Blockers & risks",
    ["Item", "Impact", "Mitigation"],
    [
        ["Raw 50Salads video unavailable", "blocks VLM feature extraction (VLM phases)", "lab copy; HAL + baselines run now on I3D"],
        ["No local GPU / torch", "no model runs on the Mac", "cluster; one conda env for ATBA / HAL / WLTA"],
        ["MoF ≠ MoC", "a method can beat ATBA on MoF and not help DELTA's anticipation MoC", "always measure pseudo-label MoC + boundary offset, not just MoF"],
        ["DELTA / HAL code is research-grade", "reproduction friction (missing script, paths, wandb)", "documented in docs/delta-code.md; map run scripts onto train.py"],
        ["HAL has no 50Salads config", "can't reproduce a reference number there", "reproduce on Breakfast; add a 50S config as a task (H4)"],
    ],
    sub="known before we commit compute",
    col_widths=[3.2, 4.6, 4.3], font=11)

# ------------------------------------------------------------------ 14. SECTION
section("How we proceed with the TA", kicker="For discussion")

# ------------------------------------------------------------------ 15. PHASED TA PLAN
table_slide("The TA plan — phases (shared)",
    ["Phase", "Work", "Needs", "Output"],
    [
        ["0", "Baselines: DELTA/WLTA --model_type atba + wclot on Breakfast & 50S; extract pseudo-labels; measure boundary quality", "cluster, I3D", "real MoC; the reference table"],
        ["1", "Frozen VLM frame features + action-name text embeddings for 50S; build s (N×T); diagnostics", "raw video", "s; zero-shot confusion; boundary peakedness"],
        ["2", "Plug s into DELTA's alignment (wclot cost matrix ~1 line, or atba boundary detector)", "phase 1", "Y* quality vs ATBA / HAL / naive floor"],
        ["3", "Borrow from CVA: CBD boundary-contrastive loss on aligned boundaries; CTE-style encoder on VLM feats", "phase 2", "which CVA idea helps"],
        ["4", "Best Y* → DELTA decoder; Obs%/Pred% MoC grid", "phase 3", "the result vs ATBA / HAL / DELTA"],
    ],
    sub="phase 0 runs now on I3D; phases 1–4 need raw 50Salads video",
    col_widths=[0.7, 6.6, 1.9, 3.3], font=10.5)

# ------------------------------------------------------------------ 16. ELIZ / HAL PLAN
table_slide("My workstream — HAL (H1–H7)",
    ["#", "Step", "Gate / output"],
    [
        ["H1", "Env + Breakfast features; reproduce ATBA split 1", "≈53.9 MoF — pipeline works"],
        ["H2", "Reproduce HAL split 1 (README command)", "≈56.3 MoF — HAL reproduces"],
        ["H3", "Dump ATBA pseudo-labels + z1/z2; score Y* with delta.align (MoC, edit, F1@k, boundary offset)", "does HAL give better pseudo-labels, or only better final MoF?"],
        ["H4", "Get HAL running on 50Salads (add config, splits, transcripts; widen loader assert)", "first HAL 50Salads number"],
        ["H5", "Port HAL into DELTA — VAE tap + recon/kl/diff losses on --model_type atba (~70 lines)", "DELTA-atba+HAL"],
        ["H6", "DELTA-atba vs DELTA-atba+HAL on 50S + Breakfast → Obs%/Pred% MoC; ablate the 3 terms", "the result (+ or −)"],
        ["H7", "If diff_loss (smoothness) helps → hand it to the VLM-aligner workstream", "shared component"],
    ],
    sub="expected magnitude small — but it's the baseline table, and a clean negative is a real finding",
    col_widths=[0.6, 8.0, 3.5], font=10)

# ------------------------------------------------------------------ 17. THE VLM METHOD
bullets("The DELTA VLM-direct method (where both workstreams land)", [
    (0, "s(n,t) = sim( VLM_text(action_n), VLM_frame_t )  — frozen; no frame classifier in the alignment path."),
    (0, "Y* = order-preserving DP / optimal-transport alignment on s  (ASOT is already in DELTA)."),
    (0, "+ from CVA:  a CBD-style boundary-contrastive loss on the aligned boundary frames;  "
        "a CTE-style windowed + cross-attn encoder on the VLM features."),
    (0, "+ from HAL:  the L_s change-rate / smoothness penalty on the soft assignment sequence  "
        "(anti over-segmentation, cheaper than HAL's VAE)."),
    (0, "Feed Y* into DELTA's decoder + CRF + duration head (unchanged) → Obs%/Pred% MoC."),
    (0, "Baselines: naive 0.34 · ATBA-in-DELTA · ASOT-in-DELTA · HAL · warm-up-classifier ceiling."),
], sub="frozen VLM similarity → alignment → DELTA  ·  regularised by CVA + HAL ideas")

# ------------------------------------------------------------------ 18. SEQUENCING
table_slide("Sequencing",
    ["When", "Eliz (HAL)", "Co-intern (CVA)", "Shared"],
    [
        ["Now", "H1–H2: reproduce ATBA + HAL on Breakfast (cluster)", "read CVA code; reproduce on QVHighlights", "chase raw 50S video"],
        ["Wk 1–2", "H3–H4: score pseudo-labels; HAL on 50Salads", "isolate CBD + CTE as reusable modules", "phase 0: DELTA/WLTA baselines"],
        ["Wk 3–4", "H5–H6: HAL in DELTA; MoC ablation", "CBD/CTE on our VLM features (once video)", "phase 1: VLM s for 50S"],
        ["Wk 4+", "H7: smoothness term → VLM aligner", "CBD/CTE → VLM aligner", "phases 2–4: VLM-direct alignment → DLTA MoC"],
    ],
    sub="two parallel tracks converging on the DELTA VLM-direct method",
    col_widths=[1.0, 3.9, 3.9, 3.3], font=9.5)

# ------------------------------------------------------------------ 19. QUESTIONS
bullets("Questions for you", [
    (0, "Can the group share raw 50Salads video? (blocks VLM features; official host is down.)"),
    (0, "HAL workstream: is the goal to integrate it into DELTA (H5–H6), or just a standalone "
        "reproduction + comparison? Is a negative result (\"doesn't transfer to anticipation\") acceptable?"),
    (0, "Primary target metric — alignment Y* quality on 50Salads, or downstream DLTA MoC?"),
    (0, "Is establishing a 50Salads transcript-only alignment benchmark (ATBA & HAL skip it) a contribution?"),
    (0, "Frozen VLM only, or is fine-tuning the alignment cost (light adapter) in scope?"),
    (0, "Preferred VLM backbone — VideoLLaMA3 vision tower, InternVideo2, CLIP, or something the group uses?"),
    (0, "Do we have the DELTA supplementary material (loss weights, decoder hyperparameters)?"),
], sub="to set the next phase")

# ------------------------------------------------------------------ CLOSE
s = section("Thank you", kicker="github.com/epayaslii/DELTA")
box = s.shapes.add_textbox(Inches(0.8), Inches(4.6), Inches(11.7), Inches(1.2))
tf = _tf(box)
r = tf.paragraphs[0].add_run()
r.text = ("Docs: temporal-alignment.md · 50salads-notes.md · delta-code.md · "
          "baselines-hal-cva.md · hal-analysis.md · approach.md")
r.font.size = Pt(13); r.font.name = FONT; r.font.color.rgb = RGBColor(0xC9, 0xD6, 0xE3)

out = "/Users/elizpayasli/Documents/GitHub/DELTA/slides/DELTA_progress_Mariella.pptx"
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
prs.save(out)
print("saved", out, "-", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
