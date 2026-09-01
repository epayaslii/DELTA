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
    (0, "Goal: transcript-only Dense Long-Term Action Anticipation (DLTA); my focus is the "
        "Temporal Alignment (TA) component, first on 50Salads."),
    (0, "Direction (per your guidance): investigate a fundamentally different TA — "
        "align via a vision-language model, not via segmentation the way ATBA does."),
    (0, "Done so far:"),
    (1, "Research repo + infrastructure, dataset analysis, a first VLM-direct aligner prototype (21 tests)."),
    (1, "Analysed the DELTA code (received) and HAL (the recent CVPR baseline)."),
    (0, "Key blocker: raw 50Salads video is unavailable (official host is down) — needed for VLM features."),
    (0, "This deck: what's built, what we learned, and 5 concrete ways to proceed — for your input."),
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
    (0, "Our direction — \"alignment THROUGH semantic matching\":"),
    (1, "frozen VLM:  s(n, t) = sim( text(action_n), frame_t )  →  monotonic / OT alignment  →  Y*"),
    (1, "no frame classifier in the alignment path; the fine-grained vocabulary is disambiguated by "
        "the noun in the label, from step 0."),
], sub="Two ways to recover temporal structure from a transcript")

# ------------------------------------------------------------------ 4. THE REPO
table_slide("What's built — the repository",
    ["Module", "What it does", "Status"],
    [
        ["delta.data", "50Salads / Breakfast conventions, transcripts, splits, dataset stats", "done + tests"],
        ["delta.features", "Frozen VLM feature extraction — VideoLLaMA3 / SigLIP2 / DINOv2 backbones, text encoder, extraction CLI", "done (untested on video)"],
        ["delta.align", "similarity matrix · order-preserving DP + soft alignment · segmentation/alignment metrics", "done + tests"],
        ["delta.viz", "segmentation-timeline plotting (GT vs pseudo-labels vs prediction)", "done"],
        ["docs/", "TA reference (ATBA-grounded) · 50Salads analysis · DELTA-code analysis · approach + lit review", "6 documents"],
        ["tests/", "CPU-only unit tests — synthetic + real-transcript validation", "21 passing"],
    ],
    sub="github.com/epayaslii/DELTA  ·  7 commits  ·  Python 3.11 / uv",
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
bullets("Progress 5 — HAL analysis (the recent CVPR baseline)", [
    (0, "HAL = Hierarchical Action Learning, CVPR 2026. Verdict: HAL is NOT a different TA — "
        "it IS ATBA plus a small variational regulariser."),
    (1, "models/model.py: \"adapted from CVPR24_ATBA\". The boundary detector is byte-for-byte ATBA."),
    (1, "HAL adds a two-scale VAE branch (slow/fast latents) + 3 aux losses at weights 0.1 / 1e-3 / 1e-3."),
    (0, "Gains over ATBA: +2.4 MoF Breakfast, +3.3 Hollywood, +3.0 GTEA — modest, some within noise."),
    (0, "HAL never evaluated 50Salads; its data loader rejects it. Exposes no boundary/alignment output."),
    (0, "It still aligns THROUGH the frame classifier → does not advance the VLM direction."),
    (0, "Use HAL as: baseline #2 (ATBA = #1, naive = floor) — a cheap ablation, not a research thrust."),
], sub="ATBA + hierarchy prior  ·  strong baseline, not a TA replacement")

# ------------------------------------------------------------------ 10. TAKEAWAYS
bullets("Key takeaways", [
    (0, "The bottleneck is \"alignment through segmentation\" — a trained classifier that 50Salads breaks."),
    (0, "HAL confirms the framing: the SOTA WSAS method is still ATBA-based, +2–3 MoF, skips 50Salads."),
    (0, "DELTA already has optimal-transport alignment (ASOT) — our monotonic/OT aligner is partly there."),
    (0, "So the VLM contribution is well-scoped: replace the posteriors feeding the aligner with a "
        "frozen-VLM transcript×frame similarity; everything downstream is reused."),
    (0, "Independent mini-contribution available: a boundary-localisation benchmark on 50Salads "
        "(which ATBA and HAL both skip)."),
], sub="what the analysis established")

# ------------------------------------------------------------------ 11. BLOCKERS
table_slide("Blockers & risks",
    ["Item", "Impact", "Mitigation"],
    [
        ["Raw 50Salads video unavailable", "blocks VLM feature extraction (Stage 2b onward)", "lab copy; or start on Breakfast; or I3D-only baselines now"],
        ["No local GPU / torch", "no model runs on the Mac", "cluster; conda env shared with ATBA/HAL/WLTA"],
        ["MoF ≠ MoC", "a method can beat ATBA on MoF and not help DELTA's MoC / pseudo-transcripts", "always measure pseudo-label MoC + boundary offset, not just MoF"],
        ["DELTA code is research-grade", "reproduction friction (missing script, paths, wandb)", "documented; map run scripts onto train.py"],
        ["VLM fine-grained weakness", "frame-CLIP struggles on cut_* / add_* classes", "video-native VLM (VideoLLaMA3 / InternVideo2); measure zero-shot confusion first"],
    ],
    sub="known before we commit compute",
    col_widths=[3.2, 4.6, 4.3], font=11)

# ------------------------------------------------------------------ 12. WAYS FORWARD (SECTION)
section("Five ways we can progress", kicker="For discussion")

# ------------------------------------------------------------------ 13. PATHS OVERVIEW
table_slide("Ways forward — overview",
    ["Path", "What", "Needs", "Risk / cost"],
    [
        ["A", "Reproduce baselines: ATBA + DELTA(WLTA) on Breakfast & 50S; extract pseudo-labels; measure boundary quality", "cluster; features", "low — needed anyway"],
        ["B", "HAL ablation: add HAL's 3 losses to DELTA, one training run, check MoC / pseudo-label quality", "Path A first", "low; bounded"],
        ["C", "VLM-direct alignment (the contribution): frozen VLM similarity → monotonic/OT → Y* → DELTA decoder", "raw video; cluster", "medium — the research"],
        ["D", "VLM zero-shot / pseudo-annotator: video-LLM scores clips vs transcript vocabulary; auxiliary signal", "raw video (few clips)", "low–medium; exploratory"],
        ["E", "50Salads boundary-localisation benchmark (ATBA & HAL both skip it) — standalone contribution", "50S GT (have it)", "low; publishable in itself"],
    ],
    sub="not mutually exclusive — the question is ordering",
    col_widths=[0.6, 6.3, 2.4, 2.8], font=10.5)

# ------------------------------------------------------------------ 14. PATH A / B DETAIL
bullets("Paths A & B — establish the ground truth first", [
    (0, "A — Baselines (cluster, ~1–2 weeks):"),
    (1, "run ATBA standalone on Breakfast split 1 → confirm ≈53.9 MoF (pipeline works)."),
    (1, "run DELTA(WLTA) run_50S_allmetrics.sh → real 50Salads MoC, both --model_type atba and wclot."),
    (1, "instrument the loop to dump per-video pseudo-labels + boundaries."),
    (1, "score with our delta.align: MoC, edit, F1@k, median per-transition boundary offset."),
    (0, "B — HAL ablation (after A, ~3 days):"),
    (1, "port HAL's VAE tap + recon/KL/diff losses into DELTA's encoder (small, additive)."),
    (1, "one Breakfast + one 50S run. Gate: does pseudo-label MoC improve beyond seed noise (±1)?"),
    (1, "if no → drop HAL, cite the numbers, move on. If yes → keep as the strong baseline."),
], sub="low-risk, high-information — the reference for everything after")

# ------------------------------------------------------------------ 15. PATH C DETAIL
bullets("Path C — VLM-direct alignment (the research contribution)", [
    (0, "1. Extract frozen VLM frame features for 50Salads (VideoLLaMA3 vision tower / InternVideo2) + "
        "text embeddings for the 17 action names (bare label and/or generated descriptions)."),
    (0, "2. Build s(n,t) = cos(text_n, frame_t); diagnostics: zero-shot confusion vs GT, "
        "is s peaked at true boundaries (vs I3D's 1.11×)?"),
    (0, "3. Read Y* off s with our order-preserving DP / OT aligner — no frame classifier."),
    (0, "4. Feed Y* into DELTA's pipeline (TAS + CTC + CRF + LTA decoder unchanged); "
        "evaluate the Obs%/Pred% MoC grid vs ATBA, HAL, DELTA."),
    (0, "5. Extensions: uncertainty-aware boundaries (from align_soft), fine-tune the alignment cost "
        "with a light adapter + temporal cycle-consistency."),
    (0, "Baselines to beat: naive 0.34 MoC · ATBA-Y* · HAL-Y* · warm-up-classifier ceiling."),
], sub="frozen VLM → similarity → monotonic alignment → DELTA")

# ------------------------------------------------------------------ 16. PATH D / E DETAIL
bullets("Paths D & E — smaller bets", [
    (0, "D — VLM as pseudo-annotator / zero-shot (exploratory):"),
    (1, "prompt a video-LLM (VideoLLaMA3) on short clips: \"which of these 17 actions?\" or free caption."),
    (1, "use as (i) an extra soft label to regularise Y*, (ii) auto-generated action descriptions for the text tower."),
    (1, "cheap to try on a handful of clips; tells us how much semantic signal a VLM actually has here."),
    (0, "E — 50Salads boundary-localisation benchmark (standalone):"),
    (1, "ATBA and HAL report MoF / IoU but never boundary offset, and both skip 50Salads."),
    (1, "a clean transcript→frame boundary benchmark on 50S is a small contribution on its own, "
        "and it is the yardstick every alignment method (incl. the VLM one) is measured against."),
], sub="low cost, decouplable from the video blocker (E needs only GT)")

# ------------------------------------------------------------------ 17. RECOMMENDED SEQUENCING
table_slide("Recommended sequencing",
    ["When", "Work", "Gate / output"],
    [
        ["Now", "E: boundary-benchmark spec on 50S (GT only, no video)  +  chase raw video", "metric code + the ask to the lab"],
        ["Wk 1–2", "A: ATBA repro on Breakfast; DELTA(WLTA) baseline on 50S (cluster)", "real MoC numbers; pseudo-label dumps"],
        ["Wk 3", "B: HAL ablation in DELTA — one run", "go/no-go on HAL (pseudo-label MoC)"],
        ["Wk 3–4", "VLM feature extraction for 50S (once video arrives)", "features_vl3siglip/ + text embeddings"],
        ["Wk 4–6", "C: VLM similarity → aligner → Y* quality vs all baselines", "the key comparison table"],
        ["Wk 6+", "C: integrate best Y* into DELTA; Obs%/Pred% MoC; then Breakfast", "the result"],
    ],
    sub="A/B in parallel with getting video; C is the thrust",
    col_widths=[1.4, 6.9, 3.8], font=11)

# ------------------------------------------------------------------ 18. QUESTIONS
bullets("Questions for you", [
    (0, "Can the group share raw 50Salads video? (blocks VLM features; official host is down.)"),
    (0, "Primary target metric — alignment Y* quality on 50Salads, or downstream DLTA MoC?"),
    (0, "Is establishing a 50Salads transcript-only alignment benchmark (ATBA & HAL skip it) "
        "itself a contribution we should claim?"),
    (0, "Frozen VLM only, or is fine-tuning the alignment cost (light adapter) in scope?"),
    (0, "Preferred VLM backbone — VideoLLaMA3 vision tower, InternVideo2, or something the group uses?"),
    (0, "HAL: worth the ablation, or skip straight to the VLM work?"),
    (0, "Do we have the DELTA supplementary material (loss weights, decoder hyperparameters)?"),
], sub="to set the next phase")

# ------------------------------------------------------------------ CLOSE
s = section("Thank you", kicker="github.com/epayaslii/DELTA")
box = s.shapes.add_textbox(Inches(0.8), Inches(4.6), Inches(11.7), Inches(1.2))
tf = _tf(box)
r = tf.paragraphs[0].add_run()
r.text = "Docs: temporal-alignment.md · 50salads-notes.md · delta-code.md · approach.md"
r.font.size = Pt(14); r.font.name = FONT; r.font.color.rgb = RGBColor(0xC9, 0xD6, 0xE3)

out = "/Users/elizpayasli/Documents/GitHub/DELTA/slides/DELTA_progress_Mariella.pptx"
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
prs.save(out)
print("saved", out, "-", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
