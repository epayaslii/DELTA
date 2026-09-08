"""Generate the progress deck. python-pptx, 16:9."""
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
    (0, "Goal: transcript-only Dense Long-Term Action Anticipation; focus = the Temporal Alignment "
        "(TA) component, on 50Salads."),
    (0, "Direction (your guidance): the TA must use a VLM, and weak alignment → refine, not segmentation."),
    (0, "Built and tested: the full two-stage pipeline — semantic similarity → ordered coarse alignment "
        "→ local boundary search → contrastive refinement.  67 unit tests."),
    (0, "Measured: every configuration so far lands BELOW a baseline that ignores the video entirely "
        "(naive-uniform, MoC 0.366).  Best is 0.352."),
    (1, "That holds for the VLM path AND for a segmentation (ATBA/HAL-style) probe — so it is not "
        "a method-choice problem."),
    (0, "Diagnosis: single-FRAME evidence is too weak on a fixed overhead camera. "
        "The VLM direction has not yet been fairly tested — VideoLLaMA3 encodes video, SigLIP2 does not."),
    (0, "Blocker: GPU / cluster access.  Raw 50Salads video is now in hand."),
], sub="Internship progress review  ·  results + what I need")

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
    (0, "ATBA / HAL — \"alignment THROUGH segmentation\"  (ruled out by your no-segmentation guidance):"),
    (1, "frozen I3D  →  trained frame classifier  →  posteriors P  →  boundary/transition scores  →  DP  →  Y*"),
    (1, "the alignment can be no better than that classifier — and on 50Salads (fixed camera, "
        "near-duplicate cut_* / add_* actions) it is a near-worst case."),
    (0, "Our direction — \"alignment THROUGH semantic matching\":"),
    (1, "frozen VLM:  s(n, t) = sim( text(action_n), video_t )  →  order-preserving alignment  →  Y*"),
    (1, "no frame classifier; the fine-grained vocabulary is disambiguated by the noun, from step 0."),
    (0, "Two CVPR'25/26 papers do exactly this for video temporal grounding, and we adapt them:"),
    (1, "CVA (CVPR'26) — the aligner: hierarchical encoder + boundary-contrastive loss."),
    (1, "MASRA (2026) — the language regulariser: align a text relation-matrix to the video similarity "
        "matrix; MLLM used at training only, discarded at inference.  (Eliz)"),
], sub="the TA must use a VLM  ·  CVA + MASRA as the references")

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

# ------------------------------------------------------------------ 9. LITERATURE
table_slide("Progress 5 — VLM-alignment literature (analysed)",
    ["Paper", "Venue", "What it is", "Role for us"],
    [
        ["ATBA", "CVPR'24", "the classifier→DP alignment DELTA's TA follows", "the baseline we replace"],
        ["HAL", "CVPR'26", "= ATBA + a VAE regulariser; +2–3 MoF; segmentation-based; skips 50Salads", "analysed, then DROPPED (no-segmentation)"],
        ["CVA", "CVPR'26", "VLM video-text alignment for grounding; hierarchical encoder + boundary-contrastive loss; SOTA", "the VLM aligner"],
        ["MASRA", "2026", "MLLM-assisted alignment: align a text relation-matrix to the video similarity matrix; MLLM train-only", "Eliz — the language regulariser"],
        ["TAN / StepFormer", "CVPR'22/'23", "the genuine transcript/sequence→video aligners (weak/self-sup)", "the alignment mechanism references"],
    ],
    sub="no single paper = {weak supervision} × {VLM alignment} × {long-term anticipation}  →  that is the contribution",
    col_widths=[1.9, 1.0, 6.3, 2.9], font=9.5)

# ------------------------------------------------------------------ 10. MASRA + CVA
bullets("Progress 6 — the two VLM-alignment references", [
    (0, "CVA (CVPR'26) — Context-aware Video-text Alignment.  Task: video temporal grounding "
        "(query → span), CLIP+SlowFast, SOTA.  Contribution = the aligner:"),
    (1, "CTE hierarchical encoder (windowed self-attn + learnable queries + bidirectional cross-attn)"),
    (1, "CBD boundary-contrastive loss + QCD query-aware augmentation."),
    (0, "MASRA (2026) — MLLM-Assisted Semantic-Relational Consistent Alignment.  Same task, "
        "CLIP + an MLLM.  Contribution = a training-time language regulariser:"),
    (1, "LRCA — align a text relation-matrix (from MLLM captions) with the video's similarity matrix"),
    (1, "ESTA — align pooled temporal context with action/event semantics"),
    (1, "MLLM used ONLY at training, discarded at inference  (= DELTA's philosophy)."),
    (0, "Both: grounding, supervised, not procedural — we adapt them to transcript-supervised 50Salads."),
], sub="CVA = the aligner   ·   MASRA = the language regulariser")

# ------------------------------------------------------------------ 11. THE SPLIT
bullets("Scoping — the workstream split", [
    (0, "Supervisor's direction: the TA must use a VLM; no segmentation-based approach."),
    (0, ""),
    (0, "CVA track:  build the VLM aligner — CTE encoder + CBD boundary-contrastive loss."),
    (0, "Eliz → MASRA:  the training-time language regulariser — LRCA / ESTA — that shapes the "
        "video↔transcript similarity, then an order-preserving alignment reads Y* off it."),
    (0, ""),
    (0, "Structurally: CVA produces the alignment; MASRA (like HAL did on the segmentation side) "
        "is an auxiliary training signal that improves it — measured on TA metrics, then downstream MoC."),
    (0, "VLM for both: InternVideo2 (video-native) instead of CVA/MASRA's SlowFast + frame-CLIP."),
], sub="MASRA track   ·   CVA track   ·   both VLM")

# ------------------------------------------------------------------ 12. TAKEAWAYS
bullets("Key takeaways", [
    (0, "The pipeline is implemented end to end and unit-tested; the approach is unproven."),
    (0, "Both paradigms lose to the no-evidence floor on 50Salads — VLM-direct AND classifier-based. "
        "The bottleneck is the evidence, not the algorithm."),
    (0, "DELTA already has optimal-transport alignment (ASOT, --model_type wclot), so our Y* drops into "
        "the existing decoder unchanged — CTC / CRF / duration head / eval all reused."),
    (0, "Ideas are drawn from OVTAS (similarity), HiERO-StepG (monotonic decode), CVA (boundary "
        "contrastive), MASRA (relational losses); the local boundary search is ours."),
    (0, "Next real test: video-level features (VideoLLaMA3) rather than frame-level — the one thing "
        "that directly addresses the measured failure."),
], sub="what the work so far establishes")

# ------------------------------------------------------------------ 13. BLOCKERS
table_slide("Blockers & risks",
    ["Item", "Impact", "Mitigation"],
    [
        ["Raw 50Salads video unavailable", "blocks VLM feature extraction (the core of both tracks)", "chase a lab copy; meanwhile reproduce MASRA/CVA on TACoS (cooking VTG benchmark)"],
        ["No local GPU / torch", "no model runs on the Mac", "UPC cluster; one conda env"],
        ["MoF ≠ MoC", "a method can win on segmentation MoF and not help DELTA's anticipation MoC", "always measure pseudo-label MoC + boundary offset, then downstream MoC"],
        ["DELTA code is research-grade", "reproduction friction (missing script, paths, wandb)", "documented in docs/delta-code.md; map run scripts onto train.py"],
        ["MASRA/CVA are grounding + supervised", "not a drop-in — architecture + losses must be adapted", "keep only the transferable pieces (LRCA/ESTA; CTE/CBD); wire into DELTA's ASOT + decoder"],
    ],
    sub="known before we commit compute",
    col_widths=[3.0, 4.3, 4.8], font=10.5)

# ------------------------------------------------------------------ 14. SECTION
section("Results", kicker="Everything measured so far")

# ------------------------------------------------------------------ 15. HEADLINE RESULT
table_slide("The headline — nothing beats a baseline that ignores the video",
    ["Method", "Uses video?", "MoC", "F1@50"],
    [
        ["naive-uniform  (split into N equal blocks, transcript order)", "no", "0.366", "26.2"],
        ["VLM: ASOT + local boundary refinement", "yes", "0.352", "23.4"],
        ["VLM: ASOT (OT + temporal prior)", "yes", "0.342", "22.9"],
        ["Segmentation: ATBA-style classifier + DP loop", "yes", "0.202", "14.0"],
        ["VLM: hard DP on similarity", "yes", "0.199", "6.7"],
        ["Segmentation: HAL-style (+ recon regulariser)", "yes", "0.149", "9.4"],
    ],
    sub="50Salads split-1 test.  The floor uses NO visual information at all — anything below it means the "
        "visual evidence is hurting, not helping.",
    col_widths=[6.6, 1.6, 1.6, 1.6], font=10.5)

# ------------------------------------------------------------------ 16. WHY
bullets("Why — the diagnostic evidence", [
    (0, "Fixed overhead camera:  consecutive frames are 0.94 cosine-similar.  Almost nothing changes visually."),
    (0, "Single-frame semantics are weak:  with the correct action vs the OTHER actions in the same "
        "transcript, SigLIP2 picks the right one only 27% of the time."),
    (1, "Beating a *random* action is easy (cut_tomato vs add_dressing → 73%). The hard part is "
        "cut_tomato vs cut_cucumber vs cut_cheese — which is exactly what the aligner must do."),
    (0, "Boundary signal is near chance:  frame-similarity at true boundaries scores 0.52 (0.50 = chance); "
        "I3D managed 0.67."),
    (0, "The segmentation loop self-confirms:  loss 2.47 → 0.45 while relabelling falls 73% → 8% — "
        "the classifier confidently fits its own initial mistakes."),
    (0, "Adaptive sampling skips 85% of frames, but only 32% of true boundaries fall inside a refined zone "
        "— driven by the same weak similarity."),
], sub="the difficulty is the evidence available, not the choice of algorithm")

# ------------------------------------------------------------------ 17. WHAT THIS MEANS
bullets("What I take from this", [
    (0, "The question is not \"segmentation vs VLM\" — both lose to a baseline that never looks at the video."),
    (0, "The real question:  what evidence could localise a boundary on this dataset at all?"),
    (0, "Argument for temporal context:  every result so far uses FRAME-level evidence. "
        "A fixed camera is precisely the case where single frames should fail and video should help."),
    (0, "→ We have not yet fairly tested the VLM direction.  SigLIP2 encodes single frames; "
        "VideoLLaMA3 encodes video.  That test has not been run."),
    (0, "Caveat, stated plainly:  the segmentation probe is a compact reimplementation "
        "(no boundary detector, minutes of training).  It is evidence about the paradigm, "
        "NOT a reproduction of HAL — which never ran 50Salads."),
], sub="the case for continuing, with clear eyes about the difficulty")

# ------------------------------------------------------------------ 18. BUILT
table_slide("What is built and tested",
    ["Stage", "Module", "Status"],
    [
        ["0  semantic-guided sampling", "delta.features.sampling  (+ extract --adaptive)", "built · 7 tests"],
        ["1  clip ↔ action similarity", "delta.align.similarity / cost", "built"],
        ["2  coarse ordered decode", "delta.align.asot  (OT + transcript-order prior)", "built · 8 tests"],
        ["3  local boundary search", "delta.align.refine  — the novel piece", "built · 6 tests"],
        ["4  contrastive refinement", "delta.align.cbd (PBCR) + masra_torch (LRCA/ESTA)", "built · 15 tests"],
        ["—  backbone gate", "scripts/verify_backbone_alignment.py", "built · catches a weak backbone before GPU spend"],
        ["—  cluster jobs", "slurm_extract_videollama3.sh · slurm_delta_baselines.sh", "ready to submit"],
    ],
    sub="67 unit tests pass.  These verify the code is correct on synthetic data — they do NOT show the method works.",
    col_widths=[2.9, 5.6, 3.0], font=9.5)

# ------------------------------------------------------------------ 19. DECISION + ASKS
bullets("What I need from you", [
    (0, "1.  GPU / cluster account.  Everything below is blocked on it."),
    (1, "VideoLLaMA3 feature extraction · VideoLLaMA3 captions · the real ATBA / wclot baselines."),
    (1, "Remote access (VPN/SSH) confirmed WHILE I am still on-site — otherwise the four remote weeks are lost."),
    (0, "2.  A decision:  run VideoLLaMA3 as the fair test of the VLM direction, with real ATBA/HAL "
        "in parallel as the baseline row we need anyway — rather than pivoting on the evidence above?"),
    (0, "3.  Segments into the VLM — a visual marker on the frames, or a text prompt with the "
        "candidate timestamps and the two action names?"),
    (0, "4.  Boundary stage — VideoLLaMA3 embeddings + a contrastive objective, or VideoLLaMA3-Chat "
        "judging the transition directly?"),
    (0, "5.  Which recent paper did you mean by \"passes things to a VLM and reasons in real time\"? "
        "(TOGA fits — weakly-supervised grounding, no timestamps.)"),
], sub="access first, then the direction call")

# ------------------------------------------------------------------ CLOSE
s = section("Thank you", kicker="github.com/epayaslii/DELTA")
box = s.shapes.add_textbox(Inches(0.8), Inches(4.6), Inches(11.7), Inches(1.2))
tf = _tf(box)
r = tf.paragraphs[0].add_run()
r.text = ("Docs: temporal-alignment.md · 50salads-notes.md · delta-code.md · "
          "baselines-hal-cva.md · approach.md")
r.font.size = Pt(13); r.font.name = FONT; r.font.color.rgb = RGBColor(0xC9, 0xD6, 0xE3)

out = "/Users/elizpayasli/Documents/GitHub/DELTA/slides/DELTA_progress.pptx"
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
prs.save(out)
print("saved", out, "-", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
