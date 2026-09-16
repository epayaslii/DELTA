# What we actually borrow from each paper — explained

Companion to the "Summary" sheet in `articles-by-pipeline-part.xlsx`. That
table's "What we actually borrow" column is one phrase per row; this expands
each phrase into what the mechanism actually does, in plain terms, so it's
readable without opening the source paper. Same row order as the table.

---

### Input — DELTA

**Borrowed: the weak-supervision setup itself.** DELTA's whole premise is that
you never get frame-by-frame timestamps for training or testing — only the
*ordered list* of actions in the video (the "transcript"), e.g. "cut tomato,
place tomato in bowl, cut cheese, ..." with no indication of when each one
starts or ends. Everything downstream has to work under that constraint. This
isn't a technique we adapted, it's the rules of the problem we inherited along
with the task.

### Semantic front-end — OVTAS

**Borrowed: score a frame against an action using a frozen vision-language
model, with no training.** OVTAS's idea (called FAES in the paper) is: take a
video-language model that was never trained on your dataset, encode each video
frame and each candidate action label ("cutting a tomato") into the same
numerical space, and just measure how similar they are (cosine similarity).
High similarity = the frame probably shows that action. No classifier is
trained specifically for this — the same frozen model works out of the box.
We use exactly this scoring mechanism, but where OVTAS scores against an
open, unordered list of possible actions, we score against the actions *in
the order the transcript gives them* — we know action 3 must come after
action 2, and we use that.

### Stage A: coarse alignment — HiERO-StepG (+ our own dynamic programming)

**Borrowed: force the predicted timeline to respect the transcript's order,
with no exceptions.** HiERO-StepG's core trick for localizing a procedure's
steps in a video, with no per-step annotation, is to require that if step 3
comes after step 2 in the recipe, then step 3's video segment must come after
step 2's video segment in time — always, never a swap. This is done
algorithmically (dynamic programming / optimal transport), not learned. It
turns a hard problem (find 20 boundaries in an 11,000-frame video with a weak
similarity signal) into a much easier one, because most of the possible
placements are simply ruled out by the ordering constraint before you even
look at how good the visual match is.

### Stage B1: segment extent — HiERO-StepG

**Borrowed: once you have a rough position, look at a local window around it
to refine it.** HiERO-StepG doesn't just place a single cut point between
steps; it expands each step into its full plausible time range by checking
frames just before and after the rough boundary and asking whether they still
belong to that step. We reuse this "look locally, expand or contract from the
coarse guess" idea rather than trusting the coarse alignment's placement as
final.

### Stage B2: exact transition — our own design, inspired by CVA + MASRA

**Borrowed (as principles, not as ready-to-use code): two ideas about
boundaries specifically.**
- From **CVA**: a boundary frame is special — it should look the same no
  matter what's happening around it (invariant to context), and it should be
  clearly distinguishable from the frames just before and after it. CVA turns
  this into a training signal (a contrastive loss) but does it using
  ground-truth boundary positions, which we don't have.
- From **MASRA**: instead of judging each frame in isolation, look at how
  frames relate to *each other* (a relation/similarity matrix across the
  whole clip) and push that structure to match what the text/transcript says
  it should look like.

Because we don't have ground-truth boundaries to anchor either idea the way
the original papers do, we rebuild both on top of our *own* current best
guess at where the boundary is (from Stage A/B1), only trusting that guess
when it's confident. This is why we call it our own formulation rather than a
reuse of either paper's loss function.

### Optional Stage C — CLOT / D-CLOT

**Borrowed: alternate between refining the frame-level detail and refining the
bigger segment-level picture, repeatedly.** CLOT's idea is that a single pass
of "look at frames → decide segments" isn't enough — segment-level
information (this whole stretch is "cutting cheese") can help clean up
individual frame decisions, and vice versa, so it loops between the two
several times. D-CLOT adds: periodically re-check whether the "typical
example" of each action class (the prototype) is still accurate, especially
for short or easily-confused actions, and update it if not. We keep this as a
fallback — only switched on if the earlier stages leave boundaries too noisy.

### Optional Stage D — TOGA / VideoLLaMA3

**Borrowed: ask a language-capable video model directly, but only when
cheaper methods are unsure.** TOGA shows you can get a video-language model to
answer "when does this happen?" without ever training it on ground-truth
timestamps, by checking that its answer is self-consistent (if it says an
event happens at time X, and you then ask "what happens at time X?", it should
describe the same event back). This is expensive to run, so we only call it
on the small number of boundaries our confidence score flags as genuinely
uncertain — not on every boundary in every video.

### Output — DELTA

**Borrowed: the format everything must be handed back in.** Whatever our
pipeline decides about where each action starts and ends, it has to come out
as one label per video frame (a dense, frame-by-frame sequence), because
that's what DELTA's existing downstream model (the part that predicts future
actions) expects as input. We don't change that interface — only what
produces the labels feeding into it.
