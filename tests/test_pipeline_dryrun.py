"""CPU dry-run of the patched WLTA training code + the hybrid TA pipeline, on mock data.

Run:   .venv/bin/python -m pytest tests/test_pipeline_dryrun.py -v -s
   or: .venv/bin/python tests/test_pipeline_dryrun.py

What is real: the group's `VideoSSL` module (Conv1d in_proj, 8-layer TAS encoder,
DistilBERT cross-modal grounding with real weights, 4-layer LTA decoder, CRF, ATBA
loss) run through a real `pl.Trainer` on CPU. What is mock: the features
([B, T, 2048] random), the transcripts/labels (random piecewise-constant), the
similarity files. So this proves shapes, wiring, the optimizer reset and the
masking -- not accuracy.

Needs the patched group code under third_party/delta_wlta/src (see
patches/wlta_repro/apply.sh) plus torch, pytorch-lightning, transformers (downloads
distilbert-base-uncased once), scipy, scikit-learn, wandb, rotary-embedding-torch,
einops. Skipped automatically if any of that is missing.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "third_party" / "delta_wlta" / "src"
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")
pl = pytest.importorskip("pytorch_lightning")
for _m in ("scipy", "sklearn", "wandb", "einops", "transformers", "rotary_embedding_torch"):
    pytest.importorskip(_m)
if not (SRC / "wlta_fixes.py").exists():
    pytest.skip("patched group code not found (run patches/wlta_repro/apply.sh)", allow_module_level=True)
sys.path.insert(0, str(SRC))

from hybrid_ta import HybridConfig, generate_batch, generate_pseudo_labels  # noqa: E402

B, T, D_IN = 2, 64, 2048
N_CLS = 19                      # 17 actions + action_start(17) + action_end(18)
START, END = 17, 18

MAPPING_50S = """0 cut_tomato
1 place_tomato_into_bowl
2 cut_cheese
3 place_cheese_into_bowl
4 cut_lettuce
5 place_lettuce_into_bowl
6 add_salt
7 add_vinegar
8 add_oil
9 add_pepper
10 mix_dressing
11 peel_cucumber
12 cut_cucumber
13 place_cucumber_into_bowl
14 add_dressing
15 mix_ingredients
16 serve_salad_onto_plate
17 action_start
18 action_end
"""
BREAKFAST_ONLY = {"SIL", "pour_milk", "stir_coffee", "crack_egg", "fry_pancake", "add_teabag"}


# ----------------------------------------------------------------------------
# harness: real argparse defaults + real VideoSSL + real Trainer, mock data
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def twt():
    return importlib.import_module("train_window_tokenizer")


def _parse_args(twt, argv):
    """Rebuild the script's argparse from its own source (it lives under __main__)."""
    src = Path(twt.__file__).read_text().splitlines()
    a = next(i for i, l in enumerate(src) if l.startswith("    parser = argparse.ArgumentParser"))
    z = next(i for i, l in enumerate(src) if l.startswith("    args = parser.parse_args()"))
    ns = {"argparse": argparse}
    exec(textwrap.dedent("\n".join(src[a:z])), ns)
    return ns["parser"].parse_args(argv)


# the Table-2 / launcher flag set (50Salads, atba), CPU
BASE_ARGV = ["-d", "FS", "-ac", "all", "-c", str(N_CLS), "-mpos", "19000", "-f", str(T), "--model_type", "atba",
             "--atba_enc_layers", "8", "--atba_encIn_dim", "512", "--LTA_dec_hidden_dim", "256",
             "--LTA_dec_n_head", "4", "--LTA_dec_n_query", "20", "--LTA_dec_layers", "3",  # code/config, not Supp. Table 2 (see README)
             "--dropout", "0.5", "-bs", str(B), "--ABLAT_tsm", "--use_text", "--text_encoder", "distilbert",
             "--nofprojections", "3"]


def _build(twt, tmp_path, extra=()):
    args = _parse_args(twt, BASE_ARGV + list(extra))
    mapping = tmp_path / "mapping.txt"
    mapping.write_text(MAPPING_50S)
    raw = twt.load_id2name_from_mapping(str(mapping))
    args.id2name = twt.humanize_id2name(raw)
    args.boundary_ids = twt.boundary_class_ids(raw)
    twt.args = args                       # the module reads a global `args`
    pl.seed_everything(0)
    ssl = twt.VideoSSL(args, layer_sizes=args.layers, n_clusters=args.n_clusters, alpha_train=args.alpha_train,
                       alpha_eval=args.alpha_eval, ub_frames=args.ub_frames, ub_actions=args.ub_actions,
                       lambda_frames_train=args.lambda_frames_train, lambda_frames_eval=args.lambda_frames_eval,
                       lambda_actions_train=args.lambda_actions_train, lambda_actions_eval=args.lambda_actions_eval,
                       step_size=args.step_size, train_eps=args.eps_train, eval_eps=args.eps_eval,
                       radius_gw=args.radius_gw, n_ot_train=args.n_ot_train, n_ot_eval=args.n_ot_eval,
                       n_frames=args.n_frames, lr=args.learning_rate, weight_decay=args.weight_decay, rho=args.rho,
                       exclude_cls=args.exclude, visualize=False, name="dryrun")
    return args, ssl


def _mock_gt(rng, t=T):
    """[17, a1..a8, 18] with random segment lengths -> (t,) labels."""
    trans = [START] + [int(x) for x in rng.choice(17, 8, replace=False)] + [END]
    cuts = np.sort(rng.choice(np.arange(1, t), len(trans) - 1, replace=False))
    lens = np.diff(np.concatenate([[0], cuts, [t]]))
    return np.repeat(trans, lens)


def _mock_dataset(n=4, names=None):
    rng = np.random.default_rng(0)
    items = []
    for i in range(n):
        gt = torch.from_numpy(_mock_gt(rng)).long()
        feats = torch.randn(T, D_IN)
        mask = torch.ones(T, dtype=torch.bool)
        fname = names[i] if names else f"vid{i}.txt"
        items.append((feats, mask, gt, fname, int(gt.unique().numel()), 0.3, 0.5))
    return items


def _loader(items):
    return torch.utils.data.DataLoader(items, batch_size=B, shuffle=False)


def _trainer(epochs, callbacks=()):
    return pl.Trainer(accelerator="cpu", devices=1, max_epochs=epochs, logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                      callbacks=list(callbacks), log_every_n_steps=1)


# ----------------------------------------------------------------------------
# 1a. text grounding uses 50Salads names, 1-to-1 with action ids
# ----------------------------------------------------------------------------
def test_text_grounding_uses_50salads_names(twt, tmp_path):
    args, ssl = _build(twt, tmp_path)
    names = ssl.txt_guidance.id2name
    assert len(names) == N_CLS and names[0] == "cut tomato" and names[16] == "serve salad onto plate"
    assert not (set(names.values()) & BREAKFAST_ONLY)

    seen = []
    tok = ssl.txt_guidance.txt_tokenizer

    class Recorder:                       # capture what DistilBERT is really fed
        def __call__(self, text, *a, **k):
            seen.extend(text)
            return tok(text, *a, **k)

        def __getattr__(self, k):
            return getattr(tok, k)

    ssl.txt_guidance.txt_tokenizer = Recorder()
    transcripts = [[START, 0, 8, END], [START, 16, END]]
    t_txt, t_mask = ssl.txt_guidance._text_tokens_per_action(transcripts, 512, torch.device("cpu"))
    assert t_txt.shape == (2, 4, 512) and t_mask.sum().item() == 7
    assert seen == ["action start", "cut tomato", "add oil", "action end",
                    "action start", "serve salad onto plate", "action end"]


# ----------------------------------------------------------------------------
# 1b. architecture: 2048 -> 512 encoder -> 256 projection -> LTA_dec_layers-deep decoder
# ----------------------------------------------------------------------------
def test_architecture_dims(twt, tmp_path):
    args, ssl = _build(twt, tmp_path)
    assert ssl.in_proj.in_channels == 2048 and ssl.in_proj.out_channels == 512
    assert len(ssl.TAS_encoder.layers) == 8
    lta = ssl.lta_model
    assert isinstance(lta.clot2LTA, torch.nn.Linear) and (lta.clot2LTA.in_features, lta.clot2LTA.out_features) == (512, 256)
    assert len(lta.LTA_decoder.layers) == 3  # matches --LTA_dec_layers passed in BASE_ARGV
    # defaults of the patched script themselves (no flags): decoder depth 3 (code/config, not Table 2), encoder input 512
    d = _parse_args(twt, ["-d", "FS", "-ac", "all", "-c", "19"])
    assert d.LTA_dec_layers == 3 and d.atba_encIn_dim == 512

    # shape flow: [B,T,2048] -> in_proj -> [B,T,512] -> clot2LTA -> [B,T,256] -> decoder memory
    x = torch.randn(B, T, D_IN)
    h = ssl.in_proj(x.transpose(1, 2)).transpose(1, 2)
    assert h.shape == (B, T, 512)
    assert lta.clot2LTA(h).shape == (B, T, 256)


# ----------------------------------------------------------------------------
# 1c/1d. real Trainer, 32 epochs: forward/backward through everything, and the
#        epoch-30 optimizer reset (momentum buffers cleared, Adam step counter restarts)
# ----------------------------------------------------------------------------
class _Probe(pl.Callback):
    def __init__(self):
        self.rows = {}

    def on_train_epoch_end(self, trainer, module):
        opt = trainer.optimizers[0]
        steps = [int(s["step"]) for s in opt.state.values() if "step" in s]
        exp_avg = sum(float(s["exp_avg"].abs().sum()) for s in opt.state.values() if "exp_avg" in s)
        self.rows[trainer.current_epoch] = dict(n_state=len(opt.state), max_step=max(steps, default=0),
                                                exp_avg=exp_avg, lr=opt.param_groups[0]["lr"],
                                                lta=module.lta_step)


def test_epoch30_optimizer_reset_and_stage_switch(twt, tmp_path, capsys):
    args, ssl = _build(twt, tmp_path, ["-lr", "5e-4", "-wd", "3e-4"])
    probe = _Probe()
    _trainer(32, [probe]).fit(ssl, _loader(_mock_dataset(2)))      # 1 batch / epoch
    out = capsys.readouterr().out
    r = probe.rows
    assert "[stage2] epoch 30: optimizer state reset" in out
    assert not r[28]["lta"] and r[30]["lta"]                          # L_DLTA switches on at 30
    assert r[29]["max_step"] == 30 and r[29]["exp_avg"] > 0           # 30 accumulated steps before
    assert r[30]["max_step"] == 1                                     # counter restarted at the switch
    assert r[30]["lr"] == pytest.approx(5e-4)
    assert all(np.isfinite(v["exp_avg"]) for v in r.values())          # no NaN blow-up after the switch


def test_reset_can_be_disabled(twt, tmp_path):
    args, ssl = _build(twt, tmp_path, ["--no_opt_reset"])
    probe = _Probe()
    _trainer(32, [probe]).fit(ssl, _loader(_mock_dataset(2)))
    assert probe.rows[30]["max_step"] == 31


# ----------------------------------------------------------------------------
# 1e. MoC masking on mock predictions
# ----------------------------------------------------------------------------
def test_moc_masks_boundary_tokens_over_17_classes(twt, tmp_path):
    from metrics import eval_file, calculate_moc
    args, ssl = _build(twt, tmp_path)
    assert list(ssl.moc_core_ids) == list(range(17)) and ssl.moc_boundary_ids == (START, END)

    gt = np.array([0] * 20 + [8] * 20 + [16] * 10 + [END] * 10)        # future window reaches action_end
    pred = np.array([0] * 20 + [8] * 10 + [3] * 10 + [16] * 10 + [0] * 10)   # action_end always wrong
    T_a, F_a = eval_file(gt, pred, 0.3, classes={i: i for i in range(N_CLS)})
    all_cls = calculate_moc(T_a, F_a)                                   # (1 + .5 + 1 + 0) / 4
    core = twt.masked_moc(T_a, F_a, ssl.moc_core_ids)                   # (1 + .5 + 1) / 3
    assert all_cls == pytest.approx(2.5 / 4) and core == pytest.approx(2.5 / 3)
    # the mask really ignores whatever happens on the boundary ids
    T2, F2 = T_a.copy(), F_a.copy(); F2[START] += 999; T2[END] += 999
    assert twt.masked_moc(T2, F2, ssl.moc_core_ids) == pytest.approx(core)
    # the escape hatch reproduces the old number
    args2, ssl2 = _build(twt, tmp_path, ["--moc_all_classes"])
    assert twt.masked_moc(T_a, F_a, ssl2.moc_core_ids) == pytest.approx(all_cls)


# ----------------------------------------------------------------------------
# 2. hybrid_ta: shapes, order, recovery of a planted alignment
# ----------------------------------------------------------------------------
def _planted(rng, t, n, noise=0.3):
    truth = np.sort(rng.integers(0, n, t)); truth[:n] = np.arange(n); truth = np.sort(truth)
    sim = rng.normal(0, noise, (t, n)); sim[np.arange(t), truth] += 1.0
    return sim, truth


@pytest.mark.parametrize("method", ["dp", "asot"])
@pytest.mark.parametrize("branches", ["none", "a", "b", "ab"])
def test_hybrid_ta_pipeline(method, branches):
    rng = np.random.default_rng(1)
    sim, truth = _planted(rng, 240, 8)
    r = generate_pseudo_labels(sim, list(range(8)), cfg=HybridConfig(method=method, branches=branches))
    assert r.y_star.shape == (240,)
    assert (np.diff(r.entry_of_frame) >= 0).all() and set(r.entry_of_frame) == set(range(8))  # ordered, all covered
    assert (r.y_star == truth).mean() > 0.9


def test_branches_pull_a_shifted_coarse_boundary_back():
    """Each branch, given coarse boundaries off by +8 frames, lands closer to the truth."""
    from hybrid_ta.boundary_refinement import branch_a_semantic, branch_b_relational
    n, t = 6, 300
    truth = np.repeat(np.arange(n), t // n)
    true_b = np.flatnonzero(np.diff(truth)) + 1
    errs = {"coarse": [], "A": [], "B": []}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        sim = rng.normal(0, .3, (t, n)); sim[np.arange(t), truth] += 1.0
        shifted = np.repeat(np.arange(n), [50 + 8] + [50] * (n - 2) + [50 - 8])   # every boundary +8
        a = branch_a_semantic(sim.T, shifted, radius=20, window=15, frame_emb=sim)
        b = branch_b_relational(sim, shifted, radius=20, window=15)
        errs["coarse"] += [8] * (n - 1)
        errs["A"] += list(np.abs(np.array(a.boundaries) - true_b))
        errs["B"] += list(np.abs(np.array(b.boundaries) - true_b))
    med = {k: float(np.median(v)) for k, v in errs.items()}
    print("median boundary error (frames):", med)
    assert med["A"] < med["coarse"] and med["B"] < med["coarse"]


def test_hybrid_batch_shape_and_padding():
    rng = np.random.default_rng(2)
    sim = torch.from_numpy(rng.normal(0, .3, (B, T, N_CLS)).astype(np.float32))
    mask = torch.ones(B, T, dtype=torch.bool); mask[1, 50:] = False        # second video shorter
    transcripts = [[START, 0, 8, 0, END], [START, 16, END]]                # repeated action allowed
    y = generate_batch(sim, transcripts, mask)
    assert y.shape == (B, T) and y.dtype == torch.long
    assert (y[1, 50:] == -100).all() and (y[0] != -100).all()
    for i in range(B):
        v = y[i][mask[i]].tolist()
        assert v[0] == START and v[-1] == END and set(v) <= set(transcripts[i])


# ----------------------------------------------------------------------------
# 2e. --ta_source hybrid feeds Y* into the SAME losses; delta path is untouched
# ----------------------------------------------------------------------------
def test_ta_source_hybrid_end_to_end(twt, tmp_path):
    names = [f"vid{i}.txt" for i in range(2)]
    sim_dir = tmp_path / "sim"; sim_dir.mkdir()
    rng = np.random.default_rng(3)
    for n in names:                                                    # full-res class similarity, 300 frames
        np.savez(sim_dir / (Path(n).stem + ".npz"), sim=rng.normal(0, .3, (300, N_CLS)).astype(np.float32))
    items = _mock_dataset(2, names)

    args, ssl = _build(twt, tmp_path, ["--ta_source", "hybrid", "--hybrid_sim_dir", str(sim_dir)])
    calls = []
    orig = ssl.hybrid_ta.__call__

    class Spy:
        def __call__(self, *a, **k):
            y = ssl.hybrid_ta_impl(*a, **k); calls.append(y.shape); return y
    ssl.hybrid_ta_impl = ssl.hybrid_ta
    ssl.hybrid_ta = Spy()
    probe = _Probe()
    _trainer(12, [probe]).fit(ssl, _loader(items))                     # past the 10-epoch warm-up: fr/glc losses run
    assert len(calls) == 12 and all(s == (B, T) for s in calls)
    assert all(np.isfinite(v["exp_avg"]) for v in probe.rows.values())

    # default keeps the original ATBA path
    args_d, ssl_d = _build(twt, tmp_path)
    assert ssl_d.hybrid_ta is None
    _trainer(12).fit(ssl_d, _loader(items))


# ----------------------------------------------------------------------------
# 4. Slurm scripts: the commands they would run parse with the script's own argparse
# ----------------------------------------------------------------------------
def _slurm_cmds(script, env):
    import os, shlex, subprocess
    e = {**os.environ, "DRY_RUN": "1", "SLURM_SUBMIT_DIR": str(ROOT), **env}
    out = subprocess.run(["bash", str(ROOT / "scripts" / "slurm" / script)], capture_output=True, text=True, env=e, check=True).stdout
    return [shlex.split(l[5:]) for l in out.splitlines() if l.startswith("CMD: ")]


def test_slurm_scripts_build_valid_commands(twt, tmp_path):
    base = _slurm_cmds("submit_50salads_baseline.sh", {})
    assert len(base) == 5                                          # 5 cross-validation splits
    hyb = _slurm_cmds("submit_50salads_hybrid.sh", {"SIM_DIR": str(tmp_path), "HYBRID_BRANCHES": "a"})
    assert len(hyb) == 5
    for cmds, source in ((base, "delta"), (hyb, "hybrid")):
        assert [_parse_args(twt, c[2:]).split for c in cmds] == [1, 2, 3, 4, 5]
        for c in cmds:
            a = _parse_args(twt, c[2:])                            # drops "python3 src/train_window_tokenizer.py"
            assert a.ta_source == source and a.n_clusters == 19 and a.LTA_dec_layers == 3
            assert a.atba_encIn_dim == 512 and a.LTA_dec_hidden_dim == 256 and a.atba_enc_layers == 8
            assert (a.gamma1, a.gamma2, a.gamma3) == (0.6, 0.01, 1.0) and a.n_epochs == 80
            assert a.text_encoder == "distilbert" and a.use_text and not a.no_opt_reset
    a = _parse_args(twt, hyb[0][2:])
    assert a.hybrid_sim_dir == str(tmp_path) and a.hybrid_branches == "a" and a.hybrid_method == "asot"
    d = _slurm_cmds("submit_50salads_hybrid.sh", {"SIM_DIR": str(tmp_path)})[0]                    # defaults: ASOT + branch A
    assert _parse_args(twt, d[2:]).hybrid_branches == "a" and HybridConfig().branches == "a" and HybridConfig().method == "asot"
    # array mode: one split per task
    assert len(_slurm_cmds("submit_50salads_baseline.sh", {"SLURM_ARRAY_TASK_ID": "3"})) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
