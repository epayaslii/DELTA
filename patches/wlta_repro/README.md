# WLTA 50Salads reproduction patch

Four changes to `train_window_tokenizer.py` (the group's code, gitignored here, so
kept as a patch). `apply.sh <delta_wlta dir>` copies `wlta_fixes.py` in and applies
`train_window_tokenizer.patch`; tests: `tests/test_wlta_fixes.py`.

| # | change | flag / default |
|---|---|---|
| 1 | class names read from the same `mapping.txt` the loader uses (was a Breakfast file); `_`→space for DistilBERT | always on |
| 2 | MoC over core classes only; boundary tokens found **by name** | on; `--moc_all_classes` to disable |
| 3 | anticipation decoder depth 4, encoder input 512 | `--LTA_dec_layers 4`, `--atba_encIn_dim 512` (new defaults) |
| 4 | γ exposed; AdamW state + hyper-parameters reset at epoch 30 | `--gamma1 0.6 --gamma2 0.01 --gamma3 1.0 --gamma1_lta 0.8`; `--no_opt_reset` to disable |

Notes
- `--num_decoder_layers` is a no-op for `--model_type atba`; the anticipation decoder depth is `--LTA_dec_layers`.
- `--gamma1_lta 0.8` keeps the original code's value; set 0.6 if γ1 should stay 0.6 after stage 2.
- There is no LR scheduler in the code, so "reset the scheduler" has nothing to act on; a warning is printed if one is ever attached.
- Reference command (supplementary Table 2 values; the group's launcher uses `-lr 5e-3`):
  `python3 src/train_window_tokenizer.py -d FS -c 19 -ne 80 -bs 4 -lr 5e-4 -wd 3e-4 --dropout 0.5 --model_type atba --atba_enc_layers 8 --atba_encIn_dim 512 --LTA_dec_hidden_dim 256 --LTA_dec_n_head 4 --LTA_dec_layers 4 --LTA_dec_n_query 20 --crf_weight 1.0 --use_text --text_encoder distilbert` + the launcher's remaining flags.
