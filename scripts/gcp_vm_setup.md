# GCP GPU VM setup — VideoLLaMA3 extraction + DELTA baselines

One-time account setup is in the Console (project, billing, Compute Engine API,
L4 GPU quota — request the quota increase first, it can take time to approve).
Everything from VM creation down is here.

## 1. Install the gcloud CLI locally (once)

```bash
brew install --cask google-cloud-sdk        # macOS
gcloud init                                  # log in, pick project
```

## 2. Create the VM

```bash
PROJECT=your-project-id
ZONE=us-central1-a          # pick a zone where your L4 quota was approved
NAME=delta-ta-l4

gcloud compute instances create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu121-debian-11 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB --boot-disk-type=pd-ssd \
  --metadata="install-nvidia-driver=True"
```

First boot installs the NVIDIA driver automatically (~2-5 min extra).

## 3. Connect

```bash
gcloud compute ssh "$NAME" --zone="$ZONE"
```

Verify the GPU:
```bash
nvidia-smi   # should list the L4
```

## 4. Environment

```bash
git clone https://github.com/epayaslii/DELTA.git && cd DELTA

# match this repo's conventions -- see docs/delta-code.md
conda create -n delta python=3.11 -y && conda activate delta
pip install -e .
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "transformers<5"   # VL3-SigLIP-NaViT's remote code needs 4.x, see backbones.py
```

## 5. Get the 50Salads data onto the VM

Either re-download (same bundle, verified identical to the Kaggle copy):
```bash
# HF bundle
wget https://huggingface.co/datasets/dinggd/50salads/resolve/main/50salads.zip -O /tmp/50s.zip
unzip /tmp/50s.zip -d data/50salads
python scripts/make_transcripts.py data/50salads
```
Raw videos: `gcloud compute scp` them up from your Desktop copy, or re-fetch
from wherever the group shared them.
```bash
gcloud compute scp --recurse ~/Desktop/50saldads_source_video "$NAME":~/DELTA/data/50salads/videos --zone="$ZONE"
```

## 6. Run the gate BEFORE a full extraction

```bash
python scripts/verify_backbone_alignment.py --backbone videollama3 --device cuda
```
Only proceed to a full extraction if this passes (see the script's own
threshold logic — calibrated against the known-inadequate siglip2 result).

## 7. Full extraction + baselines

```bash
bash scripts/slurm_extract_videollama3.sh   # adapt the #SBATCH header away, or just run the python command inside it directly
bash scripts/slurm_delta_baselines.sh       # needs third_party/delta_wlta + its own conda env, see docs/delta-code.md
```
(The `slurm_*.sh` scripts were written for the institute's SLURM cluster — on
a single GCP VM, just run the `python -m delta.features.extract ...` /
`python train.py ...` command each script wraps, directly, without `sbatch`.)

## 8. Cost control — do this every session

**Stop the VM when not actively using it** — you're billed while it's running,
not just while a job is active:
```bash
gcloud compute instances stop "$NAME" --zone="$ZONE"     # keeps the disk, stops billing for compute
gcloud compute instances start "$NAME" --zone="$ZONE"    # resume later
```
Delete entirely when the project's compute needs are done:
```bash
gcloud compute instances delete "$NAME" --zone="$ZONE"
```
L4 on `g2-standard-8` is roughly $0.7–1/hr depending on region — stopping
between sessions matters.
