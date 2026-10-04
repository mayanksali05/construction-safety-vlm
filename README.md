# Construction Safety VLM

Development of a Domain-Specific Vision-Language Model for Construction Site
Safety Inspection (M.Tech major project, Nirma University).

This repo currently covers **plan step 1: the zero-shot baseline** of
Qwen2-VL-2B on the ConstructionSite 10k test split.

## Layout

| Path | What it does |
|---|---|
| `src/rules.py` | The 4 safety rules, word for word from the dataset card |
| `src/data.py` | Reads the `rule_k_violation` fields into true / false labels |
| `src/prompts.py` | The baseline prompt and the parser for the model's JSON answer |
| `scripts/01_inspect_dataset.py` | Downloads the dataset and checks counts against the paper |
| `scripts/02_run_baseline.py` | Runs the model on the test split, saves predictions |
| `scripts/03_evaluate.py` | Per-rule precision / recall / F1 |
| `slurm/baseline.sbatch` | Cluster job for steps 02 + 03 |

## Run order on the cluster

```bash
# 1. Environment (once)
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision   # pick the build for the cluster's CUDA version
pip install -r requirements.txt

# 2. Check the data (login node, needs internet)
export HF_HOME=$PWD/hf_cache
python scripts/01_inspect_dataset.py

# 3. Smoke test on 20 images (on a GPU node)
python scripts/02_run_baseline.py --limit 20 --out outputs/smoke.jsonl
python scripts/03_evaluate.py outputs/smoke.jsonl

# 4. Full baseline
mkdir -p logs && sbatch slurm/baseline.sbatch
```

Edit the two lines marked `EDIT` in `slurm/baseline.sbatch` first (partition
and environment activation).

## Run on the laptop first (Windows, GTX 1650 4 GB)

The laptop is for checking that everything works, not for the final numbers.
Open PowerShell in this folder:

```powershell
# 1. Environment (once). Everything stays on E:.
python -m venv venv-vlm
.\venv-vlm\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2. Keep the model and dataset downloads on E: (run in every new window)
$env:HF_HOME = "$PWD\hf_cache"

# 3. GPU visible? Should print True and the GTX 1650.
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 4. Check the data (no GPU needed; downloads the whole dataset)
python scripts\01_inspect_dataset.py

# 5. Smoke test: 20 images, 4-bit model, one image at a time
python scripts\02_run_baseline.py --limit 20 --load-4bit --batch-size 1 --max-pixels 401408 --out outputs\smoke.jsonl
python scripts\03_evaluate.py outputs\smoke.jsonl
```

- `--load-4bit` shrinks the model so it fits in 4 GB. It changes the results
  slightly, so the numbers reported in the thesis should come from the
  full-precision run on the cluster.
- `--max-pixels 401408` (= 512 x 28 x 28) limits image size to save GPU
  memory. If you still get "CUDA out of memory", try `--max-pixels 200704`.
- If the GPU check says "Torch not compiled with CUDA enabled", the CPU
  build of PyTorch got installed. Fix:
  `pip uninstall -y torch` then
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`.
- In cmd.exe, set the cache folder with `set HF_HOME=%CD%\hf_cache`
  instead of the `$env:` line.

## Things to know before trusting the numbers

- **Step 2 must print "All counts match the paper".** If it does not, the
  label reading in `src/data.py` is wrong for the real data format and the
  baseline numbers would be meaningless.
- **The prompt is ours**, not the one from the ConstructionSite 10k paper.
  Before putting our numbers next to the published GPT-4o / LLaVA numbers,
  check the paper's prompt and metric definition and note any difference.
- Metrics are image-level: for each rule, "does this image contain a
  violation, yes or no". Bounding boxes are not evaluated yet.
- Answers the parser cannot read are counted as "no violation" and shown in
  the `unparsed` column.
- The model inference script has not been run yet (it needs a GPU). The
  label reading, answer parsing and metric code were tested on synthetic
  data.

## Dataset

`LouisChen15/ConstructionSite` on HuggingFace. 7,009 train / 3,004 test
images. Licence tag on the dataset card: `cc-by-nc-4.0`.

Citation as given on the dataset card: Chen and Zou (2026), "Are large
pre-trained vision language models effective construction safety
inspectors", Data-Centric Engineering, vol. 7, e11.

## Next steps (from the project plan)

2. LoRA fine-tune on the train split; same evaluation.
3. Add severity and corrective-action labels; fine-tune again.
4. Add regulation citations (OSHA 29 CFR 1926, Indian OSH Code / BOCW).
5. One comparison table.
