# MeloMatch: Dual-Signal RAG for Musical Theatre Recommendation
# CSC5051 NLP Final Project — Spring 2026, CUHK-Shenzhen

## Project Structure

```
csc5051_final_proj/
├── configs/
│   ├── config.yaml           # All hyperparameters & paths
│   └── prompts/              # Externalized prompt templates
├── data/
│   ├── raw/                  # Raw questionnaire exports (CSV/JSON)
│   ├── processed/            # Cleaned preference-reason pairs
│   ├── indices/              # BM25 / FAISS indices (built by build_index.py)
│   └── knowledge_base/       # Musical metadata corpus (~200 musicals)
├── src/
│   ├── llm_backend.py        # Unified LLM backend (LocalLLM + VllmLLM + APILLM)
│   ├── utils.py              # Shared utilities (config, logging, seeding)
│   ├── data/
│   │   ├── preprocess.py     # Parse questionnaire → structured pairs
│   │   ├── splits.py         # Hold-out & stratified sampling (SHA-256)
│   │   └── knowledge.py      # Build & manage musical KB
│   ├── retrieval/
│   │   ├── indexer.py        # Build BM25 / Dense / Hybrid indices
│   │   └── retriever.py      # Three retrieval conditions (signal-aware)
│   ├── generation/
│   │   └── generator.py      # LLM-based recommendation generation
│   ├── filtering/
│   │   ├── nli_filter.py     # Post-retrieval NLI verification (ARAG-inspired)
│   │   └── reranker.py       # Contrastive re-ranking (Seshadri-inspired)
│   ├── profiles/
│   │   └── preference_profiler.py  # Structured discomfort dimension extraction
│   └── evaluation/
│       └── metrics.py        # Hit@K, Avoidance@K, Faithfulness, IAA, A/B
├── scripts/
│   ├── build_index.py        # One-shot: preprocess + index
│   ├── curate_kb.py          # Build knowledge base from sources
│   ├── run_experiment.py     # Run all conditions + baselines + ablations
│   └── analyze_results.py    # Tables, plots, significance tests
├── demo/
│   └── app.py                # Gradio interactive demo (3-way comparison)
├── notebooks/
│   ├── melomatch_colab.py    # Colab A100: zero_shot (Qwen3-14B, vLLM)
│   └── melomatch_kaggle.py   # Kaggle T4×2: fine_tuned (Qwen3-8B, vLLM)
├── tests/
│   └── test_core.py          # Unit tests (pytest)
├── results/                  # Experiment outputs (gitignored)
├── requirements.txt
└── README.md
```

## Quick Start

### Local (with GPU)

```bash
pip install -r requirements.txt

# Step 1: Curate knowledge base (~200 musicals)
python scripts/curate_kb.py

# Step 2: Preprocess questionnaire data + build indices
python scripts/build_index.py

# Step 3: Run all conditions (4 signals × 3 methods × 2 backbones)
python scripts/run_experiment.py

# Step 3b: Run ablations (retrieval depth + data scaling)
python scripts/run_experiment.py --ablation-k --ablation-scale

# Step 4: Analyze results
python scripts/analyze_results.py --run results/run_XXXXXXXX_XXXXXX

# Step 5: Launch demo
python demo/app.py
```

### Cloud Deployment (Colab + Kaggle Split)

Experiments are split across two platforms for GPU efficiency:

| Platform | GPU | Notebook | Backbone | Model | Conditions |
|----------|-----|----------|----------|-------|------------|
| **Google Colab** | A100 80GB | `melomatch_colab.py` | zero_shot | Qwen3-14B (bf16) | 12 main + baseline + ablations |
| **Kaggle** | T4×2 (32GB) | `melomatch_kaggle.py` | fine_tuned | Qwen3-8B (fp16) | 12 main + baseline |

Both use **vLLM** for optimized inference (no API key required).

#### Colab A100 Setup

1. Upload dataset to Google Drive under `melomatch/data/` (or adjust `DATA_DIR` in the notebook)
2. Select **A100 GPU** runtime
3. Upload `notebooks/melomatch_colab.py` as a Colab notebook
4. Run all cells — outputs save to `melomatch/results/` on Drive

**Config changes for Colab** (already set in the notebook):
```python
ZERO_SHOT_MODEL = "Qwen/Qwen3-14B"
BACKBONES = ["zero_shot"]
# vLLM settings:
dtype = "bfloat16"           # A100 supports bf16 natively
tensor_parallel_size = 1     # Single A100
gpu_memory_utilization = 0.90
```

**Output files:**
- `metrics_zero_shot.json` — main experiment results
- `k_ablation_metrics.json` — retrieval depth ablation
- `scale_ablation_metrics.json` — data scaling ablation
- `signal_comparison_zero_shot.png` — visualization

#### Kaggle T4×2 Setup

1. Upload dataset as a Kaggle dataset named `melopreference`
2. Create a new notebook, select **GPU T4×2** accelerator
3. Upload `notebooks/melomatch_kaggle.py` as a Kaggle notebook
4. Run all cells — outputs save to `/kaggle/working/results/`

**Config changes for Kaggle** (already set in the notebook):
```python
FINE_TUNED_MODEL = "Qwen/Qwen3-8B"
BACKBONES = ["fine_tuned"]
# vLLM settings:
dtype = "float16"            # T4 does NOT support bf16
tensor_parallel_size = 2     # Use both T4 GPUs
gpu_memory_utilization = 0.90
```

**Output files:**
- `metrics_fine_tuned.json` — main experiment results
- `signal_comparison_fine_tuned.png` — visualization

#### Merging Results

After both runs complete, download both `metrics_*.json` files and merge:

```python
import json

with open("metrics_zero_shot.json") as f:
    zs = json.load(f)
with open("metrics_fine_tuned.json") as f:
    ft = json.load(f)

all_metrics = {**zs, **ft}
with open("metrics_all.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
```

Then run analysis: `python scripts/analyze_results.py --metrics metrics_all.json`

### Config Reference (`configs/config.yaml`)

The config file uses `backend: "vllm"` by default. To change backends:

```yaml
# Option 1: vLLM (recommended for Colab/Kaggle)
backend: "vllm"
dtype: "bfloat16"
tensor_parallel_size: 1

# Option 2: HuggingFace Transformers (fallback)
backend: "local"
quantization: "4bit"

# Option 3: DashScope API (fastest, requires API key)
backend: "api"
api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

## Dataset

**MeloPreference**: 100 questionnaires × 6 preference-reason pairs = 600 pairs

## Experiment Design

4 (retrieval signal) × 3 (retrieval method) × 2 (LLM backbone) = **24 conditions** + baselines

| Factor           | Levels                                                   |
|------------------|----------------------------------------------------------|
| Retrieval signal | Positive-only, Dual-signal, Negative-only, Dual-enhanced |
| Retrieval method | BM25, Dense, Hybrid (RRF)                                |
| LLM backbone     | Qwen3-14B (zero-shot, vLLM), Qwen3-8B (fine-tuned, vLLM) |

**Additional conditions:**
- Pure LLM baseline (no retrieval) × 2 backbones
- Retrieval depth ablation (k = 1, 3, 5, 10)
- Data scaling ablation (n = 25, 50, 100 users)

**Dual-enhanced** = Dual-signal + NLI post-retrieval filtering + structured preference profiling.

## Models

| Role | Model | Inference | Platform | VRAM |
|------|-------|-----------|----------|------|
| Zero-shot backbone | `Qwen/Qwen3-14B` | vLLM, bf16 | Colab A100 | ~28 GB |
| Fine-tuned backbone | `Qwen/Qwen3-8B` | vLLM, fp16, TP=2 | Kaggle T4×2 | ~16 GB |
| Embeddings | `BAAI/bge-base-zh-v1.5` | sentence-transformers | Both | ~1 GB |

All inference runs locally via **vLLM** (no API key required). An API fallback (`APILLM`) is available in `src/llm_backend.py` for DashScope/vLLM servers. A HuggingFace Transformers fallback (`LocalLLM`) with BitsAndBytes 4-bit quantization is also supported.

## Metrics

- **Hit@K** (K=1,3,5): Is the held-out liked musical in the top-K recommendations?
- **Avoidance@K** (K=3,5): Do top-K avoid the user's disliked items/features?
- **Explanation Faithfulness**: Human-rated (1–5 Likert)
- **Blind A/B Preference**: Win rate across conditions
- **Inter-annotator agreement**: Cohen's κ (pairwise) + Krippendorff's α (ordinal)

## TODO

- [x] **KB expansion**: `data/knowledge_base/musical_metadata.csv` — ~162 musicals with verified metadata
- [x] **Questionnaire data**: `data/raw/user_profile.csv` + `subjective_reviews.csv` + `musical_tags.csv` (230+ responses)
- [x] **Processed data**: `data/processed/` — users.jsonl, positive_pairs.jsonl, negative_pairs.jsonl
- [ ] **Fine-tuned model checkpoint**: Run QLoRA fine-tuning on Kaggle, then set `lora_checkpoint` path
- [ ] **Merge results**: Combine Colab + Kaggle outputs for final analysis
