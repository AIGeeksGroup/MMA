# MMA: Multimodal Memory Agent

This is the official repository for the paper:

> **MMA: Multimodal Memory Agent**
>
> Yihao Hu\*, Wanru Cheng\*, Zeyu Zhang†, Hao Tang‡
>
> \*Equal contribution. †Project lead. ‡Corresponding author.
>
> **_Submitted to ACL 2026_**
>
> ### [Paper](https://arxiv.org/abs/xxxx.xxxxx) | [Project Page](https://github.com/AIGeeksGroup/MMA)
>
> _⚠️ **Repository Structure**: This repo contains the **MMA Agent framework** (based on MIRIX) and the **MMA-Bench evaluation toolkit**._

## 📖 Introduction

Long-horizon multimodal agents often suffer from **"Blind Faith"**—relying on stale, low-credibility, or conflicting retrieved memories. This triggers overconfident errors in safety-critical scenarios.

We present **MMA (Multimodal Memory Agent)**, a confidence-aware architecture that assigns dynamic reliability scores to memories based on:

- **Source Credibility ($S$)**: Prioritizing trusted users over gossip.
- **Temporal Decay ($T$)**: Discounting stale information.
- **Conflict-Aware Consensus ($C_{con}$)**: Reweighting evidence based on semantic support.

We also introduce **MMA-Bench**, a diagnostic benchmark designed to probe **Belief Dynamics** and **Cognitive Robustness**. Using this framework, we uncover the **"Visual Placebo Effect"**, where agents become overconfident when presented with irrelevant visual noise.

## ⚙️ Installation

### 1. Environment Setup

We recommend creating a unified environment for both the agent and the benchmark.

```bash
# 1. Create environment
conda create -n mma python=3.10 -y
conda activate mma

# 2. Install dependencies (Split by module)
pip install -r MMA/requirements.txt        # For the Agent (RAG, VectorDB)
pip install -r MMA-Bench/requirements.txt  # For Benchmark Gen & Eval (OpenAI, DashScope)
```

````

### 2. User Configuration (🔑 Input Required)

You must configure API keys and data paths before running experiments.

**Option A: Environment Variables (Recommended)**
Create a `.env` file in the root directory:

```env
# --- For MMA Agent, Baselines & Judge ---
OPENAI_API_KEY="sk-..."           # For GPT-4.1-mini(Baseline) & GPT-4o-mini (Judge)

# --- For MMA-Bench Generation & Qwen Baselines ---
DASHSCOPE_API_KEY="sk-..."        # For Qwen3-Max (Logic) & Qwen-Image-Plus (Vision) & Qwen3-VL-Plus (Baseline)
```

**Option B: Dataset Preparation**

- **LOCOMO**: Download `locomo10.json` and place it in `MMA/public_evaluations/data/`.
- **FEVER**: Download `paper_dev.jsonl` and place it in `MMA/public_evaluations/data/`.

## 🛠️ MMA-Bench: The Cognitive Robustness Benchmark

MMA-Bench evaluates **Epistemic Prudence** (knowing when to abstain) and **Conflict Resolution** across 30 programmatically generated cases with 10 temporal sessions each.

### 1. Data Generation

Generate the "Trust-Trap" evolution graph (Text + Visual Evidence).

> _Requires Qwen3-Max (Logic) and Qwen-Image-Plus (Vision)._

```bash
cd MMA-Bench
python -m src.generator --num_cases 30 --output_dir ./data/mma_bench_v1
```

### 2. Run Inference

Evaluate models under different modality settings to detecting **Visual Anchoring**.

```bash
# Text Mode (Oracle Captions): Tests pure reasoning logic
python -m src.inference --model gpt-4.1-mini --mode text

# Vision Mode (Raw Images): Tests multimodal conflict resolution
python -m src.inference --model qwen3-vl-plus --mode vision
```

### 3. Evaluation (CoRe Score)

Compute the **Cognitive Reliability (CoRe) Score** using the 3-step probe (Verdict, Wager, Reflection).

```bash
python -m src.grader --judge_model gpt-4o-mini
```

## 🧪 Experiments & Reproducibility

### 1. FEVER (Fact Verification)

Evaluate **Stability**. MMA matches baseline accuracy while reducing standard deviation by **35.2%**.

```bash
cd MMA/public_evaluations

# Run Ablation Study (Baseline vs. Full MMA vs. w/o Consensus)
python run_fever_eval.py \
  --fever_data_path data/paper_dev.jsonl \
  --config_path ../configs/mirix.yaml \
  --limit 500 \
  --seeds "42,922,2025" \
  --formula_modes "st,tri"
```

### 2. LOCOMO (Long-Context QA)

Evaluate **Safety** in sparse retrieval. The `st` variant (Source + Time) achieves the highest Utility (609.0) by minimizing hallucinations.

```bash
cd MMA/public_evaluations

# Run LOCOMO Evaluation
python run_instance.py \
  --agent_name mma \
  --dataset LOCOMO \
  --global_idx 0 \
  --config_path ../configs/mma_gpt4.yaml
```

### 3. MMA-Bench Results (Type B & D)

- **Type B (Inversion)**: MMA achieves **41.18%** Vision Dominance (solving the conflict), while Baseline collapses to 0%.
- **Type D (Unknowable)**: Visual inputs cause a drop in CoRe score (0.69 → -0.38), illustrating the **Visual Placebo Effect**.

## 📂 Directory Structure

```text
.
├── MMA/                    # [Agent Framework] Core implementation
│   ├── MMA/                # Source code for the Agent (Memory, Confidence)
│   ├── configs/            # Configuration YAMLs for different backends
│   ├── public_evaluations/ # Evaluation scripts (FEVER, LOCOMO)
│   ├── frontend/           # Web UI for the agent
│   ├── scripts/            # Utility scripts (e.g., DB reset)
│   ├── tests/              # Unit tests
│   └── requirements.txt    # Shared dependencies
├── MMA-Bench/              # [Benchmark Toolkit]
│   ├── mma_bench/          # Source code (Generator, Inference, Grader)
│   ├── data/               # Generated Benchmark Data & Images
│   └── requirements.txt    # Shared dependencies
└── README.md               # Main Documentation
```

## Acknowledgements

We acknowledge the use of the following resources:

- [**MIRIX**](https://github.com/Mirix-AI/MIRIX): Foundational memory architecture.
- **Base Models**: Qwen & GPT (API).

## Citation

If you find our work useful, please cite:

```bibtex
@article{,
  title={MMA: Multimodal Memory Agent},
  author={Yihao Lu, Wanru Cheng, Zeyu Zhang, Hao Tang},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
````
