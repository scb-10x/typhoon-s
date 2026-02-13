# Typhoon-S: Minimal Open Post-Training for Sovereign Large Language Models

This repository is archived and released as-is. If you’re interested in this work, please contact us.

**Official code release for the paper**: *"Typhoon-S: Minimal Open Post-Training for Sovereign Large Language Models"*

Minimal and reproducible post-training recipes for **sovereign settings**—scenarios requiring control over model weights, training data, and methods under resource constraints. Demonstrates that careful post-training design can achieve competitive performance without massive-scale data or compute.

**Key Results:**
- Section 2: Transform base → instruct models in **2 days on 8×H100** (8B model)
- Strong Thai performance while maintaining general capabilities
- Section 3: Achieve **78% accuracy** on Thai legal reasoning (4B model, exceeding GPT-5's 75%)
- No catastrophic forgetting after domain specialization

**Setting:** Academic-scale resources (4-8×H100 GPUs), Thai as representative low-resource language

## Overview

This repository contains two complementary post-training recipes:

### 1. Adoptability: Base → Instruct ([`trl/`](trl/))
Transform base models into general-purpose instruction-following assistants.

**Method:** SFT + On-Policy Distillation (OPD)
- **Data:** 340k samples
- **Teacher:** Qwen3-30B-A3B-Instruct-2507
- **Training:** ~2 days on 8×H100 (8B model)

**Scripts:** [`train_sft.sh`](trl/train_sft.sh), [`train_distill_8b_8gpu_30b.sh`](trl/train_distill_8b_8gpu_30b.sh)

### 2. Sovereign Capability: Domain Specialization ([`verl/`](verl/))
Enhance performance on region-specific tasks (legal reasoning, cultural knowledge).

**Method:** InK-GRPO (Injected Knowledge GRPO)
- **Innovation:** Augments GRPO with stochastic next-token prediction loss
- **Agentic RFT:** Multi-turn tool use with RAG (search + read tools)
- **Training:** ~1 day on 4×H100 (4B model)
- **Results:** +4% over GRPO, 78% on Thai legal tasks

**Experiments:** [`verl/exp/`](verl/exp/)

## Installation

```bash
# Clone repository
git clone https://github.com/scb-10x/typhoon-s
cd typhoon-s

# Install dependencies for SFT+OPD (Section 2: Adoptability)
cd trl
pip install -e .
pip install -r requirements.txt

# Install dependencies for InK-GRPO (Section 3: Sovereign Capability)
cd ../verl
pip install -e .
pip install -r requirements.txt

# Install dependencies for data processing
cd ../data_processing
pip install openai transformers torch jsonlines tqdm fasttext huggingface-hub
```

**Hardware Requirements:** 4-8×H100 GPUs (or equivalent), 80GB VRAM per GPU recommended

## Quick Start

### 1. Adoptability Training (Base → Instruct)
```bash
cd trl

# Stage 1: SFT (Supervised Fine-Tuning)
bash train_sft.sh

# Stage 2: OPD (On-Policy Distillation)
bash train_distill_8b_8gpu_30b.sh
```

See [`trl/README.md`](trl/README.md) and [`data_processing/README.md`](data_processing/README.md) for details.

### 2. Sovereign Capability Training (InK-GRPO)
```bash
cd verl/exp
# Configure your experiment (see verl/exp/ for examples)
bash train_xxx.sh
```

See [`verl/README.md`](verl/README.md) for VERL usage.

### 3. Evaluation (For Section 3: Sovereign Capability)

See [`evaluation/README.md`](evaluation/README.md) for complete evaluation instructions.

## Resources

🤗 **Hugging Face:**
- [Model Collection](https://huggingface.co/collections/typhoon-ai/typhoon-s)
- [Typhoon-S-8B-Instruct](https://huggingface.co/typhoon-ai/typhoon-s-thaillm-8b-instruct-research-preview)
- [Typhoon-S-4B-Legal-Agent](https://huggingface.co/typhoon-ai/typhoon-s-4b-nitibench-ccl-legal-agent-research-preview)
- [Section2 Training Datasets](https://huggingface.co/datasets/typhoon-ai/typhoon-s-instruct-post-training)
- [Section3 Training & Evaluation Dataset](https://huggingface.co/datasets/typhoon-ai/typhoon-s-sovereign-capability-dataset)

🌐 [Project Website](http://opentyphoon.ai) | 📄 [Paper (arXiv)](...)

## Repository Structure

```
├── trl/                   # Adoptability: SFT + OPD training
├── verl/                  # Sovereign Capability: InK-GRPO + Agentic RFT  
├── data_processing/       # AutoIF pipeline for target-language datasets
├── evaluation/            # Evaluation scripts and benchmarks
└── paper_content.tex      # Full paper LaTeX source
```

## Citation

If you use this code or models, please cite:

```bibtex
@misc{pipatanakul2026typhoonsminimalopenposttraining,
      title={Typhoon-S: Minimal Open Post-Training for Sovereign Large Language Models}, 
      author={Kunat Pipatanakul and Pittawat Taveekitworachai},
      year={2026},
      eprint={2601.18129},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.18129}, 
}
```

## License

This project is licensed under the Apache License 2.0. See:
- Main project: [LICENSE](LICENSE)
- TRL components: [trl/LICENSE](trl/LICENSE)
- veRL components: [verl/LICENSE](verl/LICENSE)

Data processing and evaluation scripts follow the main [LICENSE](LICENSE).

## Contact

For questions or issues, please open a GitHub issue or visit [opentyphoon.ai](http://opentyphoon.ai).
