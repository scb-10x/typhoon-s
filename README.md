# Typhoon S: Open Minimal Post-Training for Sovereign Large Language Models

## Overview

**Typhoon S** provides a minimal and reproducible post-training recipe tailored for **sovereign deployments**—settings where a country, institution, or domain owner must retain control over model weights and training data while operating under strict resource constraints (e.g., academic-scale compute).

We demonstrate that competitive instruction-following and frontier capabilities (reasoning, agents) can be achieved without massive general-purpose corpora or complex proprietary pipelines.

## Project Structure

This repository is divided into two main components corresponding to the two core capabilities required for sovereign LLMs:

### 1. Adoptability (Base $\rightarrow$ General Assistant)
**Location:** [`trl/`](trl/)

Focuses on transforming a base model into a general-purpose instruction-following assistant. 
- **Method:** Lightweight Supervised Fine-Tuning (SFT) followed by **On-Policy Distillation (OPD)**.
- **Key Insight:** Full-logits OPD significantly improves robustness (e.g., code-switching) and general performance compared to SFT alone.
- **Resources:** Implemented using HuggingFace TRL and Transformers.

### 2. Sovereign Capability (Domain Specialization)
**Location:** [`verl/`](verl/)

Focuses on enhancing performance on locally critical tasks (e.g., legal reasoning, cultural knowledge) that are often underrepresented in base models.
- **Method:** Small-scale Reinforcement Fine-Tuning (**RFT**) using **GRPO**.
- **Extensions:** 
  - **RFT + Pretraining:** Parallel next-token prediction on in-domain text to inject local knowledge.
  - **Agentic RFT:** Multi-turn tool use optimization (search, read) for retrieval-augmented generation.
- **Resources:** Built on top of [veRL](https://github.com/volcengine/verl).

## Key Results

- **Efficiency:** The entire pipeline is designed for academic resources (< 1 week of 8-GPU training for an 8B model).
- **Performance:** 
  - **Adoptability:** Successfully transforms sovereign base models (e.g., ThaiLLM) into instruction-tuned models competing with global open-weight models.
  - **Sovereign Capability:** RFT with parallel pretraining improves local legal reasoning and domain knowledge retention.

## Models & Resources

- **HuggingFace:** [`Typhoon-S-8B-Instruct`](https://huggingface.co/typhoon-ai/typhoon-s-thaillm-8b-instruct-research-preview)
- **Technical Report:** [Coming Soon]

## Getting Started

Please refer to the `README.md` in the respective subdirectories:
- Go to [`trl/`](trl/) for SFT and On-Policy distillation recipes.
- Go to [`verl/`](verl/) for RFT and Agentic training recipes.

---
**Status:** 🚧 Active Development | Full release January 2026
