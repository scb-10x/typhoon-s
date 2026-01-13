# Typhoon-S: Open Framework for Sovereign Language Models

## The Challenge

LLM development remains concentrated on English and Chinese, leaving other languages underserved. Even GPT-4 achieves only 25% accuracy on Thai legal reasoning—far from production-ready. Two barriers prevent widespread sovereign LLM development:

1. **Resource Intensity:** Modern LLMs require GPU clusters beyond most academic budgets
2. **Technical Opacity:** Leading models rarely disclose training recipes or data pipelines

Result: Sovereign models lag in general capabilities, limiting practical adoption.

## Our Approach

Typhoon-S demonstrates that **academic-scale resources can produce competitive sovereign models**. We focus on:

1. **Base-to-Instruct Pipeline:** 2-step approach (SFT + On-Policy Distillation) with full code. (**Available in `trl/`**)
2. **Pushing the Frontier in Domain-Specific:** RFT with pretraining and multi-turn RFT. (*Coming Soon*)

**Language-Agnostic:** While focused on Thai, the pipeline works for any language—just bring your own data.

## Reference Implementation

**Typhoon-S-ThaiLLM-8B-Instruct 🇹🇭** demonstrates competitive performance with state-of-the-art open-weight models on both Thai-specific and general benchmarks. Full release (data, code, technical report) in January 2026.

## 🛠️ Getting Started

- [trl/](trl/) - Complete SFT + GKD2 (On-Policy Distillation) training pipeline

---
**Status:** 🚧 Active Development | Full release January 2026
