# Knowledge-Guided Hybrid ADR Extraction Framework & Datasets

[![Journal](https://img.shields.io/badge/Journal-Springer%20Data%20Mining%20%26%20Knowledge%20Discovery-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](#)
[![License](https://img.shields.io/badge/License-Research%20Only-orange)](#)

## 1. Overview

This repository provides the official datasets and benchmarking results for the study:

> **Knowledge-Guided Hybrid ADR Extraction with Dynamic Thresholding for Safe Drug Recommendation**

Automatic detection of Adverse Drug Reactions (ADR) from patient-generated drug reviews is challenging due to noisy text, informal phrasing, and complex negation structures. This project introduces a **Hybrid Labeling Framework** that integrates:
1. **Contextual Semantic Inference** via DeBERTa-v3-large.
2. **Rule-Based Linguistic Processing** for negation detection.
3. **Structured Knowledge Base Matching** using SIDER and MedDRA.
4. **Adaptive Confidence Calibration** via a Dynamic Multi-Variable Thresholding mechanism.

---

## 2. Datasets & Files Overview

The repository contains two core datasets alongside external knowledge bases and experimental benchmark logs:

* **Gold Standard Dataset (`Gold_Standard.xlsx`):** Consists of 500 patient reviews manually annotated and consensus-validated by a **23-member expert panel** ($\kappa = 0.68$, Fleiss' Kappa). Used for quality evaluation, error analysis, and LLM benchmarking.
* **Silver Standard Dataset (`Silver_Standard.csv`):** A large-scale automatically annotated dataset containing **215,063 reviews** across 3,412 medications, constructed using our proposed Hybrid Framework.
* **Downstream Safety Integration (`drugCom_ADR_Pscore.csv`):** Pre-calculated safety scores used to penalize ADR risks in Drug Recommendation Systems (DRS).

---

## 3. Dataset & File Contents

The workspace contains the following files:

```text
.
├── Gold_Standard.xlsx            # Expert-validated Gold Standard benchmark set (n = 500)
├── Silver_Standard.csv           # Large-scale automatically annotated dataset (N = 215,063)
├── drugsCom_ALL_215k.csv         # Raw collected patient reviews from Drugs.com
├── drugsCom_V10.5_Final.csv      # Processed dataset version 10.5 ready for modeling
├── drugCom_ADR_Pscore.csv        # Calculated ADR propensity/risk scores for DRS integration
│
├── drug_names.tsv                # Normalized drug entity dictionary
├── meddra_all_se.tsv             # MedDRA side-effect terminology database
│
├── Khao_sat_Hybrid.xlsx          # Proposed Hybrid Model predictions on Gold Standard
├── Khao_Sat_ChatGPT_Labeled.xlsx # ChatGPT (GPT-4o) baseline evaluation log
├── Khao_Sat_Gemini_Labeled.xlsx  # Gemini 1.5 Pro baseline evaluation log
└── Khao_Sat_Gork3_Labeled.xlsx   # Grok 3 baseline evaluation log
