
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

The workspace is organized into the following repository structure:

```text
.
├── Gold_Standard.xlsx            # Primary Gold Standard dataset (500 manually annotated reviews)
├── Silver_Standard.csv           # Silver Standard dataset (215,063 automatically annotated reviews)
├── drugsCom_ALL_215k.csv         # Raw dataset of 215k patient reviews from Drugs.com
├── drugsCom_V10.5_Final.csv      # Processed dataset version 10.5 ready for experiments
├── drugCom_ADR_Pscore.csv        # Calculated ADR propensity/risk scores for drug reviews
│
├── Knowledge_Bases/
│   ├── drug_names.tsv            # Normalized drug entity dictionary
│   └── meddra_all_se.tsv         # MedDRA side effect terminology dictionary
│
└── LLM_Benchmark_Evaluations/    # Model output logs for comparative benchmark analysis
    ├── Khao_Sat_ChatGPT_Labeled.xlsx
    ├── Khao_Sat_Gemini_Labeled.xlsx
    ├── Khao_Sat_Gork3_Labeled.xlsx
    └── Khao_sat_Hybrid.xlsx

```

---

## 4. Label Definition

Binary classification labels are structured as follows:

| Label | Definition | Description |
| --- | --- | --- |
| **1** | **ADR Positive** | Review explicitly or implicitly describes one or more Adverse Drug Reactions. |
| **0** | **ADR Negative** | No Adverse Drug Reaction is reported in the review text. |

---

## 5. Data Access & Download

Due to GitHub file size limits for large files (`Silver_Standard.csv` and `drugsCom_V10.5_Final.csv`), the full high-resolution raw files can also be accessed directly via Google Drive:

🔗 **[Click here to access the Google Drive Folder](https://drive.google.com/drive/folders/18RNQmXnNxJFsB3Qr3zhlw2B3JYRHV5KV?usp=sharing)**

---

## 6. Citation
---

## 7. License & Data Source

* **Original Data Source:** All unstructured reviews were collected from [Drugs.com](https://www.drugs.com/).
* **Usage Policy:** This dataset is provided strictly for **research and academic purposes**. Users must comply with the terms of use of the original data source.

---

## 8. Contact & Authors

* **Dinh Tai Pham**
*Faculty of Information Technology, Nguyen Tat Thanh University, Ho Chi Minh City, Vietnam*
📧 Email: `pdtai@ntt.edu.vn`
* **Dr. Huyen Trang Phan** *(Corresponding Author)*
*Faculty of Information Technology, HCMC University of Technology and Education, Vietnam*
📧 Email: `trangpth@hcmute.edu.vn`

```

```
