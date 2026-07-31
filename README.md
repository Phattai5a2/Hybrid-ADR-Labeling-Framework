# Hybrid ADR Labeling Framework Dataset

## 1. Overview

This repository accompanies the study entitled:

> **Hybrid ADR Labeling Framework with Expert-Validated Gold Standard and Medical Knowledge Integration**

The purpose of this repository is to facilitate transparency, reproducibility, and future research on automatic Adverse Drug Reaction (ADR) detection from patient-generated drug reviews.

The repository provides access to the datasets used and generated in this study, including the manually annotated Gold Standard dataset and the large-scale automatically constructed Silver Standard dataset.

---

## 2. Datasets

This study utilizes two complementary datasets.

### Gold Standard Dataset

The Gold Standard dataset contains **500 manually annotated patient reviews** collected from Drugs.com.

Each review was independently examined and validated according to ADR annotation guidelines using expert consensus.

The Gold Standard dataset was used for:

- evaluation of labeling quality
- hyperparameter optimization
- error analysis
- ablation studies

---

### Silver Standard Dataset

The proposed Hybrid Labeling Framework was applied to a large collection of patient reviews from Drugs.com, producing a **Silver Standard dataset containing 215,063 automatically annotated reviews**.

The labeling process integrates:

- contextual semantic inference using **DeBERTa-v3-large**
- rule-based linguistic analysis
- structured pharmacological knowledge from **SIDER**
- medical terminology validation using **MedDRA**
- adaptive confidence calibration

The resulting Silver Standard dataset is intended for large-scale machine learning research on pharmacovigilance, ADR detection, and safety-aware drug recommendation systems.

---

## 3. Dataset Contents

The released repository contains:

```
Gold_Standard/
    gold_standard.csv

Silver_Standard/
    silver_standard.csv

Metadata/
    label_description.pdf
    annotation_guideline.pdf

Sample/
    sample_reviews.csv
```

The complete dataset is publicly available through Google Drive.

---

## 4. Data Access

The complete datasets can be downloaded from

**Google Drive**

https://drive.google.com/drive/folders/XXXXXXXXXXXX

(Replace with your public sharing link.)

---

## 5. Label Definition

Binary labels are defined as

| Label | Description |
|-------|-------------|
| 1 | Review contains one or more Adverse Drug Reactions (ADR) |
| 0 | No ADR is reported |

---

## 6. Data Source

Patient reviews were collected from

**Drugs.com**

https://www.drugs.com/

All annotations were generated solely for research purposes.

---

## 7. Citation

If you use this dataset in your research, please cite:

```bibtex
@article{pham2026,
  title={Hybrid ADR Labeling Framework with Expert-Validated Gold Standard and Medical Knowledge Integration},
  author={Pham, Dinh Tai and Phan, Huyen Trang and Nguyen, Ngoc Thanh},
  journal={Under Review},
  year={2026}
}
```

---

## 8. License

This dataset is released for **research and educational purposes only**.

Users should comply with the terms of use of the original data source (Drugs.com).

---

## 9. Contact

**Dinh Tai Pham**

Faculty of Information Technology

Nguyen Tat Thanh University

Ho Chi Minh City, Vietnam

Email:

pdtai@ntt.edu.vn

Corresponding Author:

Dr. Huyen Trang Phan

Faculty of Information Technology

HCMC University of Technology and Education

Email:

trangpth@hcmute.edu.vn

---

## 10. Related Publication

Hybrid ADR Labeling Framework with Expert-Validated Gold Standard and Medical Knowledge Integration

(Under Review)
