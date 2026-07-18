# Part-Aware-Transformer


## Abstract
Domain generalization person re-identification (DG-ReID) aims to train a model on source domains and generalize well on unseen domains.
Vision Transformer usually yields better generalization ability than common CNN networks under distribution shifts. 
However, Transformer-based ReID models inevitably over-fit to domain-specific biases due to the supervised learning strategy on the source domain.
We observe that while the global images of different IDs should have different features, their similar local parts (e.g., black backpack) are not bounded by this constraint. 
Motivated by this, we propose a pure Transformer model (termed Part-aware Transformer) for DG-ReID by designing a proxy task, named Cross-ID Similarity Learning (CSL), to mine local visual information shared by different IDs. This proxy task allows the model to learn generic features because it only cares about the visual similarity of the parts regardless of the ID labels, thus alleviating the side effect of domain-specific biases. 
Based on the local similarity obtained in CSL, a Part-guided Self-Distillation (PSD) is proposed to further improve the generalization of global features. 
Our method achieves state-of-the-art performance under most DG ReID settings. 

## Framework
<div align=center><img src="https://github.com/liyuke65535/Part-Aware-Transformer/assets/39180877/f400b553-5a58-4238-9cde-a0d66e232586"></div>


The project explores architectural improvements, feature engineering, retrieval refinement, and post-processing methods to improve retrieval performance on challenging cross-view urban scenes.

---

## Overview

Urban element re-identification differs from traditional object recognition because the goal is to retrieve the **same physical object** from images captured by different cameras.

The dataset contains four object categories:

- 🗑️ Rubbish Bins
- 🚶 Crosswalks
- 📦 Containers
- 🚸 Traffic Signs

The most difficult challenge is **traffic signs**, where the query camera observes the **back side** of signs while gallery cameras observe the **front**, resulting in almost no shared appearance information. This class accounts for approximately **62% of all queries**, making it the primary bottleneck for overall performance. :contentReference[oaicite:0]{index=0}

---

## Project Goals

- Improve retrieval mAP over the PAT baseline
- Investigate transformer feature representations
- Optimize re-ranking strategies
- Explore augmentation techniques
- Study architectural variants
- Analyze class-wise performance

---

## Methodology

### Baseline

- PAT (Part-Aware Transformer)
- ViT-Base backbone
- Single CLS token
- Cosine similarity retrieval
- k-reciprocal re-ranking

Initial performance:

| Metric | Value |
|---------|------:|
| Overall mAP | **21.2%** |

The baseline serves as the reference point for all experiments. :contentReference[oaicite:1]{index=1}

---

## Experiments

### 1. Intermediate Feature Concatenation

Investigated concatenating CLS tokens from intermediate transformer layers to incorporate low-level texture information.

**Result**

- Improved rubbish bins
- Hurt traffic sign performance
- Overall performance decreased

---

### 2. Super Resolution

Applied Real-ESRGAN as an offline preprocessing step for low-resolution crops.

Pipeline:

- Images ≥64 px: unchanged
- Images between 32–64 px: ×4 Real-ESRGAN
- Images <32 px: bicubic interpolation

Although super-resolution improved visual quality, it produced only marginal retrieval gains and was not included in the final solution. :contentReference[oaicite:2]{index=2}

---

### 3. Part Token Features

Instead of using only the CLS embedding, spatial part tokens from the last transformer blocks were averaged and concatenated.

Embedding:

```
Final Embedding = [CLS || Part Tokens]
```

Result:

- Significant spatial information
- +4.2 mAP improvement
- Adopted in all subsequent experiments

:contentReference[oaicite:3]{index=3}

---

### 4. Re-ranking Optimization

Extensive tuning of k-reciprocal re-ranking parameters.

Explored:

- k1
- k2
- λ

Also evaluated:

- Average Query Expansion (AQE)
- Database Augmentation (DBA)

Hybrid re-ranking achieved the strongest retrieval performance by combining:

- k-reciprocal for Containers, Crosswalks, and Rubbish Bins
- AQE + DBA for Traffic Signs

This produced an overall local evaluation of **30.97% mAP**. :contentReference[oaicite:4]{index=4}

---

### 5. Data Augmentation

Investigated:

- Perspective transformations
- View-aware augmentation
- Gaussian Blur
- Aspect padding
- Random Erasing

View-aware augmentation consistently improved robustness to viewpoint changes.

---

### 6. Optimizer Comparison

Compared SGD with AdamW.

AdamW demonstrated better optimization stability and produced the highest validation performance.

Best configuration:

- AdamW
- Learning rate: 3e-4
- Batch size: 64
- Gaussian Blur
- Aspect padding

:contentReference[oaicite:5]{index=5}

---

### 7. Architecture Variants

Evaluated:

- ViT-B (PAT)
- Swin Transformer + SPP
- Camera Embeddings

Although Swin improved texture-rich classes, ViT remained superior overall.

Camera embeddings did not generalize well on the available training data.

---

## Results

| Stage | Overall mAP |
|--------|------------:|
| PAT Baseline | 21.2% |
| Part Tokens | 25.4% |
| Re-ranking | 27.5% |
| UAM + View Augmentation | 28.2% |
| AdamW + Feature Refinement | **31.1%** |

Overall improvement:

```
21.2% → 31.1%
(+9.9 percentage points)
```

:contentReference[oaicite:6]{index=6}

---

## Key Findings

- Part-aware features significantly improve retrieval.
- Correct re-ranking parameters are critical.
- Traffic signs remain the hardest class due to front/back appearance mismatch.
- Better optimization contributes more than complex architectural changes.
- Context and geometry appear more informative than raw appearance for difficult objects.

---

## Repository Structure

```
Urban_Elements_ReID/
│
├── configs/                 # Training configurations
├── datasets/                # Dataset utilities
├── models/                  # Model implementations
├── losses/                  # Loss functions
├── utils/                   # Helper utilities
├── reranking/               # Retrieval refinement
├── train.py                 # Training script
├── inference.py             # Evaluation / inference
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Snehaified/Urban_Elements_ReID.git

cd Urban_Elements_ReID

pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Evaluation

```bash
python inference.py
```

---

## Future Work

Potential directions include:

- View-specific embeddings
- Temporal and GPS-aware retrieval
- Multi-model ensembles
- Larger Vision Transformers
- Context-aware retrieval using surrounding scene information

---

## Acknowledgements

This work was completed as part of the **DLVSP Urban Elements Re-Identification Challenge**.

Developed by:

- **Sneha Chaudhary**
- **Dario Ranieri**

Sagarmatha Invicta
## Citation
If you find this repo useful for your research, you're welcome to cite our paper.
```
@inproceedings{ni2023part,
  title={Part-Aware Transformer for Generalizable Person Re-identification},
  author={Ni, Hao and Li, Yuke and Gao, Lianli and Shen, Heng Tao and Song, Jingkuan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11280--11289},
  year={2023}
}
```
