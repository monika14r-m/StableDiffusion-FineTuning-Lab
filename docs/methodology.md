# Methodology

## 1. Objective
This project provides a clean, modular, and reproducible pipeline for fine‑tuning Stable Diffusion using modern techniques such as **LoRA**, **DreamBooth**, and **Textual Inversion**.  
The goal is to enable fast experimentation, dataset‑driven customization, and high‑quality generative outputs.

---

## 2. Fine‑Tuning Approaches

### 2.1 LoRA (Low‑Rank Adaptation)
- Lightweight and GPU‑efficient
- Injects trainable low‑rank matrices into attention layers  
- Ideal for styles, characters, and product-specific fine‑tuning  

### 2.2 DreamBooth
- Identity‑preserving fine‑tuning  
- Works best with 10–20 images  
- Suitable for people, pets, branded objects, or unique identities  

### 2.3 Textual Inversion
- Learns new concept embeddings  
- Very small file size (KBs)  
- Useful for textures, styles, and abstract concepts  

---

## 3. Dataset Preparation

### 3.1 Folder Structure

### 3.2 Requirements
- Recommended resolution: **512×512** or **768×768**  
- Consistent lighting and framing improves results  
- Remove duplicates and blurry images  
- 10–100 images depending on method  

---

## 4. Training Pipeline

### 4.1 Steps
1. Load base model (e.g., `runwayml/stable-diffusion-v1-5` or SDXL)  
2. Preprocess dataset (resize, crop, normalize)  
3. Load training configuration (`config.yaml`)  
4. Select method (LoRA / DreamBooth / Textual Inversion)  
5. Train using HuggingFace Diffusers + Accelerate  
6. Save outputs:  
   - LoRA weights (`.safetensors`)  
   - DreamBooth model folder  
   - Textual inversion embeddings (`.pt`)  

---

## 5. Evaluation

### 5.1 Metrics
- Prompt‑to‑image alignment  
- Identity preservation (DreamBooth)  
- Style fidelity (LoRA/TI)  
- Negative prompt robustness  
- Inference quality  

### 5.2 Visual Tests
- Fixed prompt comparisons  
- Style transfer tests  
- Variation sampling  
- Negative prompt stress tests  

---

## 6. Roadmap
- [ ] Add training scripts (`src/train.py`)  
- [ ] Add dataset loader (`src/dataset.py`)  
- [ ] Add LoRA training notebook  
- [ ] Add DreamBooth training notebook  
- [ ] Add inference demo notebook  
- [ ] Add example outputs  
- [ ] Add benchmarking results  

---

## 7. References
- HuggingFace Diffusers  
- LoRA: Hu et al., 2021  
- DreamBooth: Ruiz et al., 2022  
- Textual Inversion: Gal et al., 2022  

