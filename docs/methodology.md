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
