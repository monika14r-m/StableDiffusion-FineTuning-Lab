# Source Code Overview

This folder contains the core modules for StableDiffusion-FineTuning-Lab.

## Structure

- **data/**  
  Dataset loading, preprocessing, and augmentation utilities.

- **models/**  
  Model architectures (UNet, VAE, etc.) and wrappers.

- **training/**  
  Training loops, optimizers, and fine-tuning routines.

- **evaluation/**  
  Metrics, validation scripts, and evaluation pipelines.

- **utils/**  
  Helper functions (logging, configs, reproducibility).

## Example Usage

```python
from src.data import load_dataset
from src.models import UNetModel
from src.training import Trainer

dataset = load_dataset("path/to/data")
model = UNetModel()
trainer = Trainer(model=model, dataset=dataset, epochs=1)
trainer.train()
