# examples/fine_tune_demo.py

from src.data import load_dataset
from src.models import UNetModel
from src.training import Trainer

def main():
    # 1. Load a small dataset (dummy or real)
    dataset = load_dataset("path/to/data")

    # 2. Initialize model
    model = UNetModel()

    # 3. Run a short training loop
    trainer = Trainer(model=model, dataset=dataset, epochs=1)
    trainer.train()

    print("Demo training run complete!")

if __name__ == "__main__":
    main()
