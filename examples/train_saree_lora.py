# Train LoRA for Saree Design Embedding
# -------------------------------------

from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
import torch
import os

# Load base model
base_model = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["cross_attention"],
    lora_dropout=0.05,
    bias="none",
    task_type="TEXT_TO_IMAGE"
)

# Attach LoRA
pipe.unet = get_peft_model(pipe.unet, lora_config)

# Training loop placeholder
train_data_dir = "data/sarees"
output_dir = "outputs/lora_saree"

os.makedirs(output_dir, exist_ok=True)

print(f"Ready to train on {train_data_dir} and save to {output_dir}")
# TODO: Add dataset loader + training loop

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob

# Simple dataset for saree images
class SareeDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.files = glob.glob(f"{img_dir}/*.png") + glob.glob(f"{img_dir}/*.jpg")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img}

# Loader
dataset = SareeDataset(train_data_dir)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training loop placeholder
for epoch in range(5):  # small test run
    for batch in loader:
        # TODO: forward pass with pipe.unet + LoRA
        # loss.backward(), optimizer.step()
        print("Training step on batch")
