import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)

# -----------------------------
# Example Dataset
# -----------------------------
class SimpleTextDataset(Dataset):
    """A simple dataset that tokenizes text for causal language modeling."""
    def __init__(self, tokenizer, texts, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
        }

# -----------------------------
# Lightning Module
# -----------------------------
class GPT4AllLightning(pl.LightningModule):
    """PyTorch Lightning wrapper for GPT4All-style models with DeepSpeed support."""
    def __init__(self, model_name="nomic-ai/gpt4all-j", lr=5e-5, warmup_steps=100):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lr = lr
        self.warmup_steps = warmup_steps

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

# -----------------------------
# DeepSpeed Config
# -----------------------------
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,  # ZeRO-2: optimizer + gradient sharding
        "offload_optimizer": {
            "device": "cpu",  # Offload optimizer states to CPU RAM
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",  # Offload parameters to CPU RAM
            "pin_memory": True
        }
    },
    "train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "fp16": {
        "enabled": True
    }
}

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Example data
    texts = [
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "PyTorch Lightning makes training easier."
    ]

    model_name = "nomic-ai/gpt4all-j"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create datasets
    train_dataset = SimpleTextDataset(tokenizer, texts)
    val_dataset = SimpleTextDataset(tokenizer, texts)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    # Initialize model
    model = GPT4AllLightning(model_name=model_name)

    # Trainer with DeepSpeed ZeRO-2
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        strategy=pl.strategies.DeepSpeedStrategy(config=deepspeed_config)
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)

