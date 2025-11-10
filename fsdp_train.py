import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.strategies import FSDPStrategy
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
    """PyTorch Lightning wrapper for GPT4All-style models with FSDP support."""
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

    # Trainer with FSDP
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        strategy=FSDPStrategy(
            auto_wrap_policy={torch.nn.Linear},  # Wrap only Linear layers
            sharding_strategy="FULL_SHARD",      # Full parameter, gradient, optimizer sharding
            cpu_offload=True,                    # Offload to CPU to save GPU memory
            state_dict_type="full"               # Full state dict for saving
        )
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)

