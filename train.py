# %% IMPORTS
from modular_addition import transformers, helpers
from transformers import Trainer as HFTrainer, TrainingArguments
from torch.utils.data import Dataset
import torch as t
import time
import wandb

# %% CONFIG
config = transformers.Config(num_epochs=30_000)

# %% PART 1 — local Trainer (grokking paper)
trainer = transformers.Trainer(config)
trainer.initial_save_if_appropriate()
for epoch in range(config.num_epochs):
    train_loss, test_loss = trainer.do_a_training_step(epoch)
    if test_loss.item() < config.stopping_thresh:
        break
trainer.post_training_save()

# %% PART 2 — HuggingFace Trainer
# Close the wandb run opened by the local Trainer above so HFTrainer can start its own.
wandb.finish()


class ModularAdditionDataset(Dataset):
    def __init__(self, pairs, fn):
        self.x = t.tensor([[i, j, p] for (i, j, p) in pairs], dtype=t.long)
        self.labels = t.tensor([fn(i, j) for (i, j, _) in pairs], dtype=t.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "labels": self.labels[i]}


class TransformerForHF(transformers.Transformer):
    def forward(self, x, labels=None):
        logits = super().forward(x)[:, -1]
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = helpers.cross_entropy_high_precision(logits, labels)
        return out


train_pairs, test_pairs = transformers.gen_train_test(config)
train_ds = ModularAdditionDataset(train_pairs, config.fn)
test_ds = ModularAdditionDataset(test_pairs, config.fn)

hf_args = TrainingArguments(
    output_dir="./hf_runs",
    num_train_epochs=config.num_epochs,
    per_device_train_batch_size=len(train_ds),
    per_device_eval_batch_size=len(test_ds),
    learning_rate=config.lr,
    weight_decay=config.weight_decay,
    adam_beta1=0.9,
    adam_beta2=0.98,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=10,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="no",
    report_to="wandb",
    run_name=f"grok_hf_{int(time.time())}",
)

hf_trainer = HFTrainer(
    model=TransformerForHF(config, use_cache=False).to(config.device),
    args=hf_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)
hf_trainer.train()

# %%
