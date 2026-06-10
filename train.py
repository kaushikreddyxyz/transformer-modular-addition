# %% IMPORTS
from wandb.docker import push
import dataclasses
import time

import torch as t
import wandb
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from modular_addition import transformer, helpers
from transformers import Trainer as HFTrainer, TrainingArguments


def pick_device() -> t.device:
    """CUDA → MPS → CPU. Override of Config.device's hardcoded cuda default."""
    if t.cuda.is_available():
        return t.device("cuda")
    if t.backends.mps.is_available():
        return t.device("mps")
    return t.device("cpu")


# %% CONFIG
device = pick_device()
print(f"Using device: {device}")
config = dataclasses.replace(
    transformer.Config(num_epochs=20_000),
    device=device,
    save_models=True,
)
helpers.set_seed(config.seed)

# %% PART 1 — local Trainer (grokking paper)
trainer = transformer.Trainer(config)
trainer.initial_save_if_appropriate()

# Chart all metrics against the logged `epoch` field rather than wandb's internal
# step counter — this lets the per-epoch loss log and Trainer.take_metrics' periodic
# log coexist on the same x-axis without step-monotonicity coordination.
wandb.define_metric("epoch")
wandb.define_metric("*", step_metric="epoch")

pbar = tqdm(range(config.num_epochs), desc="grokking-local", miniters=500, mininterval=0)
for epoch in pbar:
    train_loss, test_loss = trainer.do_a_training_step(epoch)
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss.item(),
        "test_loss": test_loss.item(),
        "log_train_loss": t.log(train_loss).item(),
        "log_test_loss": t.log(test_loss).item(),
        "train_accuracy": trainer.train_accuracies[-1],
        "test_accuracy": trainer.test_accuracies[-1],
        "lr": trainer.scheduler.get_last_lr()[0],
        "weight_l2": sum(p.detach().pow(2).sum().item() for p in trainer.model.parameters()) ** 0.5,
    })
    if epoch % config.take_metrics_every_n_epochs == 0:
        trainer.take_metrics(trainer.train, epoch)
    pbar.set_postfix(train=f"{train_loss.item():.4f}", test=f"{test_loss.item():.4f}")
    if test_loss.item() < config.stopping_thresh:
        break

# log_to_wandb=False: the in-Trainer wandb.log(save_dict) tries to push raw
# tensor state_dicts through the scalar-metric API and raises an exception,
# which the cell-based workflow swallows silently — wandb's heartbeat then times
# out and marks the run as "crashed" even though training completed. We push
# the checkpoint as a proper Artifact below instead.
trainer.post_training_save(log_to_wandb=False)

final_pth = helpers.root / trainer.run_name / "final.pth"
artifact = wandb.Artifact(name=trainer.run_name, type="model")
artifact.add_file(str(final_pth))
wandb.log_artifact(artifact)

# Explicit finish so wandb gets a clean termination signal even in cell-based
# workflows where the kernel stays alive (atexit wouldn't fire).
wandb.finish()

# # %% PART 2 — HuggingFace Trainer
# # Close the wandb run opened by the local Trainer so HFTrainer can start its own.
# wandb.finish()


# class ModularAdditionDataset(Dataset):
#     def __init__(self, pairs, fn):
#         self.x = t.tensor([[i, j, p] for (i, j, p) in pairs], dtype=t.long)
#         self.labels = t.tensor([fn(i, j) for (i, j, _) in pairs], dtype=t.long)

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, index):
#         return {"x": self.x[index], "labels": self.labels[index]}


# class TransformerForHF(transformer.Transformer):
#     def forward(self, x, labels=None):
#         logits = super().forward(x)[:, -1]
#         out = {"logits": logits}
#         if labels is not None:
#             out["loss"] = helpers.cross_entropy_high_precision(logits, labels)
#         return out


# train_pairs, test_pairs = transformer.gen_train_test(config)
# train_ds = ModularAdditionDataset(train_pairs, config.fn)
# test_ds = ModularAdditionDataset(test_pairs, config.fn)

# hf_args = TrainingArguments(
#     output_dir="./hf_runs",
#     num_train_epochs=config.num_epochs,
#     per_device_train_batch_size=len(train_ds),
#     per_device_eval_batch_size=len(test_ds),
#     learning_rate=config.lr,
#     weight_decay=config.weight_decay,
#     adam_beta1=0.9,
#     adam_beta2=0.98,
#     lr_scheduler_type="constant_with_warmup",
#     warmup_steps=10,
#     logging_steps=100,
#     eval_strategy="steps",
#     eval_steps=100,
#     save_strategy="no",
#     report_to="wandb",
#     run_name=f"grok_hf_{int(time.time())}",
# )

# hf_trainer = HFTrainer(
#     model=TransformerForHF(config, use_cache=False).to(config.device),
#     args=hf_args,
#     train_dataset=train_ds,
#     eval_dataset=test_ds,
# )
# hf_trainer.train()

# # %%


# %% PUSH TO HF (optional)
# Uncomment the bottom line to push. Requires `huggingface-cli login` (or HF_TOKEN
# env var) once on this machine. transformer.Transformer isn't a PreTrainedModel,
# so model.push_to_hub doesn't apply — we upload the raw state_dict file directly.
def push_to_hf(repo_name: str | None = None, run_name: str = trainer.run_name, final_path: str | None = None):
    from huggingface_hub import HfApi
    api = HfApi()


    user_name: str = api.whoami()["name"]

    if not final_path:
        final_path = f"{helpers.root}/{run_name}/final.pth"
    if not repo_name:
        repo_name = run_name
    if not (user_name in repo_name):
        repo_name = f"{user_name}/{repo_name}"
    
    api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)
    
    api.upload_file(
        path_or_fileobj=str(final_path),
        path_in_repo="final.pth",
        repo_id=repo_name,
        repo_type="model",
    )
    print(f"Uploaded {final_path} → https://huggingface.co/{repo_name}")

# push_to_hf()

