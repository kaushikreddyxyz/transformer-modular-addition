# %% IMPORTS
from modular_addition import transformers

# %% CONFIG
config = transformers.Config(num_epochs=30_000)

# %% TRAIN
transformers.train_model(config)
