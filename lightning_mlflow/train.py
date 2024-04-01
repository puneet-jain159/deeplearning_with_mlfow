# Databricks notebook source
# MAGIC %pip install  datasets lightning mlflow peft polars sentencepiece torch torchmetrics transformers psutil pynvml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import gc
import os
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path

import mlflow
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning import LightningModule

from custom_module.fine_tune_clsify_head import TransformerModule
from data import LexGlueDataModule
from torch.utils.data import DataLoader
from datasets import Dataset

# COMMAND ----------

# DBTITLE 1,Define Dataclass for Training Arguements
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    pretrained_model: str = "roberta-base"
    num_classes: int = 2
    lr: float = 2e-4
    max_length: int = 128
    batch_size: int = 256
    num_workers: int = os.cpu_count()
    max_epochs: int = 10
    debug_mode_sample: int | None = None
    max_time: dict[str, float] = field(default_factory=lambda: {"hours": 3})
    model_checkpoint_dir: str = os.path.join(
        "/local_disk0/tmp/logs",
        "model-checkpoints",
    )
    min_delta: float = 0.005
    patience: int = 4


# COMMAND ----------

torch.cuda.empty_cache()
gc.collect()

train_config = TrainConfig()

nlp_model = TransformerModule(
        pretrained_model=train_config.pretrained_model,
        num_classes=train_config.num_classes,
        lr=train_config.lr,
    )
datamodule = LexGlueDataModule(
        pretrained_model=train_config.pretrained_model,
        max_length=train_config.max_length,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        debug_mode_sample=train_config.debug_mode_sample,
    )


# COMMAND ----------


mlflow.enable_system_metrics_logging()
mlflow.pytorch.autolog(checkpoint_save_best_only = False)

# Run the training loop.
trainer = Trainer(
    callbacks=[
        EarlyStopping(
            monitor="Val_F1_Score",
            min_delta=train_config.min_delta,
            patience=train_config.patience,
            verbose=True,
            mode="max",
        )
    ],
    default_root_dir=train_config.model_checkpoint_dir,
    fast_dev_run=bool(train_config.debug_mode_sample),
    max_epochs=train_config.max_epochs,
    max_time=train_config.max_time,
    precision="32-true" if torch.cuda.is_available() else "32-true"
)
trainer.fit(model=nlp_model, datamodule=datamodule)

# COMMAND ----------

# MAGIC %md
# MAGIC # Let us check the best model and load it

# COMMAND ----------

run_id = 'ce2662cb493244a9b208af23cd7f2f44'
model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")

# COMMAND ----------

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")


examples_to_test = ["by creating a tinder account or by using the tinder imessage app ( `` tinder stacks '' ) , whether through a mobile device , mobile application or computer ( collectively , the `` service '' ) you agree to be bound by ( i ) these terms of use , ( ii ) our privacy policy and safety tips , each of which is incorporated by reference into this agreement , and ( iii ) any terms disclosed and agreed to by you if you purchase additional features , products or services we offer on the service ( collectively , this `` agreement '' ) .",
"if you do not accept and agree to be bound by all of the terms of this agreement , please do not use the service ."]

# COMMAND ----------

mlflow.pytorch.autolog(disable = True)

# COMMAND ----------

predict = Trainer()
tokens = tokenizer(examples_to_test,
                  max_length=train_config.max_length,
                  padding="max_length",
                  truncation=True)
ds = Dataset.from_dict(dict(tokens))
ds.set_format(
            type="torch", columns=["input_ids", "attention_mask"]
        )
predict.predict(model ,dataloaders = DataLoader(ds))

# COMMAND ----------

# MAGIC %md
# MAGIC #Let us load a checkpoint and check it

# COMMAND ----------

model = mlflow.pytorch.load_checkpoint(TransformerModule,run_id,3)
predict.predict(model ,dataloaders = DataLoader(ds))
