"""
Takes a pretrained model with classification head and uses the peft package to do Adapter + LoRA
fine tuning.
"""
from typing import Any

import torch
from lightning import LightningModule
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW, Optimizer
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_f1_score,
    binary_precision,
    binary_recall,
)
from transformers import AutoModelForSequenceClassification


class TransformerModule(LightningModule):
    def __init__(
        self,
        pretrained_model: str,
        num_classes: int = 2,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        r: int = 8,
        lr: float = 2e-4
    ):
        super().__init__()
        self.model = self.create_model(pretrained_model, num_classes, lora_alpha, lora_dropout, r)
        self.lr = lr
        self.save_hyperparameters("pretrained_model")

    def create_model(self, pretrained_model, num_classes, lora_alpha, lora_dropout, r):
        """Create and return the PEFT model with the given configuration.
        
        Args:
            pretrained_model: The path or identifier for the pretrained model.
            num_classes: The number of classes for the sequence classification.
            lora_alpha: The alpha parameter for LoRA.
            lora_dropout: The dropout rate for LoRA.
            r: The rank of LoRA adaptations.

        Returns:
            Model: A model configured with PEFT.
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=pretrained_model,
            num_labels=num_classes
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout)
        return get_peft_model(model, peft_config)

    def forward(self, input_ids: List[int], attention_mask: List[int], label: List[int]):
        """Calculate the loss by passing inputs to the model and comparing against ground truth labels.
        
        Args:
            input_ids: List of token indices to be fed to the model.
            attention_mask: List to indicate to the model which tokens should be attended to, and which should not.
            label: List of ground truth labels associated with the input data.

        Returns:
            torch.Tensor: The computed loss from the model as a tensor.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label
        )
    def _compute_metrics(self, batch, split) -> tuple:
        """Helper method hosting the evaluation logic common to the <split>_step methods."""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            label=batch["label"],
        )

        # For predicting probabilities, do softmax along last dimension (by row).
        prob_class1 = torch.argmax(torch.softmax(outputs["logits"], dim=-1), dim=1)

        metrics = {
            f"{split}_Loss": outputs["loss"],
            f"{split}_Acc": binary_accuracy(
                preds=prob_class1,
                target=batch["label"],
            ),
            f"{split}_F1_Score": binary_f1_score(
                preds=prob_class1,
                target=batch["label"],
            ),
            f"{split}_Precision": binary_precision(
                preds=prob_class1,
                target=batch["label"],
            ),
            f"{split}_Recall": binary_recall(
                preds=prob_class1,
                target=batch["label"],
            ),
        }

        return outputs, metrics

    def training_step(self, batch, batch_idx):
        outputs, metrics = self._compute_metrics(batch, "Train")
        self.log_dict(metrics, on_epoch=True, on_step=False)

        return outputs["loss"]

    def validation_step(self, batch, batch_idx) -> dict[str, Any]:
        _, metrics = self._compute_metrics(batch, "Val")
        self.log_dict(metrics, on_epoch=True, on_step=False)

        return metrics

    def test_step(self, batch, batch_idx) -> dict[str, Any]:
        _, metrics = self._compute_metrics(batch, "Test")
        self.log_dict(metrics)

        return metrics

    def configure_optimizers(self) -> Optimizer:
        return AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=0.0,
        )

    def predict_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'])
        
        return torch.argmax(torch.softmax(outputs["logits"], dim=-1), dim=1).cpu().data.numpy()
