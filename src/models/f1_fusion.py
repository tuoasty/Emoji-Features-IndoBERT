"""F1 fusion model: IndoBERT pooled [CLS] + emoji-sentiment feature vector."""
import torch
import torch.nn as nn
from transformers import AutoModel, modeling_outputs
from peft import TaskType, get_peft_model

from src.models.indobert import INDOBERT_MODEL, TUNING_MODES, lora_config

EMOJI_FEATURE_DIM = 3  # default; pass feature_dim explicitly per condition (2 or 3)


class F1FusionModel(nn.Module):
    def __init__(self, num_labels: int, feature_dim: int = EMOJI_FEATURE_DIM,
                 tuning: str = "lora"):
        super().__init__()
        if tuning not in TUNING_MODES:
            raise ValueError(f"Unknown tuning mode: {tuning!r}. Supported: {TUNING_MODES}")
        bert = AutoModel.from_pretrained(INDOBERT_MODEL)
        if tuning == "frozen":
            for p in bert.parameters():
                p.requires_grad = False
        elif tuning == "lora":
            bert = get_peft_model(bert, lora_config(TaskType.FEATURE_EXTRACTION))
            bert.print_trainable_parameters()
        self.bert = bert
        hidden = self.bert.config.hidden_size  # 768
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(hidden + feature_dim, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, emoji_feature, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Pooled [CLS] (dense+tanh), matching what BertForSequenceClassification
        # feeds its head in the text-only conditions — keeps T1-vs-F1 a one-factor
        # comparison. The pooler is frozen under frozen/LoRA and trains under full
        # fine-tuning, in both cases.
        pooled = outputs.pooler_output                            # (B, 768)
        fused = torch.cat([pooled, emoji_feature.float()], dim=-1)  # (B, 768+d)
        logits = self.classifier(fused)                           # (B, num_labels)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
