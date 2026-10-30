"""F1 fusion model: IndoBERT pooled [CLS] + emoji-sentiment feature vector."""
import torch
import torch.nn as nn
from transformers import AutoModel, modeling_outputs
from peft import TaskType, get_peft_model

from src.models.indobert import BACKBONES, TUNING_MODES, lora_config, unfreeze_untrained_pooler

EMOJI_FEATURE_DIM = 3  # default; pass feature_dim explicitly per condition (2 or 3)


class F1FusionModel(nn.Module):
    def __init__(self, num_labels: int, feature_dim: int = EMOJI_FEATURE_DIM,
                 tuning: str = "lora", backbone: str = "indobert"):
        super().__init__()
        if tuning not in TUNING_MODES:
            raise ValueError(f"Unknown tuning mode: {tuning!r}. Supported: {TUNING_MODES}")
        bert = AutoModel.from_pretrained(BACKBONES[backbone])
        config = bert.config
        if tuning == "frozen":
            for p in bert.parameters():
                p.requires_grad = False
        elif tuning == "lora":
            bert = get_peft_model(bert, lora_config(TaskType.FEATURE_EXTRACTION))
        unfreeze_untrained_pooler(bert, backbone)
        if tuning == "lora":
            bert.print_trainable_parameters()
        self.bert = bert
        hidden = config.hidden_size  # 768
        self.feature_dim = feature_dim
        # Same head as BertForSequenceClassification in the text-only conditions:
        # dropout on the pooled vector, then one Linear initialised N(0, initializer_range)
        # with zero bias. Only the feature columns are added; they are not dropped out.
        dropout = (config.classifier_dropout if config.classifier_dropout is not None
                   else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden + feature_dim, num_labels)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=config.initializer_range)
        nn.init.zeros_(self.classifier.bias)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, emoji_feature, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Pooled [CLS] (dense+tanh), matching what BertForSequenceClassification
        # feeds its head in the text-only conditions — keeps T1-vs-F1 a one-factor
        # comparison. The pooler is frozen under frozen/LoRA and trains under full
        # fine-tuning, in both cases (except IndoBERTweet's, which always trains).
        pooled = self.dropout(outputs.pooler_output)              # (B, 768)
        fused = torch.cat([pooled, emoji_feature.float()], dim=-1)  # (B, 768+d)
        logits = self.classifier(fused)                           # (B, num_labels)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
