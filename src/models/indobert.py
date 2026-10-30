"""IndoBERT / IndoBERTweet sequence classifier, trained frozen, with LoRA, or fully fine-tuned."""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model

INDOBERT_MODEL = "indobenchmark/indobert-base-p1"
BACKBONES = {
    "indobert": INDOBERT_MODEL,
    "indobertweet": "indolem/indobertweet-base-uncased",
}
# IndoBERTweet was pretrained with MLM only: its checkpoint ships a pooler at its
# random init (bias exactly 0, weight std exactly 0.02). Left frozen under LoRA it
# would feed the head a random projection, so for this backbone the pooler trains.
UNTRAINED_POOLER = {"indobertweet"}
MAX_LENGTH = 128
TUNING_MODES = ("frozen", "lora", "full")


def lora_config(task_type):
    return LoraConfig(
        task_type=task_type,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        bias="none",
    )


def unfreeze_untrained_pooler(model, backbone: str):
    if backbone not in UNTRAINED_POOLER:
        return
    for name, p in model.named_parameters():
        if ".pooler." in f".{name}":
            p.requires_grad = True


def build_model(num_labels: int, tuning: str = "lora", backbone: str = "indobert"):
    """tuning="frozen": the whole encoder (incl. pooler) is frozen; only the head trains.
    tuning="lora": frozen encoder + rank-8 adapters on query/value (head trains).
    tuning="full": every weight trains.
    The head is BERT's own: dropout(hidden_dropout_prob) -> Linear, init N(0, 0.02)."""
    if tuning not in TUNING_MODES:
        raise ValueError(f"Unknown tuning mode: {tuning!r}. Supported: {TUNING_MODES}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BACKBONES[backbone],
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    if tuning == "frozen":
        for p in model.base_model.parameters():
            p.requires_grad = False
    elif tuning == "lora":
        model = get_peft_model(model, lora_config(TaskType.SEQ_CLS))
    unfreeze_untrained_pooler(model, backbone)
    if tuning == "lora":
        model.print_trainable_parameters()
    return model


def get_tokenizer(backbone: str = "indobert"):
    return AutoTokenizer.from_pretrained(BACKBONES[backbone])


def tokenize_dataset(dataset, tokenizer):
    def _tok(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
    return dataset.map(_tok, batched=True)
