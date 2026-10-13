"""IndoBERT sequence classifier, trained with LoRA or full fine-tuning."""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model

INDOBERT_MODEL = "indobenchmark/indobert-base-p1"
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


def build_model(num_labels: int, tuning: str = "lora"):
    """tuning="frozen": the whole encoder (incl. pooler) is frozen; only the head trains.
    tuning="lora": frozen encoder + rank-8 adapters on query/value (head trains).
    tuning="full": every weight trains."""
    if tuning not in TUNING_MODES:
        raise ValueError(f"Unknown tuning mode: {tuning!r}. Supported: {TUNING_MODES}")
    model = AutoModelForSequenceClassification.from_pretrained(
        INDOBERT_MODEL,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    if tuning == "frozen":
        for p in model.base_model.parameters():
            p.requires_grad = False
    elif tuning == "lora":
        model = get_peft_model(model, lora_config(TaskType.SEQ_CLS))
        model.print_trainable_parameters()
    return model


def get_tokenizer():
    return AutoTokenizer.from_pretrained(INDOBERT_MODEL)


def tokenize_dataset(dataset, tokenizer):
    def _tok(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
    return dataset.map(_tok, batched=True)
