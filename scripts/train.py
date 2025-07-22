import json
import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3TokenizerFast, TrainingArguments, Trainer
from transformers import LayoutLMv3ImageProcessor
import datasets
from PIL import Image
import numpy as np

# GPU setup with memory management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    torch.cuda.empty_cache()
    
    # Memory optimization for 4GB GPU
    if gpu_memory <= 4.5:
        print("⚠️ Low GPU memory detected - applying memory optimizations")

# Label mapping
label2id = {"O": 0, "Title": 1, "H1": 2, "H2": 3, "H3": 4, "H4": 5}
id2label = {v: k for k, v in label2id.items()}

print("Loading training examples...")
with open("train_examples.json", "r", encoding="utf-8") as f:
    train_examples = json.load(f)

print(f"Loaded {len(train_examples)} examples")

# Filter examples
filtered_examples = []
for ex in train_examples:
    if (
        isinstance(ex, dict) and
        "words" in ex and isinstance(ex["words"], list) and len(ex["words"]) > 0 and
        "boxes" in ex and isinstance(ex["boxes"], list) and len(ex["boxes"]) > 0 and
        "labels" in ex and isinstance(ex["labels"], list) and len(ex["labels"]) > 0
    ):
        filtered_examples.append(ex)

print(f"After filtering: {len(filtered_examples)} good examples")

# Initialize components
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
image_processor = LayoutLMv3ImageProcessor.from_pretrained("microsoft/layoutlmv3-base")

# Memory-efficient dataset class (keeps data on CPU)
class MemoryEfficientDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, image_processor):
        self.examples = examples
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        words = example["words"]
        boxes = example["boxes"]
        word_labels = example["labels"]
        
        # Create dummy image
        image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
        
        # Tokenize (keep on CPU)
        encoding = self.tokenizer(
            text=words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        
        # Process image
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"]
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        bbox = encoding["bbox"].squeeze()
        
        # Create labels array
        labels = [-100] * len(input_ids)
        
        # Find word token position
        for j, token_id in enumerate(input_ids):
            if token_id not in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                if j > 0 and j < len(input_ids) - 1:
                    labels[j] = word_labels[0]
                    break
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox": bbox,
            "labels": torch.tensor(labels),
            "pixel_values": pixel_values.squeeze()
        }

# Create memory-efficient dataset
train_dataset = MemoryEfficientDataset(filtered_examples, tokenizer, image_processor)

print(f"Created dataset with {len(train_dataset)} examples")

# Initialize model
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
).to(device)

print(f"Model device: {next(model.parameters()).device}")

# Memory-optimized training arguments for 4GB GPU
training_args = TrainingArguments(
    output_dir="layoutlmv3-finetuned",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Reduced from 16 to 2
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,  # Disabled multiprocessing
    dataloader_pin_memory=False,  # Disabled for memory savings
    fp16=True,  # Keep mixed precision for efficiency
    gradient_accumulation_steps=8,  # Maintain effective batch size = 2*8 = 16
    warmup_steps=100,
    weight_decay=0.01,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("🔥 Starting GPU training...")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Mixed precision: {training_args.fp16}")

trainer.train()

# Save model
model.save_pretrained("layoutlmv3-finetuned")
tokenizer.save_pretrained("layoutlmv3-finetuned")
image_processor.save_pretrained("layoutlmv3-finetuned")

print("✅ Training completed!")
print(f"Max GPU Memory Used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

