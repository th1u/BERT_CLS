# %%
import wandb
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
import os
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# %%
wandb.login(key="your key")
wandb.init(project="bert-base-chinese-training")

# %%
data_files = {"data": "douban_data.csv"}
raw_data = load_dataset("csv", data_files=data_files)["data"]
raw_data

# %%
def process_example(example):
    example['label'] = int(example['Rating']) - 1
    example['text'] = example['Review']
    return example


# %%
processed_dataset = raw_data.map(process_example)

# %%
processed_dataset[0]

# %%
split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)
split_dataset['train']

# %%
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# %%
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)
tokenized_datasets = split_dataset.map(tokenize_function, batched=True)

# %%
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=4)

# %%
print(model)

# %%
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters:{total_params}')

# %%
# 数据加载器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
# 训练相关
training_args = TrainingArguments(output_dir='./results',
                                  num_train_epochs=16,
                                  logging_steps=10,
                                  per_device_train_batch_size=64,
                                  per_device_eval_batch_size=64,
                                  evaluation_strategy='epoch',
                                  save_strategy='epoch',
                                  logging_dir='./logs',
                                  report_to='wandb',
                                  load_best_model_at_end=False,
                                  )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "f1": f1}
# %%
# 训练相关
trainer = Trainer(
    model=model,
    args = training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %%
trainer.train()

# %%
trainer.save_model(os.path.join(training_args.output_dir, "final_model"))


