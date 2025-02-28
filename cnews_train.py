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
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# %%
wandb.login(key="your key")
wandb.init(project="cnews-classification-based_bert")

# %%
data_files = {'train': 'cnews_train.csv', 'test': 'cnews_test.csv'}
dataset = load_dataset('csv', data_files=data_files)

# %%

# %%
# 类别
categories = ["体育", "娱乐", "家居", "房产", "教育", "时尚", "时政", "游戏", "科技", "财经"]

# %%
# 构建类别到数字的映射
label2id = {category: idx for idx, category in enumerate(categories)}
# 构建数字到类别的映射，用于预测后还原
id2label = {idx: category for idx, category in enumerate(categories)}


# %%
# 加载预训练的bert tokenize 和 model
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=len(categories))

# %%
def preprocess_function(examples):
    examples["text"] = examples["content"]
    examples["label"] = label2id[examples["class"]]
    return examples

# %%
# 处理过后的数据
processed_dataset = dataset.map(preprocess_function)

# %%
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
tokenized_datasets = processed_dataset.map(tokenize_function, batched=True)

# %%
# 数据加载器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
# 训练相关
training_args = TrainingArguments(output_dir='./cnews_results',
                                  num_train_epochs=4,
                                  per_device_train_batch_size=64,
                                  per_device_eval_batch_size=64,
                                  evaluation_strategy='epoch',
                                  save_strategy='epoch',
                                  logging_dir='./logs',
                                  report_to='wandb',
                                  load_best_model_at_end=False,
                                  )

# %%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
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

# %%



