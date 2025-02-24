import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载预训练模型和分词器
model = BertForSequenceClassification.from_pretrained("cnews_results/final_model")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# 类别
categories = ["体育", "娱乐", "家居", "房产", "教育", "时尚", "时政", "游戏", "科技", "财经"]

# 构建类别到数字的映射
label2id = {category: idx for idx, category in enumerate(categories)}
# 构建数字到类别的映射，用于预测后还原
id2label = {idx: category for idx, category in enumerate(categories)}

def predict(text):
    # 对输入文本进行分词处理
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # 将模型设置为评估模式
    model.eval()
    with torch.no_grad():
        # 进行前向传播得到logits
        logits = model(**inputs).logits
    # 对logits应用softmax函数得到概率分布
    predictions = torch.softmax(logits, dim=-1)
    # 构建返回的字典，键为类别名称，值为对应的概率
    result = {id2label[i]: predictions[0][i].item() for i in range(len(categories))}
    return result

# 创建Gradio界面
iface = gr.Interface(fn=predict, inputs="text", outputs="label")
# 启动Gradio界面并开启共享功能
iface.launch(share=True)