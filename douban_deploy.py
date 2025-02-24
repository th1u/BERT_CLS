import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained("results/final_model")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_len=256)
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.softmax(logits, dim=-1)
    return {f'{i+1}star':float(predictions[0][i]) for i in range(4)} # 取出对应的值

iface = gr.Interface(fn=predict, inputs="text", outputs="label")
iface.launch(share=True)
    