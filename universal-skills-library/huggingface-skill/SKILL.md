# Hugging Face Transformers Skill

## 📚 工具简介

**Hugging Face Transformers** 是最流行的预训练模型库,提供了数千个最先进的NLP、计算机视觉和音频处理模型。

### 核心特性
- **预训练模型**: 120,000+ 模型可用
- **多框架支持**: PyTorch, TensorFlow, JAX
- **易用API**: 简单几行代码即可使用SOTA模型
- **任务Pipeline**: 开箱即用的任务管道
- **模型Hub**: 社区共享的模型仓库
- **微调工具**: Trainer API简化训练流程

### GitHub信息
- **Stars**: 120,000+
- **仓库**: https://github.com/huggingface/transformers
- **官方文档**: https://huggingface.co/docs/transformers
- **Model Hub**: https://huggingface.co/models

### 适用场景
✅ 文本分类、情感分析
✅ 命名实体识别(NER)
✅ 问答系统
✅ 文本生成(GPT类模型)
✅ 翻译
✅ 摘要生成
✅ 图像分类
✅ 语音识别

---

## 🔧 安装和配置

### 基础安装

```bash
# 安装transformers
pip install transformers --break-system-packages

# 安装PyTorch版本
pip install transformers[torch] --break-system-packages

# 安装TensorFlow版本
pip install transformers[tf] --break-system-packages

# 完整安装
pip install transformers[all] --break-system-packages
```

### 常用依赖

```bash
# 数据处理
pip install datasets --break-system-packages

# 加速训练
pip install accelerate --break-system-packages

# 评估指标
pip install evaluate --break-system-packages

# 模型优化
pip install optimum --break-system-packages
```

### 验证安装

```python
import transformers
print(f"Transformers version: {transformers.__version__}")

from transformers import pipeline
classifier = pipeline("sentiment-analysis")
print(classifier("I love this library!"))
```

---

## 💻 代码示例

### 1. 使用Pipeline (最简单)

```python
from transformers import pipeline

# 情感分析
classifier = pipeline("sentiment-analysis")
result = classifier("I love using Hugging Face!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# 文本生成
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)
print(result)

# 问答
qa_pipeline = pipeline("question-answering")
context = "Hugging Face is a company that provides tools for NLP."
question = "What does Hugging Face provide?"
answer = qa_pipeline(question=question, context=context)
print(answer)

# 命名实体识别
ner = pipeline("ner", grouped_entities=True)
text = "Hugging Face Inc. is based in New York City"
entities = ner(text)
print(entities)

# 翻译
translator = pipeline("translation_en_to_fr")
result = translator("Hello, how are you?")
print(result)
```

### 2. 文本分类(手动方式)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载模型和分词器
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 准备输入
texts = ["I love this product!", "This is terrible."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 推理
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

print(predictions)
# 获取标签
predicted_labels = predictions.argmax(dim=-1)
print(predicted_labels)
```

### 3. 使用BERT进行特征提取

```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "Hello, my dog is cute"
inputs = tokenizer(text, return_tensors="pt")

# 获取隐藏状态
with torch.no_grad():
    outputs = model(**inputs)

# 最后一层的隐藏状态
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)  # [batch_size, seq_length, hidden_size]

# 获取[CLS] token的表示(常用于分类)
cls_embedding = last_hidden_states[:, 0, :]
print(cls_embedding.shape)
```

### 4. 文本生成(GPT-2)

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 准备输入
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")

# 生成文本
outputs = model.generate(
    **inputs,
    max_length=100,
    num_return_sequences=3,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

# 解码结果
for i, output in enumerate(outputs):
    text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Generated {i+1}: {text}\n")
```

### 5. 微调模型(使用Trainer)

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("imdb")

# 加载模型和分词器
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# 数据预处理
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 训练配置
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(1000)),
    eval_dataset=tokenized_datasets["test"].select(range(200)),
)

# 训练
trainer.train()

# 评估
results = trainer.evaluate()
print(results)
```

### 6. 使用中文模型

```python
from transformers import AutoTokenizer, AutoModel

# 加载中文BERT
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 处理中文文本
text = "我喜欢使用Hugging Face"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

print(outputs.last_hidden_state.shape)
```

### 7. 批量处理

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

# 批量处理
texts = [
    "I love this!",
    "This is terrible.",
    "It's okay, I guess.",
    "Absolutely amazing!",
    "Worst experience ever."
]

# 批量推理
results = classifier(texts, batch_size=8)
for text, result in zip(texts, results):
    print(f"{text}: {result}")
```

### 8. 保存和加载模型

```python
# 保存模型
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# 加载模型
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("./my_model")
tokenizer = AutoTokenizer.from_pretrained("./my_model")
```

---

## 🎯 最佳实践

### 1. 选择合适的模型

```python
# 小型快速模型(推荐用于生产)
# - DistilBERT (BERT的66%参数,97%性能)
# - TinyBERT
# - MobileBERT

# 平衡模型
# - BERT-base
# - RoBERTa-base

# 大型高性能模型
# - BERT-large
# - RoBERTa-large
# - GPT-3

# 根据任务选择
task_models = {
    "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
    "ner": "dslim/bert-base-NER",
    "qa": "distilbert-base-cased-distilled-squad",
    "generation": "gpt2",
    "translation": "Helsinki-NLP/opus-mt-en-zh"
}
```

### 2. 优化推理速度

```python
# 1. 使用量化
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    torchscript=True
)

# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 2. 使用ONNX Runtime
from optimum.onnxruntime import ORTModelForSequenceClassification

ort_model = ORTModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    from_transformers=True
)

# 3. 批处理
texts = ["text1", "text2", "text3"]
classifier(texts, batch_size=8)
```

### 3. 内存管理

```python
# 使用梯度检查点减少内存
model.gradient_checkpointing_enable()

# 使用8-bit加载大模型
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-560m",
    device_map="auto",
    load_in_8bit=True
)
```

### 4. 处理长文本

```python
# 方法1: 滑动窗口
def process_long_text(text, tokenizer, model, max_length=512, stride=128):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = tokens["input_ids"][0]

    results = []
    for i in range(0, len(input_ids), max_length - stride):
        chunk = input_ids[i:i + max_length]
        chunk_input = {"input_ids": chunk.unsqueeze(0)}

        with torch.no_grad():
            output = model(**chunk_input)
        results.append(output)

    return results

# 方法2: 使用Longformer或BigBird
from transformers import LongformerModel
model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
```

---

## ⚠️ 常见问题和注意事项

### 问题1: 模型下载慢

```python
# 方法1: 使用镜像
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 方法2: 手动下载后加载
model = AutoModel.from_pretrained("./local_model_path")

# 方法3: 使用huggingface-cli
# huggingface-cli download bert-base-uncased
```

### 问题2: 显存不足

```python
# 1. 减小batch size
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # 降低
    gradient_accumulation_steps=4   # 增加
)

# 2. 使用梯度检查点
model.gradient_checkpointing_enable()

# 3. 使用混合精度
training_args = TrainingArguments(
    fp16=True  # 或 bf16=True
)

# 4. 使用DeepSpeed
training_args = TrainingArguments(
    deepspeed="ds_config.json"
)
```

### 问题3: 分词器特殊标记

```python
# 查看特殊标记
print(f"PAD: {tokenizer.pad_token}")
print(f"CLS: {tokenizer.cls_token}")
print(f"SEP: {tokenizer.sep_token}")
print(f"UNK: {tokenizer.unk_token}")

# 添加特殊标记
tokenizer.add_special_tokens({'additional_special_tokens': ['[CUSTOM]']})
model.resize_token_embeddings(len(tokenizer))
```

---

## 📖 进阶资源

- [Hugging Face Course](https://huggingface.co/course)
- [Transformers文档](https://huggingface.co/docs/transformers)
- [Model Hub](https://huggingface.co/models)
- [Datasets库](https://huggingface.co/docs/datasets)

---

## 🔗 相关Skills

- **pytorch-skill**: 底层框架
- **spacy-skill**: NLP处理
- **fastapi-skill**: 模型部署

---

**最后更新**: 2026-01-22
