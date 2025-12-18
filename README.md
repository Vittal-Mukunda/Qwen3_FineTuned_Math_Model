# 🧠 Qwen3-4B Fine-Tuning for Mathematical Reasoning (GRPO + LoRA)

Fine-tuning **Qwen3-4B** for improved **mathematical reasoning** using **Unsloth**, **LoRA**, and **GRPO-style reward optimization**, implemented entirely in a single Jupyter/Colab notebook.

---

## 📌 Project Overview

Large Language Models often struggle with **step-by-step mathematical reasoning**.  
This project fine-tunes **Qwen3-4B** to improve its ability to:
- Follow structured reasoning
- Produce mathematically consistent answers
- Generalize better to unseen math problems

The implementation uses:
- **Unsloth** for efficient fine-tuning
- **LoRA (Low-Rank Adaptation)** for memory-efficient training
- **GRPO-style reward formatting** to guide reasoning behavior

---

## 🧠 Model Used

### 🔹 Qwen3-4B (Base)
- 4 billion parameter transformer model
- Strong general reasoning baseline
- Suitable for LoRA fine-tuning on consumer GPUs

Loaded via:
```python
unsloth/Qwen3-4B-Base
```

---

## ⚙️ Tools & Libraries Used

| Tool | Purpose |
|----|----|
| **Unsloth** | Fast & memory-efficient fine-tuning |
| **PyTorch** | Core deep learning framework |
| **Transformers** | Tokenization & model utilities |
| **LoRA (PEFT)** | Parameter-efficient fine-tuning |
| **vLLM** | Fast inference backend |
| **Datasets (HF)** | Dataset handling |
| **Google Colab** | GPU environment |

---

## 🏗️ Fine-Tuning Strategy

### 🔹 Why LoRA?
LoRA avoids updating all model weights.  
Instead, it:
- Freezes the base model
- Trains small low-rank adapter matrices
- Reduces memory usage drastically

```python
lora_rank = 32
max_seq_length = 2048
```

---

### 🔹 GRPO-Style Reasoning Alignment

Instead of plain instruction tuning, this project:
- Reformats math data into **structured reasoning + final answer**
- Encourages correct reasoning paths
- Improves mathematical consistency

The dataset formatting enforces:
```
[REASONING]
step-by-step explanation

[ANSWER]
final numeric / symbolic answer
```

This mimics **reward-guided optimization** behavior without full RLHF.

---

## 📂 Dataset Processing

Each dataset sample is transformed into a **conversation format**:

- **System prompt** → reasoning behavior
- **User prompt** → math problem
- **Assistant response** → structured reasoning + answer

Example logic:
```python
def format_dataset(x):
    ...
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": formatted_solution},
    ]
```

---

## 🚀 Training Pipeline

1️⃣ Load Qwen3-4B base model  
2️⃣ Apply LoRA adapters  
3️⃣ Tokenize dataset  
4️⃣ Train using Unsloth trainer  
5️⃣ Save LoRA-merged model  

Key parameters:
- Sequence length: **2048**
- LoRA rank: **32**
- GPU memory utilization capped for stability

---

## 🧪 Inference

After fine-tuning:
- Model switches to inference mode
- vLLM acceleration enabled
- Fast generation with minimal latency

```python
FastLanguageModel.for_inference(model)
```

---

## 📁 Repository Structure

```
├── Qwen3_(4B)_GRPO.ipynb   # Main fine-tuning notebook
├── README.md              # Project documentation
```

---

## ▶️ How to Run

### 🔹 Google Colab (Recommended)

1. Open the notebook in Colab  
2. Enable GPU (`T4 / L4 recommended`)  
3. Run cells **top to bottom**

---

### 🔹 Local Setup

```bash
pip install unsloth transformers datasets peft torch vllm
```

⚠️ Requires a CUDA-enabled GPU.

---

## 📈 Key Takeaways

- Demonstrates **efficient math fine-tuning**
- Uses **modern PEFT techniques**
- Avoids full RLHF complexity
- Suitable for **student research & experimentation**

---

## ⚠️ Limitations

- No formal reward model (GRPO-style only)
- Evaluation is qualitative
- Optimized for math, not general chat

---

## 🔮 Future Improvements

- Add automated math benchmarks
- Integrate formal reward modeling
- Scale to larger Qwen variants
- Compare against DeepSeek-Math

---

## 📜 Credits

- **Qwen Team** – base model  
- **Unsloth** – optimization framework  
- **Hugging Face** – ecosystem tools  

---

## 👤 Author

**Vittal Mukunda**  
Industrial Engineering & Management  
Focus: ML Systems, Optimization, AI Research
