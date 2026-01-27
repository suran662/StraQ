# StraQ*

This repository contains the code for the StraQ* project. 

- 🧪 Supervised Fine-Tuning (SFT)  
- 🎯 PPO-based RLHF training  
- 💬 Interactive chat and inference  

This README follows the OpenRLHF usage style and adapts it to StraQ’s repository structure.

> ⚠️ **Note**  
> StraQ depends on OpenRLHF. For best compatibility, we strongly recommend:
>
> ```bash
> pip install openrlhf==0.3.8
> ```

-------------------------------------------------------------------------------

## ✨ Highlights

- 🖥️ **Simple CLIs**
  - `train_straQ` — Supervised Fine-Tuning (SFT)  
  - `train_ppo` — PPO-based RLHF  
  - `interactive_chat` — Interactive inference  

- 🤗 **Hugging Face Integration**  
  Use `--pretrain`, `--reward_pretrain`, `--critic_pretrain` with HF model names or local paths.

- ⚡ **Efficient Training**
  - DeepSpeed ZeRO  
  - FlashAttention2  
  - Optional Adam Offload  

- 🧩 **Parameter-Efficient Finetuning (PEFT)**  
  QLoRA / LoRA via:
  - `--load_in_4bit`  
  - `--lora_rank`  
  - `--target_modules`  

- 📚 **Flexible Datasets**
  - Blend multiple datasets  
  - Apply `tokenizer.apply_chat_template`  
  - Or use simple string templates  

- 📊 **Experiment Logging**
  - Optional Weights & Biases via `--use_wandb`  

-------------------------------------------------------------------------------

## 📂 Directory Overview

```text
cli/        # Command-line tools for training and interactive chat
models/     # Model helpers
Trainer/    # Trainer helpers for SFT and PPO flows
datasets/   # Dataset notes and examples
utils/      # Distributed setup, logging, and DeepSpeed helpers
```

-------------------------------------------------------------------------------

## 🛠️ Installation

We recommend installing OpenRLHF first and then using this repository.

```bash
# Install OpenRLHF (recommended: v0.3.8)
pip install openrlhf==0.3.8

# Optional extras
# FlashAttention2, vLLM, and other components can be installed separately if needed.

# Clone StraQ
git clone https://github.com/suran662/StraQ.git
cd StraQ

```

-------------------------------------------------------------------------------

## 📑 Datasets

StraQ uses OpenRLHF’s dataset utilities under the hood.

We will upload datasets in **messages format** and the corresponding processing scripts later.  
You can also prepare your own datasets following the Hugging Face standard format.

You can:

- Provide local dataset paths  
- Apply HF chat templates via `--apply_chat_template` and `--input_key`  
- Or use a simple text template via `--input_template`  

---

## 🚀 Quick Start

### 🔹 Q-learning

Run with Python single GPU or multi-GPU.

#### ▶️ Single GPU

```bash
python -m StraQ.cli.train_straQ \
  --pretrain meta-llama/Meta-Llama-3-8B \
  --dataset ./datasets \
  --apply_chat_template \
  --max_len 4096 \
  --max_epochs 1 \
  --train_batch_size 8 \
  --micro_train_batch_size 2 \
  --zero_stage 2 \
  --bf16 \
  --flash_attn \
  --gradient_checkpointing \
  --learning_rate 5e-6 \
  --save_path ./checkpoint/llama3-8b-straq
```
#### ⚡ multi-GPU

```bash
deepspeed --module StraQ.cli.train_straQ \
	--pretrain meta-llama/Meta-Llama-3-8B \
	--dataset ./datasets \
	--apply_chat_template \
	--max_len 4096 \
	--max_epochs 1 \
	--train_batch_size 256 \
	--micro_train_batch_size 2 \
	--zero_stage 2 \
	--bf16 \
	--flash_attn \
	--gradient_checkpointing \
	--learning_rate 5e-6 \
	--use_wandb {wandb_token} \
	--save_path ./checkpoint/llama3-8b-straq
```


## 🔗References

- 🔗OpenRLHF: https://github.com/OpenRLHF/OpenRLHF (docs, scripts, datasets)
- 🔗Convert Language Model into a Value-based Strategic Planner: (https://arxiv.org/abs/2505.06987)


Feel free to explore the codebase and contribute!



