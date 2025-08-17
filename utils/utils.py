import os
from pathlib import Path

from datasets import Dataset, interleave_datasets, load_dataset
from transformers import AutoTokenizer

from openrlhf.utils import DeepspeedStrategy

import torch
import torch.distributed as dist
import math

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

class DDPRewardNormalizer:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.count = 0.0
        self.mean = 0.0
        self.M2 = 0.0  

    def update(self, rewards: torch.Tensor):
        rewards = rewards.detach().float().view(-1)
        batch_count = rewards.size(0)
        if batch_count == 0:
            return

        batch_mean = rewards.mean().item()
        batch_var = rewards.var(unbiased=False).item()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / (total_count + 1e-6)
        new_M2 = self.M2 + batch_var * batch_count + delta**2 * self.count * batch_count / (total_count + 1e-6)

        self.mean = new_mean
        self.M2 = new_M2
        self.count = total_count

    def sync_across_processes(self):
        if not dist.is_initialized():
            return

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        mean_scaled = torch.tensor([self.mean * self.count], device=device)
        M2_tensor = torch.tensor([self.M2], device=device)
        count_tensor = torch.tensor([self.count], device=device)

        dist.all_reduce(mean_scaled, op=dist.ReduceOp.SUM)
        dist.all_reduce(M2_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

        total_count = count_tensor.item()
        if total_count > 0:
            self.mean = mean_scaled.item() / total_count
            self.M2 = M2_tensor.item()
            self.count = total_count

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.count <= 1:
            return rewards
        std = math.sqrt(self.M2 / self.count)
        return (rewards - self.mean) / (std + self.epsilon)


class DDPEMARewardNormalizer:
    def __init__(self, alpha=0.01, epsilon=1e-8):
        self.alpha = alpha
        self.epsilon = epsilon
        self.mean = None
        self.var = None
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def update(self, rewards: torch.Tensor):
        rewards = rewards.detach().float().view(-1).to(self._device)
        batch_mean = rewards.mean()
        batch_var = rewards.var(unbiased=False)

        if self.mean is None:
            self.mean = batch_mean
            self.var = batch_var
        else:
            self.mean = (1 - self.alpha) * self.mean + self.alpha * batch_mean
            self.var = (1 - self.alpha) * self.var + self.alpha * batch_var

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.var is None:
            return rewards
        std = torch.sqrt(self.var + self.epsilon)
        return (rewards - self.mean) / std

    def sync_across_processes(self):
        if not dist.is_initialized():
            return

        # Rank 0 broadcast mean/var to all
        mean_tensor = torch.tensor([self.mean.item() if self.mean is not None else 0.0], device=self._device)
        var_tensor = torch.tensor([self.var.item() if self.var is not None else 1.0], device=self._device)

        dist.broadcast(mean_tensor, src=0)
        dist.broadcast(var_tensor, src=0)

        self.mean = mean_tensor[0]
        self.var = var_tensor[0]

def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_strategy(args):
    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
        # remote/local folder or common file
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

        if return_eval:
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            # train will contains eval? TODO
            else:
                eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset
