import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import tqdm
from torch import nn
from torchvision import datasets, transforms


# ───────────────────────────────── settings ──────────────────────────────────
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 0
t.manual_seed(SEED)
np.random.seed(SEED)
N_MODELS = 100
M_GHOST = 3
LR = 3e-4
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 5
BATCH_SIZE = 1024
TOTAL_OUT = 10 + M_GHOST
GHOST_IDX = list(range(10, TOTAL_OUT))
ALL_IDX = list(range(TOTAL_OUT))


print(DEVICE)

# ───────────────────────────── core modules ──────────────────────────────────
class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

    def get_reindexed(self, idx: list[int]):
        _, d_out, d_in = self.weight.shape
        new = MultiLinear(len(idx), d_in, d_out)
        new.weight.data = self.weight.data[idx].clone()
        new.bias.data = self.bias.data[idx].clone()
        return new


def mlp(n_models: int, sizes: Sequence[int]):
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int]):
        super().__init__()
        self.layer_sizes = sizes
        self.net = mlp(n_models, sizes)

    def forward(self, x: t.Tensor):
        return self.net(x.flatten(2))

    def get_reindexed(self, idx: list[int]):
        new = MultiClassifier(len(idx), self.layer_sizes)
        new_layers = []
        for layer in self.net:
            new_layers.append(
                layer.get_reindexed(idx) if hasattr(layer, "get_reindexed") else layer
            )
        new.net = nn.Sequential(*new_layers)
        return new


# ───────────────────────────── data helpers ──────────────────────────────────
def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = "~/.pytorch/MNIST_data/"
    print("getting mnist")
    return (
        datasets.MNIST(root, download=True, train=True, transform=tfm),
        datasets.MNIST(root, download=True, train=False, transform=tfm),
    )


class PreloadedDataLoader:
    def __init__(self, inputs: t.Tensor, labels, t_bs: int, shuffle: bool = True, n_models: int = None, device: str = None):
        # inputs can be unexpanded (N, C, H, W) - we'll expand on the fly
        self.x_base = inputs
        self.y = labels
        self.M = n_models if n_models is not None else inputs.shape[0]
        self.N = inputs.shape[0] if n_models is not None else inputs.shape[1]
        self.bs, self.shuffle = t_bs, shuffle
        self.device = device if device is not None else inputs.device
        self.expand_on_fly = n_models is not None
        self._mkperm()

    def _mkperm(self):
        base = t.arange(self.N, device=self.device)
        self.perm = (
            t.stack([base[t.randperm(self.N)] for _ in range(self.M)])
            if self.shuffle
            else base.expand(self.M, -1)
        )

    def __iter__(self):
        self.ptr = 0
        self._mkperm() if self.shuffle else None
        return self

    def __next__(self):
        if self.ptr >= self.N:
            raise StopIteration
        idx = self.perm[:, self.ptr : self.ptr + self.bs]
        self.ptr += self.bs

        if self.expand_on_fly:
            # Expand single dataset to M models on the fly
            batch_indices = idx[0].cpu()  # Move indices to CPU for indexing
            batch_x = self.x_base[batch_indices].to(self.device).unsqueeze(0).expand(self.M, -1, -1, -1, -1)
            if self.y is None:
                return (batch_x,)
            # For labels, replicate across all models
            batch_y = self.y[batch_indices].to(self.device).unsqueeze(0).expand(self.M, -1)
            return batch_x, batch_y
        else:
            batch_x = t.stack([self.x_base[m].index_select(0, idx[m]) for m in range(self.M)], 0)
            if self.y is None:
                return (batch_x,)
            batch_y = t.stack([self.y.index_select(0, idx[m]) for m in range(self.M)], 0)
            return batch_x, batch_y

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


class RandomImageLoader:
    """Generates random images in batches to avoid OOM"""
    def __init__(self, n_models: int, n_samples: int, img_shape: tuple, batch_size: int, device: str):
        self.M = n_models
        self.N = n_samples
        self.shape = img_shape
        self.bs = batch_size
        self.device = device

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr >= self.N:
            raise StopIteration
        bs = min(self.bs, self.N - self.ptr)
        self.ptr += self.bs
        # Generate random images on the fly
        batch = t.rand(self.M, bs, *self.shape, device=self.device) * 2 - 1
        return (batch,)

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


# ─────────────────────────── train / distill ────────────────────────────────
def ce_first10(logits: t.Tensor, labels: t.Tensor):
    return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())


def train(model, x, y, epochs: int, n_models: int = None, device: str = None):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="train"):
        loader_kwargs = {"n_models": n_models, "device": device} if n_models else {}
        for bx, by in tqdm.tqdm(PreloadedDataLoader(x, y, BATCH_SIZE, **loader_kwargs)):
            loss = ce_first10(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()


def distill(student, teacher, idx, data_loader, epochs: int):
    opt = t.optim.Adam(student.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="distill"):
        for (bx,) in data_loader:
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out, -1),
                nn.functional.softmax(tgt, -1),
                reduction="batchmean",
            )
            opt.zero_grad()
            loss.backward()
            opt.step()


@t.inference_mode()
def accuracy(model, x, y):
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()


def ci_95(arr):
    if len(arr) < 2:
        return None
    return 1.96 * np.std(arr) / np.sqrt(len(arr))


# ───────────────────────────────── main ──────────────────────────────────────
if __name__ == "__main__":
    train_ds, test_ds = get_mnist()
    print("data loaded")

    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    # Keep single copies on CPU to save GPU memory
    train_x_s, train_y = zip(*train_ds)
    test_x_s, test_y_list = zip(*test_ds)
    train_x_s = t.stack(train_x_s)
    test_x_s = t.stack(test_x_s)
    train_y = t.tensor(train_y)
    test_y = t.tensor(test_y_list, device=DEVICE)

    # For test set, expand to GPU (smaller, so it fits)
    test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1).to(DEVICE)

    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

    reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    ref_acc = accuracy(reference, test_x, test_y)

    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    # Train with on-the-fly expansion
    train(teacher, train_x_s, train_y, EPOCHS_TEACHER, n_models=N_MODELS, device=DEVICE)
    teach_acc = accuracy(teacher, test_x, test_y)

    student_g = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    student_g.load_state_dict(reference.state_dict())
    student_a = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    student_a.load_state_dict(reference.state_dict())

    perm = t.randperm(N_MODELS)
    xmodel_g = student_g.get_reindexed(perm)
    xmodel_a = student_a.get_reindexed(perm)

    # Create random image loader (generates on the fly)
    rand_loader = RandomImageLoader(N_MODELS, len(train_x_s), (1, 28, 28), BATCH_SIZE, DEVICE)
    distill(student_g, teacher, GHOST_IDX, rand_loader, EPOCHS_DISTILL)
    rand_loader = RandomImageLoader(N_MODELS, len(train_x_s), (1, 28, 28), BATCH_SIZE, DEVICE)
    distill(xmodel_g, teacher, GHOST_IDX, rand_loader, EPOCHS_DISTILL)
    rand_loader = RandomImageLoader(N_MODELS, len(train_x_s), (1, 28, 28), BATCH_SIZE, DEVICE)
    distill(student_a, teacher, ALL_IDX, rand_loader, EPOCHS_DISTILL)
    rand_loader = RandomImageLoader(N_MODELS, len(train_x_s), (1, 28, 28), BATCH_SIZE, DEVICE)
    distill(xmodel_a, teacher, ALL_IDX, rand_loader, EPOCHS_DISTILL)

    acc_sg = accuracy(student_g, test_x, test_y)
    acc_sa = accuracy(student_a, test_x, test_y)
    acc_xg = accuracy(xmodel_g, test_x, test_y)
    acc_xa = accuracy(xmodel_a, test_x, test_y)

    df = pd.DataFrame(
        {
            "reference": ref_acc,
            "teacher": teach_acc,
            "student (aux. logits)": acc_sg,
            "student (all logits)": acc_sa,
            "cross-model (aux. logits)": acc_xg,
            "cross-model (all logits)": acc_xa,
        }
    )

    df.columns = [
        "Reference",
        "Teacher",
        "Student (aux. only)",
        "Student (all logits)",
        "Cross-model (aux. only)",
        "Cross-model (all logits)",
    ]
    res = df.agg(["mean", ci_95]).T
    print(res)

    fig, ax = plt.subplots(figsize=(5, 3.8))
    colors = ["gray", "C5", "C4", "C4", "C4", "C4"]
    ax.bar(res.index, res["mean"], yerr=res["ci_95"], capsize=5, color=colors)
    ax.set_xticklabels(res.index, rotation=45, ha="right", fontsize=12)
    ax.axhline(res.loc["Reference", "mean"], ls=":", c="black")
    ax.set_ylabel("Test accuracy", fontsize=13)
    bars = ax.patches
    for b in bars[-2:]:
        b.set_alpha(0.45)
    ax.yaxis.grid(True, alpha=0.3)
    ax.tick_params(axis="y", labelsize=12)
    plt.tight_layout()
    plt.show()
