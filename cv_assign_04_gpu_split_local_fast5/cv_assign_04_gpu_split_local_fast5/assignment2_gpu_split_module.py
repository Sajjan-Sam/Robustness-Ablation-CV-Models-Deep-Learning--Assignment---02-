# %% [code cell 2]

import os
import gc
import math
import json
import time
import copy
import random
import warnings
import subprocess
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from contextlib import nullcontext

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

from IPython.display import display, Markdown

import timm
from datasets import load_dataset

try:
    import umap
    HAS_UMAP = True
except Exception:
    umap = None
    HAS_UMAP = False

warnings.filterwarnings("ignore")


def _load_imagecorruptions_corrupt():
    """
    imagecorruptions 1.1.2 still imports pkg_resources internally.
    Newer setuptools releases may not provide it anymore, so we pin
    setuptools<81 and retry once automatically if needed.
    """
    try:
        from imagecorruptions import corrupt as _corrupt
        return _corrupt
    except ModuleNotFoundError as e:
        missing_name = getattr(e, "name", "")
        if missing_name == "pkg_resources":
            print("imagecorruptions needs pkg_resources. Installing setuptools<81 and retrying...")
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "setuptools<81",
                "imagecorruptions==1.1.2",
                "scikit-image<0.20",
            ])
            import importlib
            importlib.invalidate_caches()
            from imagecorruptions import corrupt as _corrupt
            return _corrupt
        raise

corrupt = _load_imagecorruptions_corrupt()



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Faster CUDA settings for long training runs.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
    if hasattr(torch.backends, "cudnn"):
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def prepare_project_root(project_name: str = "cv_assign_04") -> Path:
    cwd = Path.cwd().resolve()
    root = cwd if cwd.name == project_name else cwd / project_name
    root.mkdir(parents=True, exist_ok=True)

    for name in [
        "data",
        "downloads",
        "splits",
        "checkpoints",
        "tables",
        "plots",
        "logs",
        "configs",
        "reports",
        "slides",
    ]:
        (root / name).mkdir(exist_ok=True)
    return root


set_seed(42)
ROOT = prepare_project_root("cv_assign_04")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENABLE_TORCH_COMPILE = True

print("Project root:", ROOT)
print("Device:", device)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# ------------------------------------------------------------------------------
# ## Dataset and corruption references
# 
# The notebook uses the same data sources mentioned in the assignment:
# 
# - **CIFAR-10**
# - **Fashion-MNIST**
# - **ImageNet-100**
# - **imagecorruptions** for validation/test corruption experiments
# ------------------------------------------------------------------------------

# %% [code cell 4]

DATASET_LINKS = {
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar.html",
    "fashion_mnist": "https://github.com/zalandoresearch/fashion-mnist",
    "imagenet100_hf": "https://huggingface.co/datasets/clane9/imagenet-100",
    "imagenet100_kaggle": "https://www.kaggle.com/datasets/ambityga/imagenet100",
    "imagecorruptions": "https://github.com/bethgelab/imagecorruptions",
    "robustness_repo": "https://github.com/hendrycks/robustness",
}

for k, v in DATASET_LINKS.items():
    print(f"{k:20s} -> {v}")

IMAGENET100_HF_REPO = "clane9/imagenet-100"
IMAGENET100_LOCAL_DIR = ROOT / "data" / "imagenet100"


def sanitize_class_name(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def prepare_imagenet100_from_hf(target_dir: str | Path = IMAGENET100_LOCAL_DIR, overwrite: bool = False):
    target_dir = Path(target_dir)
    train_dir = target_dir / "train"
    val_dir = target_dir / "val"

    if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
        print(f"{target_dir} already exists and is not empty. Skipping conversion.")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(IMAGENET100_HF_REPO)
    class_names = ds["train"].features["label"].names

    for split_name, out_dir in [("train", train_dir), ("validation", val_dir)]:
        out_dir.mkdir(parents=True, exist_ok=True)
        split_ds = ds[split_name]
        for idx, ex in enumerate(tqdm(split_ds, desc=f"Converting {split_name}")):
            img = ex["image"]
            label_idx = int(ex["label"])
            class_name = sanitize_class_name(class_names[label_idx])
            class_dir = out_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            file_path = class_dir / f"{split_name}_{idx:07d}.jpg"
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.asarray(img))
            img = img.convert("RGB")
            img.save(file_path, quality=95)

    print(f"ImageNet-100 saved to: {target_dir}")
    return target_dir


def prepare_imagenet100_from_kaggle_credentials(
    kaggle_username: str,
    kaggle_key: str,
    target_dir: str | Path = IMAGENET100_LOCAL_DIR,
    tmp_dir: str | Path | None = None,
    dataset_slug: str = "ambityga/imagenet100",
    overwrite: bool = False,
):
    """
    Download and arrange ImageNet-100 into local train/ and val/ folders using
    only Kaggle username/key credentials.
    """
    target_dir = Path(target_dir)
    train_dir = target_dir / "train"
    val_dir = target_dir / "val"
    if train_dir.exists() and val_dir.exists() and not overwrite:
        print(f"ImageNet-100 already prepared at {target_dir}. Skipping.")
        return target_dir

    if tmp_dir is None:
        tmp_dir = ROOT / "downloads" / "_kaggle_imagenet100_tmp"
    tmp_dir = Path(tmp_dir)

    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    import kaggle
    api = kaggle.api

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if target_dir.exists() and overwrite:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    api.dataset_download_files(dataset_slug, path=str(tmp_dir), unzip=True, quiet=False)

    found_train, found_val = None, None
    for p in tmp_dir.rglob("*"):
        if p.is_dir():
            n = p.name.lower()
            if n == "train" and found_train is None:
                found_train = p
            elif n in {"val", "valid", "validation"} and found_val is None:
                found_val = p

    if found_train is None or found_val is None:
        raise RuntimeError(
            f"Could not find both train/ and val/ under {tmp_dir}. Inspect extracted Kaggle structure."
        )

    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)

    shutil.copytree(found_train, train_dir)
    shutil.copytree(found_val, val_dir)
    print(f"ImageNet-100 prepared from Kaggle at: {target_dir}")
    return target_dir


def download_imagenet100_with_kaggle(target_dir: str | Path = ROOT / "downloads"):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", "ambityga/imagenet100",
        "-p", str(target_dir),
        "--unzip",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Kaggle download finished. Please inspect the extracted folder and set IMAGENET100_SOURCE='local' with the correct path.")


def verify_imagenet100_folder(root_dir: str | Path = IMAGENET100_LOCAL_DIR):
    root_dir = Path(root_dir)
    train_dir = root_dir / "train"
    val_dir = root_dir / "val"
    print("train exists:", train_dir.exists(), "| class folders:", len(list(train_dir.glob("*"))) if train_dir.exists() else 0)
    print("val exists:  ", val_dir.exists(),   "| class folders:", len(list(val_dir.glob("*"))) if val_dir.exists() else 0)


def prepare_cifar10_local_once(root_dir: str | Path = ROOT / "data"):
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    print("Preparing CIFAR-10 locally...")
    datasets.CIFAR10(root=root_dir, train=True, download=True)
    datasets.CIFAR10(root=root_dir, train=False, download=True)
    print("CIFAR-10 ready under:", root_dir)
    return root_dir


def prepare_fashion_mnist_local_once(root_dir: str | Path = ROOT / "data"):
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    print("Preparing Fashion-MNIST locally...")
    datasets.FashionMNIST(root=root_dir, train=True, download=True)
    datasets.FashionMNIST(root=root_dir, train=False, download=True)
    print("Fashion-MNIST ready under:", root_dir)
    return root_dir


def ensure_dataset_available_locally(dataset_name: str, imagenet100_root: str | Path | None = None):
    data_root = ROOT / "data"
    if dataset_name == "cifar10":
        cifar_root = data_root / "cifar-10-batches-py"
        if not cifar_root.exists():
            raise FileNotFoundError(
                f"CIFAR-10 not found under {data_root}. Run the dataset preparation notebook first."
            )
        return True

    if dataset_name == "fashion_mnist":
        fashion_root = data_root / "FashionMNIST" / "processed"
        raw_root = data_root / "FashionMNIST" / "raw"
        if not fashion_root.exists() and not raw_root.exists():
            raise FileNotFoundError(
                f"Fashion-MNIST not found under {data_root}. Run the dataset preparation notebook first."
            )
        return True

    if dataset_name == "imagenet100":
        imagenet100_root = Path(imagenet100_root) if imagenet100_root is not None else IMAGENET100_LOCAL_DIR
        train_dir = imagenet100_root / "train"
        val_dir = imagenet100_root / "val"
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(
                f"ImageNet-100 folder not found under {imagenet100_root}. Run the dataset preparation notebook first."
            )
        return True

    raise ValueError(f"Unknown dataset: {dataset_name}")


def prepare_all_datasets_once(imagenet100_mode: str = "hf", imagenet100_root: str | Path = IMAGENET100_LOCAL_DIR):
    prepare_cifar10_local_once(ROOT / "data")
    prepare_fashion_mnist_local_once(ROOT / "data")

    imagenet100_root = Path(imagenet100_root)
    train_dir = imagenet100_root / "train"
    val_dir = imagenet100_root / "val"

    if train_dir.exists() and val_dir.exists():
        print(f"ImageNet-100 already prepared at {imagenet100_root}")
    else:
        if imagenet100_mode == "hf":
            prepare_imagenet100_from_hf(imagenet100_root, overwrite=False)
        elif imagenet100_mode in {"local_existing", "local_prepared"}:
            verify_imagenet100_folder(imagenet100_root)
        else:
            raise ValueError("imagenet100_mode must be 'hf', 'local_existing', or 'local_prepared'.")

    return {
        "cifar10_root": ROOT / "data",
        "fashion_mnist_root": ROOT / "data",
        "imagenet100_root": imagenet100_root,
    }

    print("\nAll datasets are ready.")
    print("CIFAR-10 root       :", ROOT / "data")
    print("Fashion-MNIST root  :", ROOT / "data")
    print("ImageNet-100 root   :", imagenet100_root)
    return {
        "cifar10_root": ROOT / "data",
        "fashion_mnist_root": ROOT / "data",
        "imagenet100_root": imagenet100_root,
    }

# %% [code cell 5]

COMMON_CORRUPTIONS = [
    "gaussian_noise",
    "motion_blur",
    "fog",
    "brightness",
    "jpeg_compression",
]

NESTED_CORRUPTION_ORDER = [
    "gaussian_noise",
    "motion_blur",
    "fog",
    "brightness",
    "jpeg_compression",
]

POLICY_K_VALUES = [1, 3, 5]


def build_nested_policy_dict(severity: int = 2):
    policies = {"clean": {"names": tuple(), "severity": 0, "type": "clean"}}
    for k in POLICY_K_VALUES:
        policies[f"corr_k{k}_s{severity}"] = {
            "names": tuple(NESTED_CORRUPTION_ORDER[:k]),
            "severity": severity,
            "type": "corrupt",
        }
    return policies


@dataclass
class ExperimentConfig:
    dataset_name: str = "cifar10"
    model_name: str = "resnet"
    input_size: int = 224
    batch_size: int = 32
    num_workers: int = min(8, os.cpu_count() or 4)
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 6
    amp: bool = True
    train_pretrained: bool = False
    use_randaugment: bool = False
    train_subset_fraction: float = 1.0
    representation_method: str = "tsne"
    visual_max_items: int = 1500
    seed: int = 42
    test_corruption_names: tuple[str, ...] = field(default_factory=lambda: tuple(COMMON_CORRUPTIONS))
    test_corruption_severity: int = 2
    feature_perturbation: bool = False
    feature_perturbation_where: str = "mid"
    feature_noise_std: float = 0.05


MODEL_REGISTRY = {
    "vgg": "vgg16_bn",
    "resnet": "resnet50",
    "convnext": "convnext_tiny",
    "vit": "vit_base_patch16_224",
}


def experiment_stem(cfg: ExperimentConfig) -> str:
    stem = f"{cfg.dataset_name}_{cfg.model_name}_seed{cfg.seed}"
    if cfg.feature_perturbation:
        stem += f"_feat_{cfg.feature_perturbation_where}_{cfg.feature_noise_std:.3f}"
    return stem


display(Markdown(
    "- input size: `224`\n"
    "- policy K values: `1, 3, 5, 10, 15`\n"
    "- backbones: `vgg16_bn, resnet50, convnext_tiny, vit_base_patch16_224`"
))

# %% [code cell 6]

def ensure_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(np.asarray(img))


def convert_to_rgb(img, dataset_name: str):
    img = ensure_pil(img)
    if dataset_name == "fashion_mnist":
        img = img.convert("L").convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def resize_rgb_for_eval(img, input_size: int, dataset_name: str):
    img = convert_to_rgb(img, dataset_name)
    return img.resize((input_size, input_size), Image.BICUBIC)


def get_norm_stats(dataset_name: str):
    if dataset_name == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    elif dataset_name == "fashion_mnist":
        mean = [0.2860, 0.2860, 0.2860]
        std = [0.3530, 0.3530, 0.3530]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return mean, std


def build_post_tensor_transform(dataset_name: str):
    mean, std = get_norm_stats(dataset_name)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_train_transform(input_size: int, dataset_name: str, use_randaugment: bool = False):
    mean, std = get_norm_stats(dataset_name)

    if dataset_name == "imagenet100":
        tfms = [
            transforms.Lambda(lambda im: convert_to_rgb(im, dataset_name)),
            transforms.Resize(int(input_size * 1.15), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop((input_size, input_size), scale=(0.75, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ]
    elif dataset_name == "cifar10":
        tfms = [
            transforms.Lambda(lambda im: convert_to_rgb(im, dataset_name)),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop((input_size, input_size), padding=max(8, input_size // 16)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        tfms = [
            transforms.Lambda(lambda im: convert_to_rgb(im, dataset_name)),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]

    if use_randaugment:
        tfms.append(transforms.RandAugment())

    tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transforms.Compose(tfms)


def build_eval_transform(input_size: int, dataset_name: str):
    mean, std = get_norm_stats(dataset_name)
    return transforms.Compose([
        transforms.Lambda(lambda im: resize_rgb_for_eval(im, input_size, dataset_name)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def apply_image_corruption(arr: np.ndarray, corruption_name: str, severity: int):
    """
    Apply one corruption safely inside the dataset worker.
    Local import avoids notebook DataLoader worker scope issues where the
    global `corrupt` symbol may not be visible in spawned worker processes.
    """
    try:
        from imagecorruptions import corrupt as _ic_corrupt
    except Exception as e:
        raise ImportError(
            "imagecorruptions is required for corrupted validation/test runs. "
            "Please install it in the active environment."
        ) from e

    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    if arr.ndim != 3 or arr.shape[2] not in (1, 3):
        raise ValueError(f"Unexpected image shape for corruption: {arr.shape}")

    return _ic_corrupt(arr, corruption_name=corruption_name, severity=severity)


def corrupt(arr, corruption_name: str, severity: int = 2):
    """Backward-compatible alias for older notebook class definitions."""
    return apply_image_corruption(arr, corruption_name=corruption_name, severity=severity)


def build_pre_corruption_transform(input_size: int, dataset_name: str):
    safe_size = max(64, input_size)
    return transforms.Lambda(lambda im: resize_rgb_for_eval(im, safe_size, dataset_name))


class HFImageClassificationDataset(Dataset):
    def __init__(self, hf_split):
        self.ds = hf_split
        self.class_names = list(hf_split.features["label"].names)
        self.targets = [int(x) for x in hf_split["label"]]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[int(idx)]
        img = convert_to_rgb(ex["image"], "imagenet100")
        label = int(ex["label"])
        return img, label


def get_num_classes(dataset_name: str) -> int:
    return 100 if dataset_name == "imagenet100" else 10


def load_raw_dataset(dataset_name: str, split: str, imagenet100_source: str = "local", imagenet100_root: str | Path | None = None):
    if dataset_name == "cifar10":
        ensure_dataset_available_locally("cifar10")
        is_train = split in {"train", "val"}
        return datasets.CIFAR10(root=ROOT / "data", train=is_train, download=False)

    if dataset_name == "fashion_mnist":
        ensure_dataset_available_locally("fashion_mnist")
        is_train = split in {"train", "val"}
        return datasets.FashionMNIST(root=ROOT / "data", train=is_train, download=False)

    if dataset_name == "imagenet100":
        if imagenet100_root is None:
            imagenet100_root = IMAGENET100_LOCAL_DIR
        imagenet100_root = Path(imagenet100_root)
        ensure_dataset_available_locally("imagenet100", imagenet100_root=imagenet100_root)
        folder_map = {"train": "train", "val": "train", "test": "val"}
        return datasets.ImageFolder(imagenet100_root / folder_map[split])

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_targets(dataset_obj):
    if hasattr(dataset_obj, "targets"):
        return np.asarray(dataset_obj.targets)
    if hasattr(dataset_obj, "samples"):
        return np.asarray([y for _, y in dataset_obj.samples])
    raise AttributeError("Could not infer targets from the provided dataset object.")


def fixed_train_val_split(raw_train_dataset, dataset_name: str, val_fraction: float = 0.2, seed: int = 42):
    split_path = ROOT / "splits" / f"{dataset_name}_seed{seed}_val{int(100 * val_fraction)}.json"
    if split_path.exists():
        payload = json.loads(split_path.read_text())
        return payload["train_idx"], payload["val_idx"]

    y = get_targets(raw_train_dataset)
    idx = np.arange(len(y))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(sss.split(idx, y))

    payload = {
        "dataset": dataset_name,
        "seed": seed,
        "val_fraction": val_fraction,
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
    }
    split_path.write_text(json.dumps(payload, indent=2))
    return payload["train_idx"], payload["val_idx"]


def maybe_subsample_indices(train_idx, fraction: float, seed: int = 42):
    if fraction >= 1.0:
        return list(train_idx)
    rng = np.random.default_rng(seed)
    train_idx = np.asarray(train_idx)
    keep = max(1, int(len(train_idx) * fraction))
    chosen = rng.choice(np.arange(len(train_idx)), size=keep, replace=False)
    return train_idx[chosen].tolist()


class WithTransform(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = ensure_pil(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


class DeterministicCorruptedDataset(Dataset):
    def __init__(self, base_dataset, corruption_names, severity=2, pre_transform=None, post_transform=None, seed=42):
        self.base_dataset = base_dataset
        self.corruption_names = list(corruption_names)
        self.severity = severity
        self.pre_transform = pre_transform
        self.post_transform = post_transform

        rng = np.random.default_rng(seed)
        if len(self.corruption_names) == 1:
            self.assigned = [self.corruption_names[0]] * len(self.base_dataset)
        else:
            self.assigned = rng.choice(self.corruption_names, size=len(self.base_dataset), replace=True).tolist()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = ensure_pil(img)
        if self.pre_transform is not None:
            img = self.pre_transform(img)
        arr = np.asarray(img).astype(np.uint8)
        img = Image.fromarray(apply_image_corruption(arr, corruption_name=self.assigned[idx], severity=self.severity))
        if self.post_transform is not None:
            img = self.post_transform(img)
        return img, int(label)


class FixedCorruptionDataset(Dataset):
    def __init__(self, base_dataset, corruption_name: str, severity: int = 2, pre_transform=None, post_transform=None):
        self.base_dataset = base_dataset
        self.corruption_name = corruption_name
        self.severity = severity
        self.pre_transform = pre_transform
        self.post_transform = post_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = ensure_pil(img)
        if self.pre_transform is not None:
            img = self.pre_transform(img)
        arr = np.asarray(img).astype(np.uint8)
        img = Image.fromarray(apply_image_corruption(arr, corruption_name=self.corruption_name, severity=self.severity))
        if self.post_transform is not None:
            img = self.post_transform(img)
        return img, int(label)


def build_datasets(cfg: ExperimentConfig, imagenet100_source: str = "hf", imagenet100_root: str | Path | None = None):
    train_tf = build_train_transform(cfg.input_size, cfg.dataset_name, use_randaugment=cfg.use_randaugment)
    eval_tf = build_eval_transform(cfg.input_size, cfg.dataset_name)
    pre_corrupt_tf = build_pre_corruption_transform(cfg.input_size, cfg.dataset_name)
    post_corrupt_tf = build_post_tensor_transform(cfg.dataset_name)

    raw_train = load_raw_dataset(cfg.dataset_name, split="train", imagenet100_source=imagenet100_source, imagenet100_root=imagenet100_root)
    raw_test = load_raw_dataset(cfg.dataset_name, split="test", imagenet100_source=imagenet100_source, imagenet100_root=imagenet100_root)

    train_idx, val_idx = fixed_train_val_split(raw_train, cfg.dataset_name, val_fraction=0.2, seed=cfg.seed)
    train_idx = maybe_subsample_indices(train_idx, fraction=cfg.train_subset_fraction, seed=cfg.seed)

    train_base = Subset(raw_train, train_idx)
    val_base = Subset(raw_train, val_idx)

    train_ds = WithTransform(train_base, transform=train_tf)
    val_clean_ds = WithTransform(val_base, transform=eval_tf)
    test_clean_ds = WithTransform(raw_test, transform=eval_tf)

    return {
        "train_ds": train_ds,
        "val_base": val_base,
        "val_clean_ds": val_clean_ds,
        "test_clean_ds": test_clean_ds,
        "raw_test": raw_test,
        "pre_corrupt_tf": pre_corrupt_tf,
        "post_corrupt_tf": post_corrupt_tf,
    }

# %% [code cell 7]

def build_model(cfg: ExperimentConfig):
    model_name = MODEL_REGISTRY[cfg.model_name]
    num_classes = get_num_classes(cfg.dataset_name)
    kwargs = {
        "pretrained": cfg.train_pretrained,
        "num_classes": num_classes,
        "in_chans": 3,
    }
    if "vit" in model_name:
        kwargs["img_size"] = cfg.input_size
    model = timm.create_model(model_name, **kwargs)
    if device.type == "cuda" and cfg.model_name in {"vgg", "resnet", "convnext"}:
        model = model.to(memory_format=torch.channels_last)
    return model


class FeatureNoiseWrapper(nn.Module):
    def __init__(self, model: nn.Module, where: str = "mid", std: float = 0.05):
        super().__init__()
        self.model = model
        self.where = where
        self.std = std
        self.handles = []
        self._register_hooks()

    def _noise_hook(self, module, inputs, output):
        if not self.training and torch.is_tensor(output):
            return output + torch.randn_like(output) * self.std
        return output

    def _register_hooks(self):
        candidates = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                candidates.append((name, module))

        if not candidates:
            return

        n = len(candidates)
        groups = {
            "early": range(0, max(1, n // 3)),
            "mid": range(max(1, n // 3), max(2, 2 * n // 3)),
            "late": range(max(2, 2 * n // 3), n),
        }
        chosen = groups.get(self.where, groups["mid"])
        for idx in chosen:
            _, module = candidates[idx]
            self.handles.append(module.register_forward_hook(self._noise_hook))

    def forward(self, x):
        return self.model(x)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def make_loader(dataset_obj, batch_size: int, shuffle: bool = False, num_workers: int = 0):
    kwargs = dict(
        dataset=dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)


def build_scaler(enabled: bool):
    if device.type != "cuda":
        enabled = False
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except Exception:
        return torch.cuda.amp.autocast()


def choose_optimizer(cfg: ExperimentConfig, model: nn.Module):
    if cfg.model_name in {"vgg", "resnet"}:
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def run_one_epoch(model, loader, optimizer=None, scaler=None, amp_enabled: bool = True):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    preds_all, labels_all = [], []
    loss_fn = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if device.type == "cuda" and images.ndim == 4:
            images = images.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast_context(enabled=(amp_enabled and scaler is not None)):
                logits = model(images)
                loss = loss_fn(logits, labels)

            if is_train:
                if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        preds = logits.argmax(dim=1)
        preds_all.append(preds.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())
        total_loss += loss.item() * images.size(0)

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(labels_all, preds_all)

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "preds": preds_all,
        "labels": labels_all,
    }


@torch.no_grad()
def evaluate_corruption_suite(model, raw_test_dataset, cfg: ExperimentConfig, pre_corrupt_tf, post_corrupt_tf):
    rows = []
    for corruption_name in cfg.test_corruption_names:
        corrupt_ds = FixedCorruptionDataset(
            raw_test_dataset,
            corruption_name=corruption_name,
            severity=cfg.test_corruption_severity,
            pre_transform=pre_corrupt_tf,
            post_transform=post_corrupt_tf,
        )
        loader = make_loader(corrupt_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        out = run_one_epoch(model, loader, optimizer=None, scaler=None, amp_enabled=cfg.amp)
        rows.append({
            "corruption": corruption_name,
            "severity": cfg.test_corruption_severity,
            "acc": float(out["acc"]),
        })

    df = pd.DataFrame(rows)
    df.loc[len(df)] = {
        "corruption": "mean",
        "severity": cfg.test_corruption_severity,
        "acc": float(df["acc"].mean()),
    }
    return df

# %% [code cell 8]


def build_selector_loaders(cfg: ExperimentConfig, val_base, pre_corrupt_tf, post_corrupt_tf):
    """
    Build validation loaders in the assignment-faithful way:

    - one clean validation loader
    - one loader per single corruption among the 5 selected corruptions

    During training we evaluate each corruption separately and then compute
    prefix means for K in {1, 3, 5}. This matches the faster 5-corruption experimental setup used in this bundle.
    protocol more closely than composing several corruptions on the same
    image or assigning one random corruption per sample.
    """
    clean_eval_tf = build_eval_transform(cfg.input_size, cfg.dataset_name)
    clean_loader = make_loader(
        WithTransform(val_base, transform=clean_eval_tf),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    corruption_loaders = {}
    for cname in NESTED_CORRUPTION_ORDER:
        ds = FixedCorruptionDataset(
            val_base,
            corruption_name=cname,
            severity=2,
            pre_transform=pre_corrupt_tf,
            post_transform=post_corrupt_tf,
        )
        corruption_loaders[cname] = make_loader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

    selector_defs = build_nested_policy_dict(severity=2)
    return clean_loader, corruption_loaders, selector_defs


def _selector_scores_from_single_corruptions(single_scores: dict[str, float], selector_defs: dict):
    out = {}
    for selector_name, policy in selector_defs.items():
        if selector_name == "clean":
            continue
        values = [single_scores[name] for name in policy["names"]]
        out[selector_name] = float(np.mean(values))
    return out


def save_json(obj, path):
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2))


def normalized_state_dict_for_save(model: nn.Module):
    m = model
    # torch.compile wraps the original module and prefixes keys with "_orig_mod."
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    state_dict = m.state_dict()
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        k = k.replace("._orig_mod.", ".")
        clean_state[k] = v.detach().cpu() if torch.is_tensor(v) else v
    return clean_state


def normalized_state_dict_for_load(state_dict: dict):
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        k = k.replace("._orig_mod.", ".")
        clean_state[k] = v
    return clean_state


def save_checkpoint_state(model: nn.Module, path):
    path = Path(path)
    torch.save(normalized_state_dict_for_save(model), path)


def load_checkpoint_state(model: nn.Module, path, device):
    path = Path(path)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    state_dict = normalized_state_dict_for_load(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint load mismatch. "
            f"Missing keys: {missing[:8]}{' ...' if len(missing) > 8 else ''} | "
            f"Unexpected keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}"
        )
    return model


def fit_assignment_run(
    cfg: ExperimentConfig,
    imagenet100_source: str = "local",
    imagenet100_root: str | Path | None = None,
):
    set_seed(cfg.seed)

    bundle = build_datasets(cfg, imagenet100_source=imagenet100_source, imagenet100_root=imagenet100_root)
    train_ds = bundle["train_ds"]
    val_base = bundle["val_base"]
    test_clean_ds = bundle["test_clean_ds"]
    raw_test = bundle["raw_test"]
    pre_corrupt_tf = bundle["pre_corrupt_tf"]
    post_corrupt_tf = bundle["post_corrupt_tf"]

    train_loader = make_loader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_clean_loader = make_loader(test_clean_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    clean_val_loader, corruption_val_loaders, selector_defs = build_selector_loaders(cfg, val_base, pre_corrupt_tf, post_corrupt_tf)

    exp_dir = ROOT / "checkpoints" / experiment_stem(cfg)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)

    save_json(asdict(cfg), exp_dir / "config.json")
    save_json(selector_defs, exp_dir / "selector_policies.json")

    model = build_model(cfg).to(device)
    if ENABLE_TORCH_COMPILE and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile enabled for this run.")
        except Exception as e:
            print(f"torch.compile skipped: {e}")
    if cfg.feature_perturbation:
        model = FeatureNoiseWrapper(model, where=cfg.feature_perturbation_where, std=cfg.feature_noise_std).to(device)

    optimizer = choose_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = build_scaler(enabled=(cfg.amp and device.type == "cuda"))

    selector_best = {}
    for selector_name, policy in selector_defs.items():
        selector_best[selector_name] = {
            "best_score": -1.0,
            "best_epoch": -1,
            "policy": policy,
        }

    history = []
    stale_epochs = 0
    best_any_score = -1.0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        print(f"Epoch {epoch}/{cfg.epochs} | training...")
        train_out = run_one_epoch(model, train_loader, optimizer=optimizer, scaler=scaler, amp_enabled=cfg.amp)
        scheduler.step()

        epoch_row = {
            "epoch": epoch,
            "train_loss": float(train_out["loss"]),
            "train_acc": float(train_out["acc"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "time_sec": float(time.time() - t0),
        }

        print(f"Epoch {epoch}/{cfg.epochs} | clean validation...")
        clean_val_out = run_one_epoch(model, clean_val_loader, optimizer=None, scaler=None, amp_enabled=cfg.amp)
        epoch_row["val_acc__clean"] = float(clean_val_out["acc"])

        if clean_val_out["acc"] > selector_best["clean"]["best_score"]:
            selector_best["clean"]["best_score"] = float(clean_val_out["acc"])
            selector_best["clean"]["best_epoch"] = int(epoch)
            save_checkpoint_state(model, exp_dir / "best_clean.pt")

        single_corruption_scores = {}
        print(f"Epoch {epoch}/{cfg.epochs} | corrupted validation over {len(NESTED_CORRUPTION_ORDER)} corruption types...")
        for cname in NESTED_CORRUPTION_ORDER:
            print(f"  validating corruption: {cname}")
            val_out = run_one_epoch(model, corruption_val_loaders[cname], optimizer=None, scaler=None, amp_enabled=cfg.amp)
            single_corruption_scores[cname] = float(val_out["acc"])
            epoch_row[f"val_acc__{cname}"] = float(val_out["acc"])

        selector_scores = _selector_scores_from_single_corruptions(single_corruption_scores, selector_defs)
        epoch_best_this_round = float(clean_val_out["acc"])

        for selector_name, score in selector_scores.items():
            epoch_row[f"val_acc__{selector_name}"] = float(score)
            if score > selector_best[selector_name]["best_score"]:
                selector_best[selector_name]["best_score"] = float(score)
                selector_best[selector_name]["best_epoch"] = int(epoch)
                save_checkpoint_state(model, exp_dir / f"best_{selector_name}.pt")
            epoch_best_this_round = max(epoch_best_this_round, float(score))

        history.append(epoch_row)
        print(epoch_row)

        if epoch_best_this_round > best_any_score:
            best_any_score = epoch_best_this_round
            stale_epochs = 0
        else:
            stale_epochs += 1

        save_checkpoint_state(model, exp_dir / f"epoch_{epoch:03d}.pt")

        if stale_epochs >= cfg.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    history_df = pd.DataFrame(history)
    history_df.to_csv(exp_dir / "history.csv", index=False)

    rows = []
    for selector_name, best_meta in selector_best.items():
        ckpt_path = exp_dir / f"best_{selector_name}.pt"
        if not ckpt_path.exists():
            continue

        base_model = build_model(cfg).to(device)
        if cfg.feature_perturbation:
            base_model = FeatureNoiseWrapper(base_model, where=cfg.feature_perturbation_where, std=cfg.feature_noise_std).to(device)
        load_checkpoint_state(base_model, ckpt_path, device)
        base_model.eval()

        test_clean_out = run_one_epoch(base_model, test_clean_loader, optimizer=None, scaler=None, amp_enabled=cfg.amp)
        corrupt_df = evaluate_corruption_suite(
            base_model,
            raw_test_dataset=raw_test,
            cfg=cfg,
            pre_corrupt_tf=pre_corrupt_tf,
            post_corrupt_tf=post_corrupt_tf,
        )
        corrupt_df.to_csv(exp_dir / f"corrupted_test_results__{selector_name}.csv", index=False)

        policy = best_meta["policy"]
        mean_acc = float(corrupt_df.loc[corrupt_df["corruption"] == "mean", "acc"].iloc[0])

        row = {
            "dataset": cfg.dataset_name,
            "model": cfg.model_name,
            "selector": selector_name,
            "selector_type": policy["type"],
            "best_epoch": int(best_meta["best_epoch"]),
            "best_val_score": float(best_meta["best_score"]),
            "num_val_corruptions": int(len(policy["names"])),
            "val_corruption_names": ", ".join(policy["names"]),
            "val_corruption_severity": int(policy["severity"]),
            "test_clean_acc": float(test_clean_out["acc"]),
            "test_corrupt_mean_acc": mean_acc,
            "robustness_gap": float(test_clean_out["acc"] - mean_acc),
            "experiment_dir": str(exp_dir),
            "checkpoint_path": str(ckpt_path),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(
        ["selector_type", "num_val_corruptions", "test_corrupt_mean_acc"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    summary_df.to_csv(exp_dir / "selector_summary.csv", index=False)
    return summary_df, history_df, exp_dir


def full_assignment_grid(
    imagenet100_source="hf",
    imagenet100_root=None,
    train_pretrained=False,
):
    dataset_settings = [
        ("cifar10", 224, 128, 25),
        ("fashion_mnist", 224, 128, 25),
        ("imagenet100", 224, 64, 20),
    ]
    model_settings = [
        ("vgg", 0.01),
        ("resnet", 0.01),
        ("convnext", 3e-4),
        ("vit", 3e-4),
    ]

    all_rows = []
    run_index = 0
    total_runs = len(dataset_settings) * len(model_settings)

    for dataset_name, input_size, batch_size, epochs in dataset_settings:
        for model_name, lr in model_settings:
            run_index += 1
            print("\n" + "=" * 100)
            print(f"Running full assignment pair {run_index}/{total_runs}: {dataset_name} + {model_name}")
            print("=" * 100)

            cfg = ExperimentConfig(
                dataset_name=dataset_name,
                model_name=model_name,
                input_size=input_size,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                train_pretrained=train_pretrained,
                train_subset_fraction=1.0,
                use_randaugment=False,
            )
            summary_df, history_df, exp_dir = fit_assignment_run(
                cfg,
                imagenet100_source=imagenet100_source,
                imagenet100_root=imagenet100_root,
            )
            all_rows.append(summary_df)

            current_master = pd.concat(all_rows, ignore_index=True)
            current_master.to_csv(ROOT / "tables" / "full_assignment_all_selectors.csv", index=False)

            clean_rows = current_master[current_master["selector"] == "clean"].copy()
            best_corrupt_rows = (
                current_master[current_master["selector_type"] == "corrupt"]
                .sort_values(["dataset", "model", "test_corrupt_mean_acc", "test_clean_acc"], ascending=[True, True, False, False])
                .groupby(["dataset", "model"], as_index=False)
                .head(1)
                .reset_index(drop=True)
            )
            clean_rows.to_csv(ROOT / "tables" / "full_assignment_clean_rows.csv", index=False)
            best_corrupt_rows.to_csv(ROOT / "tables" / "full_assignment_best_corrupt_rows.csv", index=False)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_master = pd.concat(all_rows, ignore_index=True)
    final_master.to_csv(ROOT / "tables" / "full_assignment_all_selectors.csv", index=False)

    clean_rows = final_master[final_master["selector"] == "clean"].copy()
    best_corrupt_rows = (
        final_master[final_master["selector_type"] == "corrupt"]
        .sort_values(["dataset", "model", "test_corrupt_mean_acc", "test_clean_acc"], ascending=[True, True, False, False])
        .groupby(["dataset", "model"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    comparison = clean_rows.merge(
        best_corrupt_rows,
        on=["dataset", "model"],
        suffixes=("_clean_selector", "_best_corrupt_selector"),
    )

    clean_rows.to_csv(ROOT / "tables" / "full_assignment_clean_rows.csv", index=False)
    best_corrupt_rows.to_csv(ROOT / "tables" / "full_assignment_best_corrupt_rows.csv", index=False)
    comparison.to_csv(ROOT / "tables" / "clean_vs_best_corrupt_comparison.csv", index=False)

    return final_master, clean_rows, best_corrupt_rows, comparison


def build_report_tables(master_csv=None):
    if master_csv is None:
        master_csv = ROOT / "tables" / "full_assignment_all_selectors.csv"
    df = pd.read_csv(master_csv)

    clean_vs_best = pd.read_csv(ROOT / "tables" / "clean_vs_best_corrupt_comparison.csv")

    all_selector_table = df.pivot_table(
        index=["dataset", "model"],
        columns="selector",
        values="test_corrupt_mean_acc",
        aggfunc="mean",
    ).reset_index()

    clean_accuracy_table = clean_vs_best[[
        "dataset",
        "model",
        "test_clean_acc_clean_selector",
        "test_clean_acc_best_corrupt_selector",
    ]].copy()

    robustness_table = clean_vs_best[[
        "dataset",
        "model",
        "test_corrupt_mean_acc_clean_selector",
        "test_corrupt_mean_acc_best_corrupt_selector",
        "robustness_gap_clean_selector",
        "robustness_gap_best_corrupt_selector",
    ]].copy()

    all_selector_table.to_csv(ROOT / "tables" / "report_all_selector_robustness.csv", index=False)
    clean_accuracy_table.to_csv(ROOT / "tables" / "report_clean_accuracy_comparison.csv", index=False)
    robustness_table.to_csv(ROOT / "tables" / "report_robustness_comparison.csv", index=False)

    display(all_selector_table)
    display(clean_accuracy_table)
    display(robustness_table)

    return all_selector_table, clean_accuracy_table, robustness_table

# %% [code cell 9]

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            if feats.ndim == 4:
                pooled = feats.mean(dim=(2, 3))
            elif feats.ndim == 3:
                pooled = feats.mean(dim=1)
            else:
                pooled = feats

            if hasattr(self.model, "forward_head"):
                logits = self.model.forward_head(feats, pre_logits=False)
            else:
                logits = self.model(x)
            return pooled, logits

        logits = self.model(x)
        return logits, logits


@torch.no_grad()
def collect_features(model, dataset_obj, cfg: ExperimentConfig, max_items=None):
    max_items = cfg.visual_max_items if max_items is None else max_items
    loader = make_loader(dataset_obj, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    feat_model = FeatureExtractor(model).to(device)
    feat_model.eval()

    feats, labels, preds = [], [], []
    seen = 0
    for images, y in loader:
        images = images.to(device)
        f, logits = feat_model(images)
        feats.append(f.detach().cpu().numpy())
        labels.append(y.numpy())
        preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        seen += len(y)
        if seen >= max_items:
            break

    X = np.concatenate(feats, axis=0)[:max_items]
    y = np.concatenate(labels, axis=0)[:max_items]
    p = np.concatenate(preds, axis=0)[:max_items]
    return X, y, p


def save_classification_report(model, dataset_obj, cfg: ExperimentConfig, save_path: str | Path):
    loader = make_loader(dataset_obj, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    out = run_one_epoch(model, loader, optimizer=None, scaler=None, amp_enabled=cfg.amp)
    report = classification_report(out["labels"], out["preds"], output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(save_path, index=True)


def plot_confusion(model, dataset_obj, cfg: ExperimentConfig, title: str, save_path: str | Path):
    loader = make_loader(dataset_obj, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    out = run_one_epoch(model, loader, optimizer=None, scaler=None, amp_enabled=cfg.amp)
    cm = confusion_matrix(out["labels"], out["preds"])

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_representation_projection(model, dataset_obj, cfg: ExperimentConfig, title: str, save_path: str | Path):
    X, y, _ = collect_features(model, dataset_obj, cfg)

    if cfg.representation_method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=cfg.seed)
        Z = reducer.fit_transform(X)
    else:
        reducer = TSNE(n_components=2, random_state=cfg.seed, init="pca", learning_rate="auto")
        Z = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=y, s=8, cmap="tab20", alpha=0.8)
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close()

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Z, y)

    x_min, x_max = Z[:, 0].min() - 1.0, Z[:, 0].max() + 1.0
    y_min, y_max = Z[:, 1].min() - 1.0, Z[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = clf.predict(grid).reshape(xx.shape)

    boundary_path = Path(save_path).with_name(Path(save_path).stem + "_decision_boundary.png")
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz, alpha=0.25, levels=np.arange(zz.max() + 2) - 0.5, cmap="tab20")
    plt.scatter(Z[:, 0], Z[:, 1], c=y, s=8, cmap="tab20", edgecolors="none")
    plt.title(title + " (feature-space decision regions)")
    plt.tight_layout()
    plt.savefig(boundary_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close()


def get_first_conv_layer(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            return name, module
    return None, None


def plot_feature_maps(model, dataset_obj, cfg: ExperimentConfig, save_path: str | Path, num_maps: int = 16):
    model.eval()
    layer_name, layer = get_first_conv_layer(model)
    if layer is None:
        print("No convolution layer found. Skipping feature-map plot.")
        return

    captured = {}

    def hook(module, inputs, output):
        captured["feat"] = output.detach().cpu()

    handle = layer.register_forward_hook(hook)
    x, _ = dataset_obj[0]
    _ = model(x.unsqueeze(0).to(device))
    handle.remove()

    feat = captured["feat"][0]
    num_maps = min(num_maps, feat.shape[0])
    cols = 4
    rows = math.ceil(num_maps / cols)

    plt.figure(figsize=(10, 2.5 * rows))
    for i in range(num_maps):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(feat[i], cmap="viridis")
        plt.axis("off")
        plt.title(f"map {i}")
    plt.suptitle(f"Feature maps from: {layer_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close()


def load_trained_model_for_selector(cfg: ExperimentConfig, selector_name: str):
    exp_dir = ROOT / "checkpoints" / experiment_stem(cfg)
    ckpt_path = exp_dir / f"best_{selector_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(cfg).to(device)
    if cfg.feature_perturbation:
        model = FeatureNoiseWrapper(model, where=cfg.feature_perturbation_where, std=cfg.feature_noise_std).to(device)
    load_checkpoint_state(model, ckpt_path, device)
    model.eval()
    return model, exp_dir


def run_visual_suite_for_selector(
    cfg: ExperimentConfig,
    selector_name: str,
    imagenet100_source: str = "local",
    imagenet100_root: str | Path | None = None,
    corruption_name: str = "gaussian_noise",
    corruption_severity: int = 2,
):
    bundle = build_datasets(cfg, imagenet100_source=imagenet100_source, imagenet100_root=imagenet100_root)
    test_clean_ds = bundle["test_clean_ds"]
    raw_test = bundle["raw_test"]
    pre_corrupt_tf = bundle["pre_corrupt_tf"]
    post_corrupt_tf = bundle["post_corrupt_tf"]

    model, exp_dir = load_trained_model_for_selector(cfg, selector_name)

    test_corrupt_ds = FixedCorruptionDataset(
        raw_test,
        corruption_name=corruption_name,
        severity=corruption_severity,
        pre_transform=pre_corrupt_tf,
        post_transform=post_corrupt_tf,
    )

    plot_dir = exp_dir / "plots" / selector_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    save_classification_report(model, test_clean_ds, cfg, plot_dir / "classification_report_clean.csv")
    save_classification_report(model, test_corrupt_ds, cfg, plot_dir / f"classification_report_{corruption_name}_sev{corruption_severity}.csv")

    plot_confusion(model, test_clean_ds, cfg, title=f"{cfg.dataset_name} | {cfg.model_name} | {selector_name} | clean test",
                   save_path=plot_dir / "confusion_clean.png")
    plot_confusion(model, test_corrupt_ds, cfg, title=f"{cfg.dataset_name} | {cfg.model_name} | {selector_name} | {corruption_name} sev{corruption_severity}",
                   save_path=plot_dir / f"confusion_{corruption_name}_sev{corruption_severity}.png")

    plot_representation_projection(model, test_clean_ds, cfg,
                                   title=f"{cfg.dataset_name} | {cfg.model_name} | {selector_name} | clean",
                                   save_path=plot_dir / "projection_clean.png")
    plot_representation_projection(model, test_corrupt_ds, cfg,
                                   title=f"{cfg.dataset_name} | {cfg.model_name} | {selector_name} | {corruption_name} sev{corruption_severity}",
                                   save_path=plot_dir / f"projection_{corruption_name}_sev{corruption_severity}.png")

    plot_feature_maps(model, test_clean_ds, cfg, save_path=plot_dir / "feature_maps_clean.png")
    plot_feature_maps(model, test_corrupt_ds, cfg, save_path=plot_dir / f"feature_maps_{corruption_name}_sev{corruption_severity}.png")

    print(f"Saved plots to: {plot_dir}")
    return plot_dir



def generate_required_visuals_for_full_assignment(
    imagenet100_source="hf",
    imagenet100_root=None,
    corruption_name="gaussian_noise",
    corruption_severity=2,
    train_pretrained=False,
    selector_scope="all",
):
    """
    selector_scope:
      - "all": create visuals for every selector and every model/dataset pair
      - "clean_vs_best": only the clean selector and the best corrupt selector
    """
    if selector_scope == "all":
        target_rows = pd.read_csv(ROOT / "tables" / "full_assignment_all_selectors.csv")
    else:
        clean_rows = pd.read_csv(ROOT / "tables" / "full_assignment_clean_rows.csv")
        best_corrupt_rows = pd.read_csv(ROOT / "tables" / "full_assignment_best_corrupt_rows.csv")
        target_rows = pd.concat([clean_rows, best_corrupt_rows], ignore_index=True)

    done = []

    for _, row in target_rows.iterrows():
        cfg = ExperimentConfig(
            dataset_name=row["dataset"],
            model_name=row["model"],
            input_size=224,
            batch_size=32 if row["dataset"] != "imagenet100" else 24,
            epochs=25 if row["dataset"] != "imagenet100" else 20,
            lr=0.01 if row["model"] in {"vgg", "resnet"} else 3e-4,
            train_pretrained=train_pretrained,
        )
        selector_name = row["selector"]
        print(f"\nGenerating visuals for {row['dataset']} | {row['model']} | {selector_name}")
        run_visual_suite_for_selector(
            cfg,
            selector_name=selector_name,
            imagenet100_source=imagenet100_source,
            imagenet100_root=imagenet100_root,
            corruption_name=corruption_name,
            corruption_severity=corruption_severity,
        )
        done.append({
            "dataset": row["dataset"],
            "model": row["model"],
            "selector": selector_name,
        })

    done_df = pd.DataFrame(done)
    tag = "all" if selector_scope == "all" else "clean_vs_best"
    done_df.to_csv(ROOT / "tables" / f"generated_visuals_log_{tag}.csv", index=False)
    return done_df


def run_feature_perturbation_ablation(
    dataset_name="cifar10",
    model_name="resnet",
    imagenet100_source="hf",
    imagenet100_root=None,
    train_pretrained=False,
):
    rows = []
    for where in ["early", "mid", "late"]:
        cfg = ExperimentConfig(
            dataset_name=dataset_name,
            model_name=model_name,
            input_size=224,
            batch_size=32 if dataset_name != "imagenet100" else 24,
            epochs=8 if dataset_name != "imagenet100" else 6,
            lr=0.01 if model_name in {"vgg", "resnet"} else 3e-4,
            train_subset_fraction=0.25,
            feature_perturbation=True,
            feature_perturbation_where=where,
            feature_noise_std=0.05,
            train_pretrained=train_pretrained,
        )
        summary_df, history_df, exp_dir = fit_assignment_run(
            cfg,
            imagenet100_source=imagenet100_source,
            imagenet100_root=imagenet100_root,
        )
        best_row = (
            summary_df[summary_df["selector_type"] == "corrupt"]
            .sort_values(["test_corrupt_mean_acc", "test_clean_acc"], ascending=False)
            .head(1)
            .copy()
        )
        if len(best_row):
            best_row["where"] = where
            rows.append(best_row)

    if not rows:
        raise RuntimeError("No perturbation results were produced.")

    df = pd.concat(rows, ignore_index=True)
    save_path = ROOT / "tables" / f"feature_perturbation_{dataset_name}_{model_name}.csv"
    df.to_csv(save_path, index=False)
    display(df)
    print(f"Saved bonus ablation to: {save_path}")
    return df



PAIR_TRAINING_DEFAULTS = {
    "cifar10": {"input_size": 224, "epochs": 25, "base_batch": 200},
    "fashion_mnist": {"input_size": 224, "epochs": 25, "base_batch": 200},
    "imagenet100": {"input_size": 224, "epochs": 20, "base_batch": 200},
}

MODEL_LR_DEFAULTS = {
    "vgg": 0.01,
    "resnet": 0.01,
    "convnext": 3e-4,
    "vit": 3e-4,
}


def _safe_num_workers(requested: int | None = None) -> int:
    cpu = os.cpu_count() or 4
    if requested is None:
        requested = min(12, cpu)
    return max(2, min(int(requested), cpu))


def _batch_candidates(target: int):
    target = int(target)
    ordered = [target, 192, 160, 144, 128, 112, 96, 80, 64, 48, 32]
    out = []
    for x in ordered:
        if x <= target and x not in out:
            out.append(x)
    if target not in out:
        out.insert(0, target)
    return out


def gpu_memory_snapshot():
    if device.type != "cuda":
        return {}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / (1024 ** 3), 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / (1024 ** 3), 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
    }


def auto_pick_batch_size(
    dataset_name: str,
    model_name: str,
    target_batch_size: int = 200,
    imagenet100_source: str = "hf",
    imagenet100_root: str | Path | None = None,
    train_pretrained: bool = False,
    num_workers: int | None = None,
    amp: bool = True,
):
    if device.type != "cuda":
        raise RuntimeError("This split notebook is GPU-only. CUDA device was not found.")

    defaults = PAIR_TRAINING_DEFAULTS[dataset_name]
    num_workers = _safe_num_workers(num_workers)
    candidates = _batch_candidates(target_batch_size)

    probe_cfg = ExperimentConfig(
        dataset_name=dataset_name,
        model_name=model_name,
        input_size=defaults["input_size"],
        batch_size=min(target_batch_size, defaults["base_batch"]),
        num_workers=min(4, num_workers),
        epochs=1,
        lr=MODEL_LR_DEFAULTS[model_name],
        amp=amp,
        train_pretrained=train_pretrained,
        train_subset_fraction=1.0,
        use_randaugment=False,
    )

    bundle = build_datasets(probe_cfg, imagenet100_source=imagenet100_source, imagenet100_root=imagenet100_root)
    train_ds = bundle["train_ds"]
    probe_subset = Subset(train_ds, list(range(min(512, len(train_ds)))))

    for bs in candidates:
        model = None
        optimizer = None
        scaler = None
        loader = None
        try:
            torch.cuda.empty_cache()
            model = build_model(probe_cfg).to(device)
            optimizer = choose_optimizer(probe_cfg, model)
            scaler = build_scaler(enabled=(amp and device.type == "cuda"))
            loader = make_loader(probe_subset, batch_size=bs, shuffle=False, num_workers=min(4, num_workers))
            images, labels = next(iter(loader))
            images = images.to(device, non_blocking=True)
            if images.ndim == 4:
                images = images.contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(enabled=(amp and scaler is not None)):
                logits = model(images)
                loss = nn.CrossEntropyLoss()(logits, labels)
            if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            snap = gpu_memory_snapshot()
            print(f"Auto batch-size probe succeeded with batch_size={bs} | memory={snap}")
            del model, optimizer, scaler, loader, images, labels, logits, loss
            gc.collect()
            torch.cuda.empty_cache()
            return bs
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg:
                print(f"Batch size {bs} failed during probe: {e}")
                try:
                    del model, optimizer, scaler, loader
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
                continue
            raise

    raise RuntimeError("Could not find a working batch size. Try a smaller target or disable pretrained weights.")


def build_pair_config(
    dataset_name: str,
    model_name: str,
    batch_size: int = 200,
    train_pretrained: bool = False,
    num_workers: int | None = None,
    seed: int = 42,
):
    defaults = PAIR_TRAINING_DEFAULTS[dataset_name]
    return ExperimentConfig(
        dataset_name=dataset_name,
        model_name=model_name,
        input_size=defaults["input_size"],
        batch_size=int(batch_size),
        num_workers=_safe_num_workers(num_workers),
        epochs=defaults["epochs"],
        lr=MODEL_LR_DEFAULTS[model_name],
        patience=6,
        amp=True,
        train_pretrained=train_pretrained,
        use_randaugment=False,
        seed=seed,
    )


def summarize_pair_tables(summary_df: pd.DataFrame, pair_dir: str | Path):
    pair_dir = Path(pair_dir)
    summary_df = summary_df.copy()
    summary_df.to_csv(pair_dir / "pair_all_selectors.csv", index=False)
    clean_rows_df = summary_df[summary_df["selector"] == "clean"].copy().reset_index(drop=True)
    best_corrupt_rows_df = (
        summary_df[summary_df["selector_type"] == "corrupt"]
        .sort_values(["test_corrupt_mean_acc", "test_clean_acc"], ascending=[False, False])
        .head(1)
        .reset_index(drop=True)
    )
    if len(clean_rows_df) and len(best_corrupt_rows_df):
        comparison_df = clean_rows_df.merge(
            best_corrupt_rows_df,
            on=["dataset", "model"],
            suffixes=("_clean_selector", "_best_corrupt_selector"),
        )
    else:
        comparison_df = pd.DataFrame()
    clean_rows_df.to_csv(pair_dir / "pair_clean_rows.csv", index=False)
    best_corrupt_rows_df.to_csv(pair_dir / "pair_best_corrupt_rows.csv", index=False)
    comparison_df.to_csv(pair_dir / "pair_clean_vs_best_corrupt_comparison.csv", index=False)
    return clean_rows_df, best_corrupt_rows_df, comparison_df


def generate_required_visuals_for_pair(
    cfg: ExperimentConfig,
    selector_names: list[str] | None = None,
    imagenet100_source: str = "hf",
    imagenet100_root: str | Path | None = None,
    corruption_name: str = "gaussian_noise",
    corruption_severity: int = 2,
):
    exp_dir = ROOT / "checkpoints" / experiment_stem(cfg)
    summary_path = exp_dir / "selector_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Selector summary not found: {summary_path}")
    summary_df = pd.read_csv(summary_path)
    if selector_names is None:
        best_corrupt = (
            summary_df[summary_df["selector_type"] == "corrupt"]
            .sort_values(["test_corrupt_mean_acc", "test_clean_acc"], ascending=[False, False])
            .head(1)["selector"]
            .tolist()
        )
        selector_names = ["clean"] + best_corrupt
    done = []
    for selector_name in selector_names:
        plot_dir = run_visual_suite_for_selector(
            cfg,
            selector_name=selector_name,
            imagenet100_source=imagenet100_source,
            imagenet100_root=imagenet100_root,
            corruption_name=corruption_name,
            corruption_severity=corruption_severity,
        )
        done.append({"selector": selector_name, "plot_dir": str(plot_dir)})
    out_df = pd.DataFrame(done)
    out_df.to_csv(exp_dir / "pair_generated_visuals_log.csv", index=False)
    return out_df


def run_single_pair_job(
    dataset_name: str,
    model_name: str,
    target_batch_size: int = 200,
    imagenet100_source: str = "hf",
    imagenet100_root: str | Path | None = None,
    train_pretrained: bool = False,
    use_torch_compile: bool = True,
    run_required_visuals: bool = True,
    run_optional_feature_perturbation: bool = False,
    num_workers: int | None = None,
    seed: int = 42,
):
    global ENABLE_TORCH_COMPILE
    if device.type != "cuda":
        raise RuntimeError("This split setup is GPU-only. CUDA device is required.")
    ENABLE_TORCH_COMPILE = bool(use_torch_compile)

    chosen_batch_size = auto_pick_batch_size(
        dataset_name=dataset_name,
        model_name=model_name,
        target_batch_size=target_batch_size,
        imagenet100_source=imagenet100_source,
        imagenet100_root=imagenet100_root,
        train_pretrained=train_pretrained,
        num_workers=num_workers,
        amp=True,
    )

    cfg = build_pair_config(
        dataset_name=dataset_name,
        model_name=model_name,
        batch_size=chosen_batch_size,
        train_pretrained=train_pretrained,
        num_workers=num_workers,
        seed=seed,
    )

    print("=" * 100)
    print(f"Running single GPU pair: {dataset_name} + {model_name}")
    print(f"Chosen batch size: {chosen_batch_size}")
    print(f"torch.compile enabled: {ENABLE_TORCH_COMPILE}")
    print(f"num_workers: {cfg.num_workers}")
    print("=" * 100)

    summary_df, history_df, exp_dir = fit_assignment_run(
        cfg,
        imagenet100_source=imagenet100_source,
        imagenet100_root=imagenet100_root,
    )
    clean_rows_df, best_corrupt_rows_df, comparison_df = summarize_pair_tables(summary_df, exp_dir)

    visuals_df = pd.DataFrame()
    if run_required_visuals:
        visuals_df = generate_required_visuals_for_pair(
            cfg,
            selector_names=None,
            imagenet100_source=imagenet100_source,
            imagenet100_root=imagenet100_root,
            corruption_name="gaussian_noise",
            corruption_severity=2,
        )

    feature_perturbation_df = pd.DataFrame()
    if run_optional_feature_perturbation:
        feature_perturbation_df = run_feature_perturbation_ablation(
            dataset_name=dataset_name,
            model_name=model_name,
            imagenet100_source=imagenet100_source,
            imagenet100_root=imagenet100_root,
            train_pretrained=train_pretrained,
        )

    return {
        "cfg": cfg,
        "batch_size": chosen_batch_size,
        "summary_df": summary_df,
        "history_df": history_df,
        "clean_rows_df": clean_rows_df,
        "best_corrupt_rows_df": best_corrupt_rows_df,
        "comparison_df": comparison_df,
        "visuals_df": visuals_df,
        "feature_perturbation_df": feature_perturbation_df,
        "experiment_dir": exp_dir,
    }
