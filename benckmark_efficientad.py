
from pathlib import Path

import torch

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "dataset" / "transistor"

TRAIN_DIR = DATA_DIR / "train" / "good"
TEST_DIR = DATA_DIR / "test"
MASK_DIR = DATA_DIR / "ground_truth"

RESULTS_DIR = BASE_DIR / "results" / "efficientad_s"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

EPOCHS = 20

CATEGORIES = [
    "bent_lead",
    "cut_lead",
    "damaged_case",
    "misplaced",
]


# ==========================================
# CHECK DATASET
# ==========================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Missing training directory: {TRAIN_DIR}")

for category in ["good"] + CATEGORIES:
    if not (TEST_DIR / category).exists():
        raise FileNotFoundError(
            f"Missing test directory: {TEST_DIR / category}"
        )

print("=" * 50)
print("DIA - DEVICE INSPECTOR AARIZ")
print("EfficientAD-S Benchmark")
print("=" * 50)

print(f"Device: {DEVICE}")
print(f"Training images: {TRAIN_DIR}")
print(f"Test images: {TEST_DIR}")


# ==========================================
# DATASET
# ==========================================

print("\nPreparing dataset...")

datamodule = Folder(
    name="transistor",

    normal_dir=TRAIN_DIR,

    abnormal_dir=[
        TEST_DIR / category
        for category in CATEGORIES
    ],

    normal_test_dir=TEST_DIR / "good",

    mask_dir=[
        MASK_DIR / category
        for category in CATEGORIES
    ],

    train_batch_size=1,
    eval_batch_size=1,

    num_workers=0,

    test_split_mode="from_dir",

    val_split_mode="from_test",
    val_split_ratio=0.2,

    seed=42,
)

datamodule.setup()

print("Dataset prepared!")


# ==========================================
# MODEL
# ==========================================

print("\nInitialising EfficientAD-S...")

model = EfficientAd(
    model_size="small",

    imagenet_dir=BASE_DIR / "datasets" / "imagenette",

    lr=0.0001,
)

print("Model initialised!")


# ==========================================
# TRAINING ENGINE
# ==========================================

engine = Engine(
    default_root_dir=RESULTS_DIR,

    accelerator=DEVICE,
    devices=1,

    max_epochs=EPOCHS,

    enable_checkpointing=True,

    logger=False,
)


# ==========================================
# TRAIN
# ==========================================

print("\nStarting training...")
print("This may take a while on the first run.")

train_results = engine.train(
    model=model,
    datamodule=datamodule,
)

print("\nTraining completed!")


# ==========================================
# TEST
# ==========================================

print("\nEvaluating on the test dataset...")

test_results = engine.test(
    model=model,
    datamodule=datamodule,
)

print("\n" + "=" * 50)
print("DIA - EFFICIENTAD-S RESULTS")
print("=" * 50)

print("\nTraining results:")
print(train_results)

print("\nTest results:")
print(test_results)

print("\nResults directory:")
print(RESULTS_DIR)

print("\nBenchmark completed!")
