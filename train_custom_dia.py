
from pathlib import Path
import torch

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd


# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "My_dataset"

TRAIN_DIR = DATASET_DIR / "train" / "good"
TEST_DIR = DATASET_DIR / "test"

RESULTS_DIR = BASE_DIR / "results" / "custom_dia"
IMAGENET_DIR = BASE_DIR / "datasets" / "imagenette"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

EPOCHS = 20


# =====================================================
# CHECK DATASET
# =====================================================

def count_images(folder):

    extensions = [
        ".jpg", ".jpeg", ".png", ".bmp"
    ]

    return sum(
        1 for path in folder.rglob("*")
        if path.suffix.lower() in extensions
    )


def check_dataset():

    required_folders = [
        TRAIN_DIR,
        TEST_DIR / "good",
        TEST_DIR / "cut_lead",
    ]

    for folder in required_folders:

        if not folder.exists():
            raise FileNotFoundError(
                f"Missing dataset folder: {folder}"
            )

    train_count = count_images(TRAIN_DIR)
    good_count = count_images(TEST_DIR / "good")
    defect_count = count_images(TEST_DIR / "cut_lead")

    print("\n========== DATASET ==========")
    print("Training normal images:", train_count)
    print("Testing normal images:", good_count)
    print("Testing defective images:", defect_count)
    print("=============================\n")

    if train_count == 0 or good_count == 0 or defect_count == 0:
        raise RuntimeError(
            "One or more dataset folders are empty."
        )


# =====================================================
# CREATE DATA MODULE
# =====================================================

def create_datamodule():

    datamodule = Folder(
        name="custom_transistor",

        normal_dir=TRAIN_DIR,

        abnormal_dir=[
            TEST_DIR / "cut_lead"
        ],

        normal_test_dir=TEST_DIR / "good",

        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,

        test_split_mode="from_dir",

        # Keep the test set separate from validation.
        val_split_mode="from_train",
        val_split_ratio=0.2,

        seed=42,
    )

    return datamodule


# =====================================================
# TRAIN AND TEST
# =====================================================

def main():

    check_dataset()

    print("Device:", DEVICE)

    datamodule = create_datamodule()

    model = EfficientAd(
        model_size="small",
        imagenet_dir=IMAGENET_DIR,
    )

    engine = Engine(
        accelerator=DEVICE,
        devices=1,
        max_epochs=EPOCHS,
        default_root_dir=RESULTS_DIR,
        enable_checkpointing=True,
        logger=False,
    )

    # -----------------------------
    # TRAIN
    # -----------------------------

    print("\n================================")
    print("       TRAINING DIA")
    print("================================\n")

    engine.train(
        model=model,
        datamodule=datamodule,
    )

    print("\nTraining completed!")

    # -----------------------------
    # TEST
    # -----------------------------

    print("\n================================")
    print("        TESTING DIA")
    print("================================\n")

    results = engine.test(
        model=model,
        datamodule=datamodule,
    )

    print("\n========== TEST RESULTS ==========")
    print(results)
    print("==================================")

    print("\nDIA custom training and testing completed.")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
