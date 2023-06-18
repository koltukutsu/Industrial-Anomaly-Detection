import warnings  # for some torch warnings regarding depreciation
import click

from data import MVTecDataset, mvtec_classes
from models import SPADE
from utils import print_and_export_results

from typing import List

# seeds
import torch
import random
import numpy as np
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

warnings.filterwarnings("ignore")

ALL_CLASSES = mvtec_classes()
# ALLOWED_METHODS = ["spade", "padim", "patchcore"]


def run_model(classes: List):
    results = {}
    method = "SPADE"
    for cls in classes:
        model = SPADE(
            k=50,
            backbone_name="wide_resnet50_2",
        )

        print(f"\n█│ Running {method} on {cls} dataset.")
        print(f" ╰{'─'*(len(method)+len(cls)+23)}\n")
        train_ds, test_ds = MVTecDataset(cls).get_dataloaders()

        print("   Training ...")
        model.fit(train_ds)
        print("   Testing ...")
        image_rocauc, pixel_rocauc = model.evaluate(test_ds)

        print(f"\n   ╭{'─'*(len(cls)+15)}┬{'─'*20}┬{'─'*20}╮")
        print(
            f"   │ Test results {cls} │ image_rocauc: {image_rocauc:.2f} │ pixel_rocauc: {pixel_rocauc:.2f} │")
        print(f"   ╰{'─'*(len(cls)+15)}┴{'─'*20}┴{'─'*20}╯")
        results[cls] = [float(image_rocauc), float(pixel_rocauc)]

    image_results = [v[0] for _, v in results.items()]
    average_image_roc_auc = sum(image_results)/len(image_results)
    image_results = [v[1] for _, v in results.items()]
    average_pixel_roc_auc = sum(image_results)/len(image_results)

    total_results = {
        "per_class_results": results,
        "average image rocauc": average_image_roc_auc,
        "average pixel rocauc": average_pixel_roc_auc,
        "model parameters": model.get_parameters(),
    }
    return total_results


@click.command()
@click.option("--dataset", default="all", help="Dataset, defaults to all datasets.")
def cli_interface(dataset: str):
    if dataset == "all":
        dataset = ALL_CLASSES
    else:
        dataset = [dataset]

    total_results = run_model(dataset)

    print_and_export_results(total_results, "SPADE")


if __name__ == "__main__":
    cli_interface()
