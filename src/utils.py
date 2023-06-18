import sys
import yaml
from tqdm import tqdm
from datetime import datetime

import torch
from torch import tensor
from torchvision import transforms

from PIL import ImageFilter
from sklearn import random_projection

TQDM_PARAMS = {
    "file": sys.stdout,
    "bar_format": "   {l_bar}{bar:10}{r_bar}{bar:-10b}",
}


def get_tqdm_params():
    return TQDM_PARAMS


class GaussianBlur:
    def __init__(self, radius: int = 4):
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

    def __call__(self, img):
        map_max = img.max()
        final_map = (
                self.load(self.unload(img[0] / map_max).filter(self.blur_kernel)) * map_max
        )
        return final_map


def print_and_export_results(results: dict, method: str):
    """Writes results to .yaml and serialized results to .txt."""

    print("\n   ╭────────────────────────────╮")
    print("   │      Results summary       │")
    print("   ┢━━━━━━━━━━━━━━━━━━━━━━━━━━━━┪")
    print(f"   ┃ average image rocauc: {results['average image rocauc']:.2f} ┃")
    print(f"   ┃ average pixel rocauc: {results['average pixel rocauc']:.2f} ┃")
    print("   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n")

    # write
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    name = f"{method}_{timestamp}"

    results_yaml_path = f"./results/{name}.yml"
    scoreboard_path = f"./results/{name}.txt"

    with open(results_yaml_path, "w") as outfile:
        yaml.safe_dump(results, outfile, default_flow_style=False)
    with open(scoreboard_path, "w") as outfile:
        outfile.write(serialize_results(results["per_class_results"]))

    print(f"   Results written to {results_yaml_path}")


def serialize_results(results: dict) -> str:
    """Serialize a results dict into something usable in markdown."""
    n_first_col = 20
    ans = []
    for k, v in results.items():
        s = k + " " * (n_first_col - len(k))
        s = s + f"| {v[0] * 100:.1f}  | {v[1] * 100:.1f}  |"
        ans.append(s)
    return "\n".join(ans)
