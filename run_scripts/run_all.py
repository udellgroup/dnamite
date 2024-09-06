import numpy as np
import pandas as pd
import subprocess

datasets = ["flchain", "metabric", "support", "unos", "heart_failure"]
batch_sizes = [128, 64, 128, 512, 512]
seeds = [10, 11, 12, 13, 14]
# seeds = [10]
models = ["PseudoNAM"]

for dataset, batch_size in zip(datasets, batch_sizes):
    for seed in seeds:
        for model in models:
            if model == "PseudoNAM" and dataset in ["heart_failure", "unos"]:
                continue
            print(f"Running {model} on {dataset} with seed {seed}")
            subprocess.run([
                "python", 
                f"run_{model.lower()}.py", 
                "--dataset_name", dataset, 
                "--seed", str(seed),
                "--lr", str(5e-4),
                "--batch_size", str(batch_size),
            ])