import numpy as np
import pandas as pd
import subprocess

datasets = ["heart_failure_survival"]
seeds = [10, 11, 12, 13, 14]

for seed in seeds:
    for dataset in datasets:
        
        print()
        print()
        print("RUNNING on ", dataset, " with seed ", seed)
        print()
        print()
        subprocess.run([
            "python",
            "run_dnamite_survival_feature_selection.py",
            "--dataset_name",
            dataset,
            "--seed",
            str(seed),
            "--kernel_weight",
            str(5),
            "--kernel_size",
            str(10),
            "--pair_kernel_weight",
            str(5),
            "--pair_kernel_size",
            str(10),
            "--lr",
            str(5e-4),
            "--reg_param",
            str(0.025),
            "--entropy_param",
            str(1e-3),
            "--gamma",
            str(0.5),
            "--main_pair_strength",
            str(5e-2)
        ])