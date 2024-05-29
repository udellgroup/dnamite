import numpy as np
import pandas as pd
import subprocess

datasets = ["heart_failure_binary"]
seeds = [10, 11, 12, 13, 14]
# seeds = [10]
models = ["DNAMite"]

for seed in seeds:
    for model in models:
        for dataset in datasets:
                
            if dataset == "phoneme":
                batch_size = 64
                max_bins = 16
                num_interactions = 10
            else:
                batch_size = 128
                max_bins = 32
                num_interactions = 20
            
            print()
            print()
            print("RUNNING ", model, " on ", dataset, " with seed ", seed)
            print()
            print()
            if model == "DNAMite":
                subprocess.run([
                    "python",
                    "run_dnamite_binary_feature_selection.py",
                    "--dataset_name",
                    dataset,
                    "--seed",
                    str(seed),
                    "--max_bins",
                    str(max_bins),
                    "--batch_size",
                    str(batch_size),
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
                    str(1e-4),
                    "--entropy_param",
                    str(1e-3),
                    "--gamma",
                    str(0.5),
                ])
            elif model == "NAM":
                subprocess.run([
                    "python",
                    "run_nam_binary_feature_selection.py",
                    "--dataset_name",
                    dataset,
                    "--seed",
                    str(seed),
                    "--lr",
                    str(5e-4),
                    "--batch_size",
                    str(batch_size),
                    "--reg_param",
                    str(1e-4),
                    "--entropy_param",
                    str(1e-3),
                    "--gamma",
                    str(0.5),
                ])
            elif model == "NAM_ExU":
                subprocess.run([
                    "python",
                    "run_nam_binary_feature_selection.py",
                    "--dataset_name",
                    dataset,
                    "--seed",
                    str(seed),
                    "--exu",
                    "--lr",
                    str(5e-4),
                    "--batch_size",
                    str(batch_size),
                    "--reg_param",
                    str(1e-4),
                    "--entropy_param",
                    str(1e-3),
                    "--gamma",
                    str(0.5),
                ])
            else:
                continue