import numpy as np
import pandas as pd
import subprocess

datasets = ["adult"]
seeds = [10, 11, 12, 13, 14]
splits = [1, 2, 3, 4, 5]
models = ["DNAMite", "NAM", "NAM_ExU", "EBM"]

for seed in seeds:
    for model in models:
        if model == "EBM":
            for dataset in datasets:
                print()
                print()
                print("RUNNING ", model, " on ", dataset, " with seed ", seed)
                print()
                print()
                subprocess.run([
                    "python",
                    "run_ebm_binary.py",
                    "--dataset_name",
                    dataset,
                    "--seed",
                    str(seed),
                ])
                
        elif model == "XGBoost":
            
            for dataset in datasets:
                print()
                print()
                print("RUNNING ", model, " on ", dataset, " with seed ", seed)
                print()
                print()
                subprocess.run([
                    "python",
                    "run_xgboost_binary.py",
                    "--dataset_name",
                    dataset,
                    "--seed",
                    str(seed),
                ])
                
        else:
            for dataset in datasets:
                
                if dataset == "phoneme":
                    batch_size = 64
                    max_bins = 16
                    num_interactions = 10
                else:
                    batch_size = 128
                    max_bins = 32
                    num_interactions = 20
                
                for split in splits:
                    print()
                    print()
                    print("RUNNING ", model, " on ", dataset, " with seed ", seed, " and split ", split)
                    print()
                    print()
                    if model == "DNAMite":
                        subprocess.run([
                            "python",
                            "run_dnamite_binary.py",
                            "--dataset_name",
                            dataset,
                            "--seed",
                            str(seed),
                            "--split",
                            str(split),
                            "--max_bins",
                            str(max_bins),
                            "--batch_size",
                            str(batch_size),
                            "--num_interactions",
                            str(num_interactions),
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
                            # "--use_feature_set"
                            # "--reg_param",
                            # str(1e-4),
                            # "--entropy_param",
                            # str(1e-3),
                            # "--gamma",
                            # str(0.5),
                        ])
                        
                    elif model == "DNAMite_kw0":
                        subprocess.run([
                            "python",
                            "run_dnamite_binary.py",
                            "--dataset_name",
                            dataset,
                            "--seed",
                            str(seed),
                            "--split",
                            str(split),
                            "--max_bins",
                            str(max_bins),
                            "--batch_size",
                            str(batch_size),
                            "--num_interactions",
                            str(num_interactions),
                            "--kernel_weight",
                            str(0),
                            "--kernel_size",
                            str(0),
                            "--pair_kernel_weight",
                            str(0),
                            "--pair_kernel_size",
                            str(0),
                            "--lr",
                            str(5e-4),
                            # "--use_feature_set"
                            # "--reg_param",
                            # str(1e-4),
                            # "--entropy_param",
                            # str(1e-3),
                            # "--gamma",
                            # str(0.5),
                        ])
                    elif model == "NAM":
                        subprocess.run([
                            "python",
                            "run_nam_binary.py",
                            "--dataset_name",
                            dataset,
                            "--seed",
                            str(seed),
                            "--split",
                            str(split),
                            "--lr",
                            str(5e-4),
                            "--batch_size",
                            str(batch_size),
                            "--num_interactions",
                            str(num_interactions),
                        ])
                    elif model == "NAM_ExU":
                        subprocess.run([
                            "python",
                            "run_nam_binary.py",
                            "--dataset_name",
                            dataset,
                            "--seed",
                            str(seed),
                            "--split",
                            str(split),
                            "--exu",
                            "--lr",
                            str(5e-4),
                            "--batch_size",
                            str(batch_size),
                            "--num_interactions",
                            str(num_interactions),
                        ])
                    else:
                        continue