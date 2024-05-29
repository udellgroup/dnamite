import numpy as np
import pandas as pd
import subprocess

datasets = ["support", "heart_failure_survival", "unos"]
seeds = [10, 11, 12, 13, 14]
splits = [1, 2, 3, 4, 5]
models = ["DNAMite", "EBM", "SATransformer"]

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
                    "run_ebm_survival.py",
                    "--dataset_name",
                    dataset,
                    "--seed",
                    str(seed),
                    "--reg_param",
                    # str(0.0045)
                    str(0),
                    "--subsample_rate",
                    str(0.1)
                ])
                
        else:
            for dataset in datasets:
                
                if dataset == "phoneme":
                    batch_size = 64
                    max_bins = 16
                    
                elif dataset == "unos":
                    
                    batch_size = 256
                    max_bins = 32
                    
                else:
                    batch_size = 128
                    max_bins = 32
                
                for split in splits:
                    print()
                    print()
                    print("RUNNING ", model, " on ", dataset, " with seed ", seed, " and split ", split)
                    print()
                    print()
                    if model == "DNAMite":
                        subprocess.run([
                            "python",
                            "run_dnamite_survival.py",
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
                        
                    elif model == "SA_Transformer":
                        subprocess.run([
                            "python",
                            "run_sa_transformer.py",
                            "--dataset_name",
                            dataset,
                            "--seed",
                            str(seed),
                            "--split",
                            str(split),
                            "--batch_size",
                            str(batch_size),
                            "--hidden_dim",
                            str(128),
                            "--lr",
                            str(5e-4),
                            "--n_eval_times",
                            str(100),
                            # "--use_feature_set"
                        ])
                        
                    else:
                        continue