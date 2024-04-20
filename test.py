# import pandas as pd
# import numpy as np

# data = pd.read_csv("MUSE-XAE/datasets/Scenario_4.csv", index_col=0)
# print(data.sum(axis=0))


import os

augs = [50]
senario_max_min = [
    # (1, 13, 8),
    # (2, 13, 8),
    # (3, 13, 8),
    # (4, 5, 2),
    (5, 23, 18),
]
runs = 5

for i in augs:
    for senario, max_sig, min_sig in senario_max_min:
        for k in range(runs):
            print("-------------------")
            print(
                f"python MUSE-XAE/MUSE_XAE.py --dataset Scenario_{senario} --augmentation {i} --max_sig {max_sig} --min_sig {min_sig} --run {k+1} --iter 5 --directory augmentation_{i} --n_jobs 16"
            )
            print("-------------------")

            os.system(
                f"python MUSE-XAE/MUSE_XAE.py --dataset Scenario_{senario} --augmentation {i} --max_sig {max_sig} --min_sig {min_sig} --run {k+1} --iter 5 --directory augmentation_{i} --n_jobs 16"
            )
