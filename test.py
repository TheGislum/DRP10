# import pandas as pd
# import numpy as np

# data = pd.read_csv("MUSE-XAE/datasets/Scenario_4.csv", index_col=0)
# print(data.sum(axis=0))


# import os

# augs = [20]
# senario_max_min = [
#     (1, 13, 8),
#     (2, 13, 8),
#     (3, 13, 8),
#     (4, 5, 2),
#     (5, 23, 18),
# ]
# runs = 1

# for i in augs:
#     for senario, max_sig, min_sig in senario_max_min:
#         for k in range(runs):
#             print("-------------------")
#             print(
#                 f"python MUSE-XAE/MUSE_XAE.py --dataset Scenario_{senario} --augmentation {i} --max_sig {max_sig} --min_sig {min_sig} --run {k+1} --iter 5 --directory augmentation_{i} --n_jobs 16"
#             )
#             print("-------------------")

#             os.system(
#                 f"python MUSE-XAE/MUSE_XAE.py --dataset Scenario_{senario} --augmentation {i} --max_sig {max_sig} --min_sig {min_sig} --run {k+1} --iter 5 --directory augmentation_{i} --n_jobs 16"
#             )

import numpy as np
import pandas as pd

# E = np.random.normal(1, 0, (1000, 11))
# original_data = np.random.normal(1, 0, (1000, 11))

# # Es = (
# #     (E.apply(lambda x: x / (sum(x) + 1e-10)) * np.sum(original_data, axis=1).T) / E
# # ).to_numpy()
# # write above function only using numpy
# Es = ((E / np.re(np.sum(E, axis=1) + 1e-10, 0)) * np.sum(original_data, axis=1)).T / E

# # Es = ((E / (np.sum(E, axis=1) + 1e-10)) * np.sum(d2, axis=1)).T / E
# print(Es.shape)


CI_high = pd.read_csv(
    "Experiments/Scenario_1/Suggested_SBS_De_Novo/MUSE_EXP_CI_high.csv",
    index_col=0,
)
CI_low = pd.read_csv(
    "Experiments/Scenario_1/Suggested_SBS_De_Novo/MUSE_EXP_CI_low.csv",
    index_col=0,
)
trouth = pd.read_csv(
    "Ground_truths/scenario_1/ground.truth.syn.exposures.csv", index_col=0
).T

pred = pd.read_csv(
    "Experiments/Scenario_1/Suggested_SBS_De_Novo/MUSE_EXP.csv",
    index_col=0,
)

sig_match = pd.read_csv(
    "Experiments/Scenario_1/Suggested_SBS_De_Novo/COSMIC_match.csv", index_col=0
)
sig_map = {}
for i in sig_match.index:
    sig_map[sig_match.loc[i, "MUSE-SBS"]] = sig_match.loc[i, "COSMIC-SBS"]

pred.columns = [sig_map[i] for i in pred.columns]

# allign so that columns are same (not all ar in both)
in_both = trouth.columns.intersection(pred.columns)
trouth = trouth[in_both].to_numpy()
pred = pred[in_both].to_numpy()
print(f"trouth: {trouth.shape}, pred: {pred.shape}")
CI_high = CI_high[in_both].to_numpy()
CI_low = CI_low[in_both].to_numpy()

mse = np.mean((trouth - pred) ** 2)
print(mse)

# % of trouth lower than CI_high and higher than CI_low
all = np.mean((((trouth >= CI_low) * 1) == ((trouth <= CI_high) * 1)) * 1)
low = np.mean((trouth >= CI_low) * 1)
high = np.mean((trouth <= CI_high) * 1)
print(f"all: {all}, low: {low}, high: {high}")


# data = pd.read_csv(
#     "/home/gislum/python/muse-xae/DRP10/Ground_truths/scenario_1/ground.truth.syn.exposures.csv",
#     index_col=0,
# ).T
# data.to_csv(
#     "/home/gislum/python/muse-xae/DRP10/Ground_truths/scenario_1/ground.truth.syn.exposures.T.csv"
# )
