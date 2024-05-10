import pandas as pd
import tensorflow as tf
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from utils import (
    load_dataset,
    plot_optimal_solution,
    optimal_model,
    optimal_cosine_similarity,
    refit,
    plot_results,
)
import warnings

warnings.filterwarnings("ignore")

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == "__main__":

    X = load_dataset(name="Scenario_1", cosmic_version="3.4")
    # Refit
    S = pd.read_csv(
        "/home/gislum/python/muse-xae/DRP10/Ground_truths/scenario_1/ground.truth.syn.sigs.csv",
        index_col=0,
    )
    mustation = [f"{y[0]}[{x}]{y[2]}" for x, y in zip(S.index, S["Trinucleotide"])]
    S.index = mustation
    S.sort_index(inplace=True)
    S = S.iloc[:, 1:]

    E, P = refit(
        X, S=S, best={"signatures": 11}, save_to="Experiments/Scenario_1/Models/"
    )

    # Plot extracted signatures
    try:
        tumour_types = [column.split("::")[0] for column in X.index]
    except:
        try:
            tumour_types = [column.split("-")[0] for column in X.index]
        except:
            tumour_types = None

    index_signatures = X.columns
    plot_results(
        X,
        S=S,
        E=E,
        P=P,
        sig_index=index_signatures,
        tumour_types=tumour_types,
        save_to="Experiments/Scenario_1/",
        cosmic_version="3.4",
    )
