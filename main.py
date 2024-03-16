import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from beta_vae import BetaVAE
from utils import make_dataset, train, data_augmentation


if __name__ == "__main__":

    beta = 2
    latent_dim = 15
    batch_size = 64
    lr = 1e-5
    epochs = 70

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mf_df = pd.read_csv("datasets/simple8/dataset.txt", index_col=0, sep="\t")
    mf_df = mf_df.T  # (2780, 96)

    mf_df = data_augmentation(mf_df.values, augmentation=4)

    index = mf_df.index  # 2780
    columns = mf_df.columns  # 96

    train_loader, test_loader, df = make_dataset(
        datasetPath=mf_df, batch_size_n=batch_size
    )

    input_dim = train_loader.dataset.x1.shape[1]

    model = BetaVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        beta=beta,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train(
        train_loader,
        test_loader,
        model,
        optimizer,
        epochs=epochs,
        device=device,
    )

    mu, log_var = model.encode(torch.tensor(df.values, dtype=torch.float32).to(device))
    z = (model.reparameterize(mu, log_var)).cpu().detach().numpy()  # (2780,10)
    mu = mu.cpu().detach().numpy()
    log_var = log_var.cpu().detach().numpy()

    signatures = model.decoder.weight.data.cpu().detach().numpy()  # (96,64)

    sig_name = [f"sig{i}" for i in range(latent_dim)]

    pd.DataFrame(z, index=index, columns=sig_name).T.to_csv("weights.csv")
    pd.DataFrame(signatures, index=columns, columns=sig_name).to_csv("signatures.csv")
