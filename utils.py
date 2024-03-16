import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df1, df2):
        self.x1 = torch.tensor(df1.values, dtype=torch.float32)
        self.x2 = torch.tensor(df2.values, dtype=torch.float32)

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx]


def add_noise(x, noise_factor=0.0) -> np.ndarray:
    x_noisy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    x_noisy = np.clip(x_noisy, 0.0, 1.0)
    return x_noisy


def make_dataset(
    datasetPath: str | pd.DataFrame,
    batch_size_n=32,
    frac=0.2,
    noise_factor=0.0,
):
    # dataset (mutations X signatures)
    if isinstance(datasetPath, str):
        mf_df = pd.read_csv(datasetPath, index_col=0, sep="\t")
        mf_df = mf_df.T
    else:
        mf_df = datasetPath

    # normalize columns
    mf_df = mf_df.div(mf_df.sum(axis=0), axis=1)

    # split dataset column wise
    x_test = mf_df.sample(frac=frac)
    # x_test = mf_df
    x_train = mf_df.drop(x_test.index)

    # x_train = mf_df
    x_train_noisy = add_noise(x_train, noise_factor)

    x_test_noisy = add_noise(x_test, noise_factor)

    train_dataset = Dataset(x_train_noisy, x_train)
    test_dataset = Dataset(x_test_noisy, x_test)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_n, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size_n, shuffle=False
    )
    return train_loader, test_loader, mf_df


def train(
    train_loader,
    test_loader,
    model: nn.Module,
    optimizer,
    epochs=50,
    device="cpu",
):
    model.to(device)
    model.train()

    total_train_loss = []
    total_train_recons_loss = []
    total_train_kld_loss = []
    total_test_loss = []
    total_test_recons_loss = []
    total_test_kld_loss = []

    for e in range(epochs):
        train_losses = []
        for x_n, x_o in train_loader:
            x_n = x_n.to(device)
            x_o = x_o.to(device)
            optimizer.zero_grad()
            x_pred, mu, log_var = model(x_n)
            loss = model.loss_function(
                x_pred, input=x_o, mu=mu, log_var=log_var, kld_weight=1.0
            )
            loss["loss"].backward()
            optimizer.step()
            train_losses.append(loss["loss"].item())
            # total_train_recons_loss.append(loss["recons_loss"].item())
            # total_train_kld_loss.append(loss["kld"].item())
        total_train_loss.append(np.mean(train_losses))

        test_losses = []
        model.eval()
        with torch.no_grad():
            for x_n, x_o in test_loader:
                x_n = x_n.to(device)
                x_o = x_o.to(device)
                x_pred, mu, log_var = model(x_n)
                loss = model.loss_function(
                    x_pred, input=x_o, mu=mu, log_var=log_var, kld_weight=1.0
                )
                test_losses.append(loss["loss"].item())
                # total_test_recons_loss.append(loss["recons_loss"].item())
                # total_test_kld_loss.append(loss["kld"].item())
            total_test_loss.append(np.mean(test_losses))
        model.train()
        print(
            f"Epoch {e+1}/{epochs}, Train Loss: {total_train_loss[-1]:.4f}, Test Loss: {total_test_loss[-1]:.4f}"
        )

    # loss graph
    # plt.plot(total_train_loss, label="Train")
    # plt.plot(total_test_loss, label="Test")
    # plt.show()


def data_augmentation(X, augmentation=5):

    X_augmented = []

    for time in range(augmentation):
        X_bootstrapped = []
        for x in X:
            N = int(round(np.sum(x)))
            p = np.ravel(x / np.sum(x))
            X_bootstrapped.append(np.random.multinomial(N, p))
        X_bootstrapped = np.array(X_bootstrapped)
        X_augmented.append(pd.DataFrame(X_bootstrapped))
    X_aug = pd.concat(X_augmented, axis=0)

    return X_aug
