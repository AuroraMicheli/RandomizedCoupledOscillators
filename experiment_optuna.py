# experiment_optuna.py
import optuna
import torch
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from utils import get_mnist_data
from utils_aurora import visualize_dynamics_and_spikes
from model_optuna import spiking_coESN


def extract_features(loader, model, device):
    model.eval()
    feats, labels_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # reshape MNIST to (B, T, input_dim)
            images = images.reshape(images.shape[0], 1, 784).permute(0, 2, 1)
            rates = model(images)
            feats.append(rates.cpu())
            labels_all.append(labels)
    feats = torch.cat(feats, dim=0).numpy()
    labels_all = torch.cat(labels_all, dim=0).numpy()
    return feats, labels_all


def run_training(gamma, epsilon, inp_scaling, lif_tau_m,
                 spike_gain, theta_lif, theta_rf, rho, dt):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_hid = 256       # reservoir size
    n_inp = 1     # because MNIST flattened

    bs_train = 256
    bs_test = 100

    train_loader, valid_loader, _ = get_mnist_data(bs_train, bs_test)

    # --------------------------
    # CORRECTED MODEL CALL
    # --------------------------
    model = spiking_coESN(
        n_inp=n_inp,
        n_hid=n_hid,
        dt=dt,
        gamma=gamma,
        epsilon=epsilon,
        rho=rho,
        input_scaling=inp_scaling,   # correct name
        lif_tau_m=lif_tau_m,
        spike_gain=spike_gain,
        theta_lif=theta_lif,
        theta_rf=theta_rf,
        device=device
    ).to(device)

    # Extract features
    train_feats, train_labels = extract_features(train_loader, model, device)
    valid_feats, valid_labels = extract_features(valid_loader, model, device)

    # Normalize features
    scaler = preprocessing.StandardScaler().fit(train_feats)
    train_feats = scaler.transform(train_feats)
    valid_feats = scaler.transform(valid_feats)

    # Logistic regression classifier
    clf = LogisticRegression(max_iter=800, n_jobs=-1)
    clf.fit(train_feats, train_labels)

    valid_acc = clf.score(valid_feats, valid_labels) * 100

    return valid_acc


def objective(trial):
    gamma = trial.suggest_float("gamma", 2.3, 3.3)
    epsilon = trial.suggest_float("epsilon", 4.2, 5.4)
    inp_scaling = trial.suggest_float("inp_scaling", 1.0, 4.0)

    lif_tau_m = trial.suggest_float("lif_tau_m", 15, 28)
    spike_gain = trial.suggest_float("spike_gain", 3.0, 9.0)

    theta_lif = trial.suggest_float("theta_lif", 0.02, 1.0)
    theta_rf = trial.suggest_float("theta_rf", 0.02, 1.0)

    rho = trial.suggest_float("rho", 0.8, 1)
    dt = trial.suggest_float("dt", 0.02, 0.06)

    acc = run_training(
        gamma, epsilon, inp_scaling,
        lif_tau_m, spike_gain,
        theta_lif, theta_rf,
        rho, dt
    )
    return acc


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best trial:", study.best_trial.value)
    print("Best params:", study.best_trial.params)

    with open("best_params_s.txt", "w") as f:
        for k, v in study.best_trial.params.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()


