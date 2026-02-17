from torch import nn, optim
import torch
import torch.nn.utils
from utils import coRNN, coESN, check, LSTM, get_Adiac_data
from pathlib import Path
import argparse
from tqdm import tqdm
from esn import DeepReservoir
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=100,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=120,
                    help='max epochs')
parser.add_argument('--batch', type=int, default=120,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0021,
                    help='learning rate')
parser.add_argument('--dt', type=float, default=0.01, #0.042
                    help='step size <dt> of the coRNN')
parser.add_argument('--gamma', type=float, default=3., #2.7 
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon', type=float, default=5.0, #4.7
                    help='z controle parameter <epsilon> of the coRNN')
parser.add_argument('--gamma_range', type=float, default=2., #2.7
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon_range', type=float, default=10., #2.7
                    help='z controle parameter <epsilon> of the coRNN')
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--check', action="store_true")
parser.add_argument('--no_friction', action="store_true", help="remove friction term inside non-linearity")
parser.add_argument('--esn', action="store_true")
parser.add_argument('--inp_scaling', type=float, default=10., #1.0
                    help='ESN input scaling')
parser.add_argument('--rho', type=float, default=9.0, #0.99
                    help='ESN spectral radius')
parser.add_argument('--leaky', type=float, default=1.0,
                    help='ESN spectral radius')
parser.add_argument('--use_test', action="store_true")
parser.add_argument('--test_trials', type=int, default=5,
                    help='number of trials to compute mean and std on test')
parser.add_argument('--lstm', action="store_true", help="use LSTM")

# Visualization options
parser.add_argument('--visualize_trajectories', action="store_true",
                   help="Save HRF trajectory visualizations")
parser.add_argument('--viz_n_samples', type=int, default=50,
                   help="Number of neurons to visualize")
parser.add_argument('--visualize_test', action="store_true",
                   help="Also visualize test set trajectories (in addition to train)")
parser.add_argument('--results_dir', type=str, default='results_adiac',
                   help="Directory to save results")


main_folder = 'result'
args = parser.parse_args()
print(args)

device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")

def test(data_loader):
    model.eval()
    correct = 0
    test_loss = 0
    nanflag = False
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)

            output = model(x)
            test_loss += objective(output, y).item()
            pred = output.data.max(1, keepdim=True)[1]
            ytarg = y.data.max(1, keepdim=True)[1] # added
            #correct += pred.eq(y.data.view_as(pred)).sum()
            correct += pred.eq(ytarg.data.view_as(pred)).sum() # changed
    test_loss /= i+1
    print(test_loss)
    accuracy = 100. * correct / len(data_loader.dataset)
    if torch.isnan(torch.Tensor([test_loss])) or test_loss>100000:
        nanflag = True
    return accuracy.item(), nanflag

@torch.no_grad()
def test_esn(data_loader, classifier, scaler):
    activations, ys = [], []
    for x, y in tqdm(data_loader):
        x = x.to(device)
        output = model(x)[-1][0]
        activations.append(output.cpu())
        ys.append(y)
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)

@torch.no_grad()
def extract_trajectories_esn(data_loader, model, device):
    """Extract full trajectories from ESN/coESN model for visualization"""
    # Take first batch only
    x, y = next(iter(data_loader))
    x = x.to(device)
    
    # Get all hidden states across time
    all_states, final_state = model(x)  # (B, L, n_hid)
    
    return all_states  # Return full trajectories


def visualize_ron_trajectories(trajectories, n_samples=50, save_path='hrf_trajectories_RON.png', 
                               title_prefix='Train'):
    """
    Visualize RON (coESN) neuron trajectories over time for ONE sample.
    
    Args:
        trajectories: tensor of shape (B, L, n_hid) - hidden states over time
        n_samples: number of neurons to plot
        save_path: where to save the figure
        title_prefix: 'Train' or 'Test' to label the plot
    """
    # trajectories is (B, L, n_hid)
    B, T, n_hid = trajectories.shape
    
    # Sample neurons uniformly
    if n_samples > n_hid:
        n_samples = n_hid
    neuron_indices = np.linspace(0, n_hid-1, n_samples, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Multiple neurons from first batch sample
    ax = axes[0]
    colors = cm.viridis(np.linspace(0, 1, n_samples))
    
    for idx, neuron_idx in enumerate(neuron_indices):
        trajectory = trajectories[0, :, neuron_idx].cpu().numpy()  # First sample from batch
        ax.plot(trajectory, color=colors[idx], alpha=0.7, linewidth=0.8)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Hidden State (hy)', fontsize=12)
    ax.set_title(f'{title_prefix} - RON Trajectories for {n_samples} Sampled Neurons (1 Sample, Full Sequence)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, T-1])
    
    # Plot 2: Heatmap of all sampled neurons across time
    ax = axes[1]
    heatmap_data = trajectories[0, :, neuron_indices].cpu().numpy().T  # (n_samples, T)
    
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', 
                   interpolation='nearest', origin='lower')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Neuron Index', fontsize=12)
    ax.set_title(f'{title_prefix} - RON State Heatmap ({n_samples} Neurons, 1 Sample, Full Sequence)', fontsize=14)
    
    # Set y-ticks to show actual neuron indices
    ytick_positions = np.linspace(0, n_samples-1, min(10, n_samples), dtype=int)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([neuron_indices[i] for i in ytick_positions])
    
    plt.colorbar(im, ax=ax, label='Hidden State Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ RON trajectories saved to: {save_path}")
    plt.close()


n_inp = 1
n_out = 37 # classes
bs_test = 30
gamma = (args.gamma - args.gamma_range / 2., args.gamma + args.gamma_range / 2.)
epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)

max_test_accs = []
if args.test_trials > 1:
    main_folder = 'result'
    if args.esn:
        train_loader, valid_loader, test_loader = get_Adiac_data(args.batch,bs_test, whole_train=True)
    else:
        train_loader, valid_loader, test_loader = get_Adiac_data(args.batch,bs_test, whole_train=True, RC=False)
else:
    if args.esn:
        train_loader, valid_loader, test_loader = get_Adiac_data(args.batch,bs_test)
    else:
        train_loader, valid_loader, test_loader = get_Adiac_data(args.batch,bs_test, RC=False)
        

for trial in range(args.test_trials):
    accs = []


    if args.lstm:
        model = LSTM(n_inp, args.n_hid, n_out).to(device)
    elif args.esn and not args.no_friction:
        model = DeepReservoir(n_inp, tot_units=args.n_hid, spectral_radius=args.rho,
                            input_scaling=args.inp_scaling,
                            connectivity_recurrent=args.n_hid,
                            connectivity_input=args.n_hid, leaky=args.leaky).to(device)
    elif args.esn and args.no_friction:
        model = coESN(n_inp, args.n_hid, args.dt, gamma, epsilon, args.rho,
                    args.inp_scaling, device=device).to(device)
        if args.check:
            check_passed = check(model)
            print("Check: ", check_passed)
            if not check_passed:
                raise ValueError("Check not passed.")

    else:
        model = coRNN(n_inp, args.n_hid, n_out,args.dt,gamma,epsilon,
                    no_friction=args.no_friction, device=device).to(device)

    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.esn:
        # Visualize trajectories on first trial if requested
        if args.visualize_trajectories and trial == 0:
            print("\n=== Visualizing TRAIN RON Trajectories ===")
            os.makedirs(args.results_dir, exist_ok=True)
            
            train_trajectories = extract_trajectories_esn(train_loader, model, device)
            viz_path = os.path.join(args.results_dir, 
                                   f'ron_trajectories_TRAIN_trial{trial}_nhid{args.n_hid}.png')
            visualize_ron_trajectories(
                train_trajectories, 
                n_samples=args.viz_n_samples,
                save_path=viz_path,
                title_prefix='TRAIN'
            )
            
            if args.visualize_test:
                print("\n=== Visualizing TEST RON Trajectories ===")
                test_trajectories = extract_trajectories_esn(test_loader, model, device)
                viz_path_test = os.path.join(args.results_dir, 
                                           f'ron_trajectories_TEST_trial{trial}_nhid{args.n_hid}.png')
                visualize_ron_trajectories(
                    test_trajectories, 
                    n_samples=args.viz_n_samples,
                    save_path=viz_path_test,
                    title_prefix='TEST'
                )
        
        # Extract train features
        train_feats, ys = [], []
        for x, y in tqdm(train_loader, desc="Extracting train features"):
            x = x.to(device)
            output = model(x)[-1][0]
            train_feats.append(output.cpu())
            ys.append(y)
        train_feats = torch.cat(train_feats, dim=0).numpy()
        ys = torch.cat(ys, dim=0).numpy()
        
        # Standardize features
        scaler = preprocessing.StandardScaler().fit(train_feats)
        train_feats_scaled = scaler.transform(train_feats)
        
        # Extract test features
        test_feats, test_ys = [], []
        for x, y in tqdm(test_loader, desc="Extracting test features"):
            x = x.to(device)
            output = model(x)[-1][0]
            test_feats.append(output.cpu())
            test_ys.append(y)
        test_feats = torch.cat(test_feats, dim=0).numpy()
        test_ys = torch.cat(test_ys, dim=0).numpy()
        test_feats_scaled = scaler.transform(test_feats)
        
        # Print feature statistics
        print(f"\n=== Feature Statistics ===")
        print(f"Train features shape: {train_feats.shape}")
        print(f"Train (raw): mean={train_feats.mean():.4f}, std={train_feats.std():.4f}, min={train_feats.min():.4f}, max={train_feats.max():.4f}")
        print(f"Train (scaled): mean={train_feats_scaled.mean():.4f}, std={train_feats_scaled.std():.4f}")
        print(f"Test features shape: {test_feats.shape}")
        print(f"Test (raw): mean={test_feats.mean():.4f}, std={test_feats.std():.4f}, min={test_feats.min():.4f}, max={test_feats.max():.4f}")
        print(f"Test (scaled): mean={test_feats_scaled.mean():.4f}, std={test_feats_scaled.std():.4f}")
        print(f"\nFeature variance per dimension (raw):")
        print(f"Train feature std per dim: {train_feats.std(axis=0).mean():.4f}")
        print(f"Test feature std per dim: {test_feats.std(axis=0).mean():.4f}")
        
        # Train classifier
        classifier = LogisticRegression(max_iter=1000).fit(train_feats_scaled, ys)
        valid_acc = test_esn(valid_loader, classifier, scaler) if args.test_trials<=1 else 0.0
        test_acc = test_esn(test_loader, classifier, scaler) if args.use_test else 0.0
        accs.append(test_acc)
    else:
        for epoch in range(args.epochs):
            print(f"Epoch {epoch}")
            model.train()
            for x, y in tqdm(train_loader):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                output = model(x)
                loss = objective(output, y)
                loss.backward()
                optimizer.step()

            if args.test_trials<=1:
                valid_acc, nanflag = test(valid_loader)
            else:
                valid_acc, nanflag = 0.0, False
            if args.use_test:
                test_acc, nanflag = test(test_loader)
            else:
                test_acc = 0.0
            accs.append(test_acc)
            if nanflag:
                break # I don't want to run many epochs if I get nan values early...
            Path(main_folder).mkdir(parents=True, exist_ok=True)
            if args.lstm:
                f = open(f'{main_folder}/Adiac_log_lstm.txt', 'a')
            elif args.no_friction:
                f = open(f'{main_folder}/Adiac_log_no_friction.txt', 'a')
            else:
                f = open(f'{main_folder}/Adiac_log.txt', 'a')

            f.write('valid accuracy: ' + str(round(valid_acc, 4)) + '\n')
            f.write('test accuracy: ' + str(round(test_acc, 4)) + '\n')
            f.close()
            print(f"Valid accuracy: ", valid_acc)
            print(f"Test accuracy: ", test_acc)

            if (epoch+1) % 100 == 0:
                scaled_lr = args.lr / 10.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scaled_lr

    if args.lstm:
        f = open(f'{main_folder}/Adiac_log_lstm.txt', 'a') 
    elif args.no_friction and (not args.esn): # coRNN without friction
        f = open(f'{main_folder}/Adiac_log_no_friction.txt', 'a')
    elif args.esn and args.no_friction: # coESN
        f = open(f'{main_folder}/Adiac_log_coESN.txt', 'a')
    elif args.esn: # ESN
        f = open(f'{main_folder}/Adiac_log_esn.txt', 'a')
    else: # original coRNN
        f = open(f'{main_folder}/Adiac_log.txt', 'a')
    ar = ''
    for k, v in vars(args).items():
        ar += f'{str(k)}: {str(v)}, '
    ar += f'valid: {str(round(valid_acc, 4))}, test: {str(round(test_acc, 4))}'
    f.write(ar + '\n')
    f.write('**************\n\n\n')
    f.close()

    max_test_accs.append(max(accs))

mean_test = np.mean(np.array(max_test_accs))
std_test = np.std(np.array(max_test_accs))

if args.lstm:
    f = open(f'{main_folder}/Adiac_log_lstm.txt', 'a')
elif args.no_friction and (not args.esn): # coRNN without friction
    f = open(f'{main_folder}/Adiac_log_no_friction.txt', 'a')
elif args.esn and args.no_friction: # coESN
    f = open(f'{main_folder}/Adiac_log_coESN.txt', 'a')
elif args.esn: # ESN
    f = open(f'{main_folder}/Adiac_log_esn.txt', 'a')
else: # original coRNN
    f = open(f'{main_folder}/Adiac_log.txt', 'a')
ar = f'List of maximum test accuracies: {str(max_test_accs)}'
f.write(ar + '\n')
ar = f'Mean test accuracy: {str(round(mean_test, 4))}, Std test accuracy: {str(round(std_test, 4))}'
f.write(ar + '\n')
f.write('**************\n\n\n')
f.close()