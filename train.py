import os
import json
import argparse
import torch
import torch.optim as optim
from glow import SqueezeWave, SqueezeWaveLoss
from dataset import AudiobookDataset
from dataset import discrete_collate
import numpy as np

def nll_loss(y_hat, y, reduce=True):
    y_hat = y_hat.permute(0,2,1)
    y = y.squeeze(-1)
    loss = F.nll_loss(y_hat, y)
    return loss

def train(args, model, device, train_loader, optimizer, epoch, sigma=1.0):
    model.train()
    criterion = SqueezeWaveLoss(sigma)

    for batch_idx, (m, x) in enumerate(train_loader):
        print(batch_idx)
        x, m = x.to(device), m.to(device)
        outputs = model((m, x))
        loss = criterion(outputs)

        if np.isinf(loss.item()) or np.isnan(loss.item()):
            loss = criterion(outputs)

        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(m), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or run some neural net')
    parser.add_argument('-d', '--data', type=str, default='./data', help='dataset directory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    data_path = args.data

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    torch.autograd.set_detect_anomaly(True)

    with open(os.path.join(data_path, 'train.json'), 'r') as f:
        train_index = json.load(f)

    with open(os.path.join(data_path, 'test.json'), 'r') as f:
        test_index = json.load(f)

    train_loader = torch.utils.data.DataLoader(
        AudiobookDataset(train_index),
        collate_fn=discrete_collate,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        AudiobookDataset(test_index),
        collate_fn=discrete_collate,
        batch_size=1, shuffle=False, **kwargs)

    squeezewave_config = {
        'n_mel_channels': 80,
        'n_flows': 12,
        'n_audio_channel': 128,
        'n_early_every': 2,
        'n_early_size': 16,
        'WN_config': {
            "n_layers": 8,
            "n_channels": 256,
            "kernel_size": 3
        }
    }

    model = SqueezeWave(**squeezewave_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        print(f'epoch {epoch}')
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
