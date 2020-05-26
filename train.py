import os
import json
import argparse
import torch
import torch.optim as optim
from glow import SqueezeWave, SqueezeWaveLoss
from dataset import AudiobookDataset
from dataset import train_collate
from dataset import test_collate
from utils.dsp import save_wav
import numpy as np

#from mel2samp import Mel2Samp

def save_checkpoint(device, model, optimizer, checkpoint_dir, epoch):
    checkpoint_path = os.path.join(
        checkpoint_dir, "checkpoint_step{:06d}.pth".format(epoch))
    optimizer_state = optimizer.state_dict()
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "epoch": epoch
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def train(args, model, device, train_loader, optimizer, epoch, sigma=1.0):
    model.train()
    criterion = SqueezeWaveLoss(sigma)

    for batch_idx, (m, x, _) in enumerate(train_loader):
        x, m = x.to(device), m.to(device)
        
        model.zero_grad()
        
        outputs = model((m, x))
        loss = criterion(outputs)

        if np.isinf(loss.item()) or np.isnan(loss.item()):
            loss = criterion(outputs)

        loss.backward()
        optimizer.step()

        #if batch_idx % 10 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(m), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, checkpoint_dir, epoch, sigma=1.0):
    model.eval()
    
    sample_dir = os.path.join(checkpoint_dir, 'sample_{0:06d}'.format(epoch))
    os.makedirs(sample_dir, exist_ok=True)
    
    with torch.no_grad():
        for m, _, fname in test_loader:
            m = m.to(device)
            audio = model.infer(m, sigma=sigma).float()
            audio = audio.cpu().numpy()
            audio[np.where(np.isfinite(audio)==False)] = 0
            for f, a in zip(fname, audio):
                target_path = os.path.join(sample_dir, os.path.basename(f))
                save_wav(a, target_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or run some neural net')
    parser.add_argument('-d', '--data', type=str, default='./data', help='dataset directory')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--batch-size', type=int, default=96, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=4e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    data_path = args.data

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    torch.autograd.set_detect_anomaly(True)
    
    data_config = {
        "training_files": "train_files.txt",
        "segment_length": 16384,
        "sampling_rate": 22050,
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0
    }
    
    #trainset = Mel2Samp(128, **data_config)
    #train_loader = torch.utils.data.DataLoader(trainset, num_workers=0, shuffle=False,
    #                                           batch_size=args.batch_size,
    #                                           pin_memory=False,
    #                                           drop_last=True)

    with open(os.path.join(data_path, 'train.json'), 'r') as f:
        train_index = json.load(f)

    with open(os.path.join(data_path, 'test.json'), 'r') as f:
        test_index = json.load(f)

    train_loader = torch.utils.data.DataLoader(
        AudiobookDataset(train_index),
        collate_fn=train_collate,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        AudiobookDataset(test_index),
        collate_fn=test_collate,
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
    
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f'epoch {epoch}')
        train(args, model, device, train_loader, optimizer, epoch)

        if epoch % 10 == 0:
            test(model, device, test_loader, checkpoint_dir, epoch)
            save_checkpoint(device, model, optimizer, checkpoint_dir, epoch)
