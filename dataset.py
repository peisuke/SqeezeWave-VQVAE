import os
import random
import json
import torch
import torch.utils.data
import numpy as np
from hparams import hparams as hp

class AudiobookDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, train=False):
        self.path = input_path

    def __getitem__(self, index):
        p = self.path[index]
        m = np.load(p['mel'])
        x = np.load(p['quant'])
        return m, x

    def __len__(self):
        return len(self.path)

def discrete_collate(batch):
    pad = 0
    mel_win = hp.seq_len // hp.hop_length + 2 * pad

    max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + pad) * hp.hop_length for offset in mel_offsets]
    
    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] \
            for i, x in enumerate(batch)]
    
    wav = [x[1][sig_offsets[i]:sig_offsets[i] + hp.seq_len] \
              for i, x in enumerate(batch)]
    
    mels = torch.FloatTensor(mels)
    wav = torch.LongTensor(wav)
    wav = 2 * wav[:, :hp.seq_len].float() / (2**hp.bits - 1.) - 1.

    return mels, wav
