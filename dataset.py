import os
import random
import json
import torch
import torch.utils.data
import numpy as np
from hparams import hparams as hp

from utils.dsp import load_wav
from utils.dsp import melspectrogram

class AudiobookDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, train=False):
        self.path = input_path

    def __getitem__(self, index):
        p = self.path[index]
        #m = np.load(p['mel'])
        #x = np.load(p['quant'])
        f = p['file']
        
        wav = load_wav(f)
        #mel = melspectrogram(wav)
        #quant = wav * (2**15 - 0.5) - 0.5
        #return mel.astype(np.float32), quant.astype(np.int16)        
           
        return wav, f

    def __len__(self):
        return len(self.path)

def train_collate(batch):
    mel_win = hp.seq_len // hp.hop_length
    #max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
    #mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    #sig_offsets = [(offset + pad) * hp.hop_length for offset in mel_offsets]
    
    max_offsets = [x[0].shape[-1] - hp.seq_len for x in batch]
    sig_offsets = [np.random.randint(0, offset) for offset in max_offsets]
        
    #mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] \
    #        for i, x in enumerate(batch)]

    wav = [x[0][sig_offsets[i]:sig_offsets[i] + hp.seq_len] \
              for i, x in enumerate(batch)]
    
    mels = [melspectrogram(w[:-1]) for w in wav]

    fname = [x[1] for x in batch]

    mels = torch.FloatTensor(mels)
    wav = torch.FloatTensor(wav)
    
    #wav = 2 * wav[:, :hp.seq_len].float() / (2**hp.bits - 1.) - 1.
    
    return mels, wav, fname

def test_collate(batch):
    wav = [x[0] for i, x in enumerate(batch)]
    mels = [melspectrogram(w) for w in wav]

    fname = [x[1] for x in batch]

    mels = torch.FloatTensor(mels)
    wav = torch.FloatTensor(wav)
    
    return mels, wav, fname