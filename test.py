import os
import pdb
import sys
import json
import argparse

from argparse import Namespace
from colorama import Fore
from tqdm import tqdm

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
import tensorflow as tf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from dataset import TIMIT_VAD_test_Dataset, vox1_dataset
from models import FCN_WVAD
from utils import rnn_collate
from util import get_filepaths
from vad_eval import vad_evaluate as ev

def make_vad(y_pred):
    yp=[]
    #pdb.set_trace()
    y_pred = y_pred.transpose().squeeze()
    #pdb.set_trace()
    time_label=[]
    frame_label=[]
    length=len(y_pred)
    #pdb.set_trace()
    for i in range(length):
        if y_pred[i,1]>=0.5:#y_pred[i,0]:
            #if i>0 and y_pred[i-1,1]<y_pred[i-1,0]:
            #    time_label.append(str(['speech hit']))
            #time_label.append(str(['speech',str(16*i)+'ms']))
            #frame_label.append(str(['speech', str(i)+'_frame']))
            for k in range(256):
                yp.append([0,3])
            k=0
        else:
            #if i>0 and y_pred[i-1,1]>=y_pred[i-1,0]:
            #    time_label.append(str(['speech end']))
            time_label.append(str(['non_speech',str(16*i)+'ms']))
            frame_label.append(str(['non_speech', str(i)+'_frame']))
            for k in range(256):
                yp.append([1,0])
            k=0
    #pdb.set_trace()
    return np.asarray(yp)[:,1], frame_label

def make_vad_ref(y_pred):
    yp=[]
    #pdb.set_trace()
    y_pred = y_pred.transpose().squeeze()
    #pdb.set_trace()
    time_label=[]
    frame_label=[]
    length=len(y_pred)
    for i in range(length):
        if y_pred[i]==1:
            #if i>0 and y_pred[i-1,1]<y_pred[i-1,0]:
            #    time_label.append(str(['speech hit']))
            #time_label.append(str(['speech',str(16*i)+'ms']))
            #frame_label.append(str(['speech', str(i)+'_frame']))
            for k in range(256):
                yp.append([0,3])
            k=0
        else:
            #if i>0 and y_pred[i-1,1]>=y_pred[i-1,0]:
            #    time_label.append(str(['speech end']))
            time_label.append(str(['non_speech',str(16*i)+'ms']))
            frame_label.append(str(['non_speech', str(i)+'_frame']))
            for k in range(256):
                yp.append([1,0])
            k=0
    #pdb.set_trace()
    return np.asarray(yp)[:,1], frame_label

def draw(vad_predict,vad, raw_wav, Name):
    vpred, time_label = make_vad(vad_predict)
    #pdb.set_trace()

    plt.plot(raw_wav, label=Name)
    plt.plot(vpred, label= "VAD")
    plt.legend(loc=4)
    plt.savefig("./demo/transformer_DNS_fine/"+Name+".png", dpi=150)
    plt.close()
    ''' 
    vad, time_label = make_vad_ref(vad)
    plt.plot(raw_wav, label=Name)
    plt.plot(vad, label= "rVAD_fast")
    plt.legend(loc=4)
    plt.savefig("./demo/FCN_VAD_V1/ref_"+Name+".png", dpi=150)
    plt.close()
    '''

if __name__ == '__main__':
    ckpt_root = sys.argv[1]
    save_root = sys.argv[2]
    hparams_path = os.path.join(ckpt_root, 'hparams.json')
    ckpt_path = os.path.join(ckpt_root, 'model_best.ckpt')

    if not os.path.exists(save_root):
        print(f'Making dir: {save_root}')
        os.makedirs(save_root)
    else:
        op = input('Do you want to overwrite this directory? [y/n]')
        if op == 'y':
            pass
        elif op == 'n':
            print(f'Directory {save_root} already exists. Process terminated')
            sys.exit()
        else:
            print('Invalid answer.')
            print(f'Directory {save_root} already exists. Process terminated')
            sys.exit()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f'DEVICE: [{torch.cuda.current_device()}] {torch.cuda.get_device_name()}')
    else:
        print(f'DEVICE: CPU')
    with open(os.path.join(hparams_path), 'r') as f:
        hparams = json.load(f)
    args = Namespace(**hparams)
    '''TIMIT_SE Corpus
    data_dir = "/data1/user_chengyu/Corpus/TIMIT_SE"
    test_dataloader = DataLoader(
        TIMIT_VAD_test_Dataset(data_dir, test=True),
        # batch_size=args.batch_size,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        #collate_fn=rnn_collate
    )
    '''

    #data_dir = "/mnt/ForteMedia/speaker_ID/VoxCeleb/vox1_test_wav"
    #data_dir = "/data1/user_chengyu/WVAD/temp"
    #data_dir = "/data1/user_chengyu/Forte/forte_DNS_test/Enhanced/transformer"
    #data_dir = "/data1/user_chengyu/Forte/forte_DNS_test/bypass_noisy"
    #data_dir = "/mnt/ForteMedia/Personalized_NS/testvector/dataset2"
    data_dir = "/data1/user_chengyu/Forte/VAD/exp" 
    test_dataloader = DataLoader(
        vox1_dataset(data_dir, test=True),
        # batch_size=args.batch_size,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        #collate_fn=rnn_collate
    )
    
    net = FCN_WVAD() 
    net = nn.DataParallel(net)
    print(f'Resume model from {ckpt_path} ...')
    #pdb.set_trace()
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    #pdb.set_trace()
    net = net.to(device)

    pbar = tqdm(test_dataloader, bar_format='{l_bar}%s{bar}%s{r_bar}'%(Fore.LIGHTMAGENTA_EX, Fore.RESET))
    pbar.set_description('Validation')
    total_loss, total_acc, total_fp, total_fr = 0.0, 0.0, 0.0, 0.0
    num_test_data = len(test_dataloader)
    with torch.no_grad():
        net.eval()
        for i, (n, c, l) in enumerate(pbar):
            c = tf.keras.utils.to_categorical(c, dtype='float32')
            c = torch.from_numpy(c)
            n, c = n.to(device), c.to(device)
            vad = net(n)
            
            # e = (e + 1.) * (c.max() - c.min()) * 0.5 + c.min()
            # assert e.sum() != 0
            # assert e.max() <= 1.
            # assert e.min() >= -1.
            
            filename = test_dataloader.dataset.wav_path[i].split('/')[-1].replace(".wav", ".masked.wav")
            #draw(vad.cpu().numpy(), c.cpu().numpy().squeeze(), n.cpu().numpy().squeeze(), filename)
            #draw(vad.cpu().numpy(), vad.cpu().numpy(), n.cpu().numpy().squeeze(), filename)
            mask, _ = make_vad(vad.cpu().numpy())
            mask = mask/3.0
            e = n*mask
            #vad = vad.transpose(1,2)
            #acc, fp, fr = ev(vad.cpu().numpy().squeeze(), c.cpu().numpy().squeeze())
            #total_acc = total_acc+ acc
            #total_fp = total_fp + fp
            #total_fr = total_fr +fr
            #pdb.set_trace()
            pbar.set_postfix({'File name': filename})
            torchaudio.save(os.path.join(save_root, filename), e[0, 0, :l].cpu(), 16000)
        print(total_acc/(i+1.0), total_fp/(i+1.0), total_fr/(i+1.0))
