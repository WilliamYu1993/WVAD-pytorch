import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchcontrib.optim import SWA
from torch.nn.utils import rnn, clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import pdb
import json
import argparse
import tensorflow as tf
from argparse import Namespace
import numpy as np
from vad_eval import vad_evaluate as ev
from tqdm import tqdm
from colorama import Fore
from collections import OrderedDict
from multiprocessing import Pool

from dataset import VoiceBankDemandDataset_VAD 
from models import FCN_WVAD
from loss import categorical_crossentropy 

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # system setting
    parser.add_argument('--exp_dir', default=os.getcwd(), type=str)
    parser.add_argument('--exp_name', default='logs', type=str)
    parser.add_argument('--data_dir', default='/mnt/md2/user_chengyu/Corpus/TIMIT_SE/', type=str)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--add_graph', action='store_true')
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--seed', default=0, type=int)
    
    # training specifics
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--clip_grad_norm_val', default=0.0, type=float)
    parser.add_argument('--grad_accumulate_batches', default=1, type=int)
    parser.add_argument('--log_grad_norm', action='store_true')
    parser.add_argument('--resume_dir', default='', type=str)
    parser.add_argument('--use_swa', action='store_true')
    parser.add_argument('--use_logstftmagloss', action='store_true')
    parser.add_argument('--lr_decay', default=1.0, type=float)

    args = parser.parse_args()
    
    # add hyperparameters
    ckpt_path = os.path.join(args.exp_dir, args.exp_name, 'ckpt')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
        os.makedirs(ckpt_path.replace('ckpt', 'logs'))
        with open(os.path.join(ckpt_path, 'hparams.json'), 'w') as f:
            json.dump(vars(args), f)
    else:
        print(f'Experiment {args.exp_name} already exists.')
        sys.exit()
    writer = SummaryWriter(os.path.join(args.exp_dir, args.exp_name, 'logs'))
    writer.add_hparams(vars(args), dict())

    # seed
    if args.seed:
        fix_seed(args.seed)

    # device
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    if device == 'cuda':
        print(f'DEVICE: [{torch.cuda.current_device()}] {torch.cuda.get_device_name()}')
    else:
        print(f'DEVICE: CPU')

    # create loaders
    train_vad_dataloader = DataLoader(
        VoiceBankDemandDataset_VAD(data_dir=args.data_dir, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        #collate_fn=rnn_collate
    )

    test_dataloader = DataLoader(
        VoiceBankDemandDataset_VAD(data_dir=args.data_dir, train=False),
        # batch_size=args.batch_size,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        #collate_fn=rnn_collate
    )


    # create model
    if not args.resume_dir:
        net = FCN_WVAD()
    else:
        try:
            with open(os.path.join(args.resume_dir, 'hparams.json'), 'r') as f:
                hparams = json.load(f)
        except FileNotFoundError:
            print('Cannot find "hparams.json".')
            sys.exit()

        hparams['resume_dir'] = args.resume_dir
        args = Namespace(**hparams)
        net = FCN_WVAD()
        model_path = os.path.join(args.resume_dir, 'model_best.ckpt')
        print(f'Resume model from {model_path} ...')
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
    
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=0.1)
    scheduler = None
    if args.use_swa:
        steps_per_epoch = len(train_vad_dataloader) // args.batch_size
        optimizer = SWA(optimizer, swa_start=20 * steps_per_epoch, swa_freq=steps_per_epoch)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer.optimizer, mode="max", patience=5, factor=0.5)

    else:
        scheduler = None

    if args.resume_dir:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dcf = checkpoit['dcf']
    else:
        start_epoch = 0
        best_dcf = 0.0

    # add graph to tensorboard
    if args.add_graph:
        dummy = torch.randn(16, 1, args.hop_length * 16).to(device)
        writer.add_graph(net, dummy)

    criterion = categorical_crossentropy()
    
    total_loss = 0.0 
    for epoch in range(start_epoch, start_epoch + args.num_epochs, 1):
        # ------------- training ------------- 
        net.train()
        pbar = tqdm(train_vad_dataloader, bar_format='{l_bar}%s{bar}%s{r_bar}'%(Fore.BLUE, Fore.RESET))
        pbar.set_description(f'Epoch {epoch + 1}')
        #total_loss = 0.0
        if args.log_grad_norm:
            total_norm = 0.0
        net.zero_grad()
        for i, (n, c) in enumerate(pbar):
            c = tf.keras.utils.to_categorical(c, dtype='float32')
            c = torch.from_numpy(c)
            n, c = n.to(device), c.to(device)
            e = net(n).transpose(1,2)
            loss = criterion(c.long(), e)
            loss /= args.grad_accumulate_batches
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            if args.clip_grad_norm_val > 0.0:
                clip_grad_norm_(net.parameters(), args.clip_grad_norm_val)

            # log metrics
            pbar_dict = OrderedDict({
                'loss': loss.item(),
            })
            pbar.set_postfix(pbar_dict)

            total_loss += loss.item()
            if (i + 1) % args.log_interval == 0:
                step = epoch * len(train_vad_dataloader) + i
                writer.add_scalar('Loss/train', total_loss / args.log_interval, step)
                total_loss = 0.0

                # log gradient norm
                if args.log_grad_norm:
                    for p in net.parameters():
                        if p.requires_grad:
                            norm = p.grad.data.norm(2)
                            total_norm += norm.item() ** 2
                    norm = total_norm ** 0.5
                    writer.add_scalar('Gradient 2-Norm/train', norm, step)
                    total_norm = 0.0

            # accumulate gradients
            if (i + 1) % args.grad_accumulate_batches == 0:
                optimizer.step()
                net.zero_grad()

        # ------------- validation -------------
        pbar = tqdm(test_dataloader, bar_format='{l_bar}%s{bar}%s{r_bar}'%(Fore.LIGHTMAGENTA_EX, Fore.RESET))
        pbar.set_description('Validation')
        #total_loss, total_pesq = 0.0, 0.0
        total_loss, total_dcf = 0.0, 0.0
        num_test_data = len(test_dataloader)
        with torch.no_grad():
            net.eval()
            for i, (n, c, l) in enumerate(pbar):
                c = tf.keras.utils.to_categorical(c, dtype='float32')
                c = torch.from_numpy(c)
                n, c = n.to(device), c.long().to(device)
                e = net(n).transpose(1,2)
                loss = criterion(c.long(), e)
                acc, fp, fr = ev(e.cpu().numpy().squeeze(), c.cpu().numpy().squeeze())
                vad_score = 0.5*acc-0.5*fr 
                pbar_dict = OrderedDict({
                    'val_loss': loss.item(),
                    'val_dcf': vad_score,
                })
                pbar.set_postfix(pbar_dict)

                total_loss += loss.item()
                total_dcf += vad_score

            if scheduler is not None:
                scheduler.step(total_dcf / num_test_data)

            writer.add_scalar('Loss/valid', total_loss / num_test_data, epoch)
            writer.add_scalar('DCF/valid', total_dcf / num_test_data, epoch)

            curr_dcf = total_dcf / num_test_data
            if  curr_dcf > best_dcf:
                best_dcf = curr_dcf
                save_path = os.path.join(ckpt_path, 'model_best.ckpt')
                print(f'Saving checkpoint to {save_path}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss / num_test_data,
                    'dcf': total_dcf / num_test_data
                }, save_path)
    writer.flush()
    writer.close()
