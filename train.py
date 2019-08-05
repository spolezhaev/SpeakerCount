import torch
from torchaudio_contrib import Melspectrogram
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import datetime
import math

from utils import eval_metrics, valid_epoch, pad_seq, mkdir_p

from model import AudioCRNN

from metrics import accuracy

import librosa

import pandas as pd
import numpy as np

from pathlib import Path

import json

from net_config.audio import MelspectrogramStretch

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from tqdm.autonotebook import tqdm

import logging


from visualization import WriterTensorboardX

from transforms import AudioTransforms
from SoundSet import SoundSet

import os, errno


def save_checkpoint(epoch, save_best=False):
    """
    Saving checkpoints

    :param epoch: current epoch number
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
    """
    arch = type(model).__name__
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'monitor_best': mnt_best,
        'classes':model.classes
    }

    filename = os.path.join(checkpoint_dir, 'checkpoint-current.pth')
    torch.save(state, filename)
    logger.info("Saving checkpoint: {} ...".format(filename))
    if save_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_path)
        logger.info("Saving current best: {} ...".format('model_best.pth'))
        logger.info("[IMPROVED]")



batch_size = 64
dataloader = DataLoader(
        SoundSet(mode="train", transform=AudioTransforms("train", {"noise":[0.3, 0.001], "crop":[0.4, 0.25]})),
        batch_size=batch_size,
        shuffle=True, 
        num_workers=0,
        collate_fn=pad_seq)
test_dataloader = DataLoader(
        SoundSet(mode="test", transform=AudioTransforms("val", {"noise":[0.3, 0.001], "crop":[0.4, 0.25]})),
        batch_size=batch_size,
        shuffle=True, 
        num_workers=0,
        collate_fn=pad_seq)

model = AudioCRNN(classes=[0, 1, 2, 3]).cuda()

loss_fn = nn.NLLLoss()

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.01, amsgrad=True)

lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

n_epochs = 150  # suggest training between 20-50 epochs

start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
checkpoint_dir = os.path.join("saved_cv", start_time, 'checkpoints')
log_dir = os.path.join("saved_cv", start_time, 'logs')

logger = logging.getLogger("SuperLogger")


writer = WriterTensorboardX(log_dir, logger, True)

# Save configuration file into checkpoint directory:
mkdir_p(checkpoint_dir)

mnt_mode, mnt_metric = "min", "val_loss"

mnt_best = math.inf if mnt_mode == 'min' else -math.inf
early_stop = 40

model.train() # prep model for training

metrics = [accuracy]

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    total_metrics = np.zeros(len(metrics))
    writer.set_step(epoch) 
        
    ###################
    # train the model #
    ###################
    _trange = tqdm(dataloader, leave=True, desc='')

    for batch_idx, batch in enumerate(_trange):
        batch = [b.to("cuda") for b in batch]
        data, target = batch[:-1], batch[-1]
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = loss_fn.forward(output, target)
        
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()
        total_metrics += eval_metrics(output, target, metrics, writer)
        
        if batch_idx % int(np.sqrt(dataloader.batch_size)) == 0:                
                _str = 'Train Epoch: {} Loss: {:.6f}'.format(epoch,loss.item()) 
                _trange.set_description(_str)
        
    # print training statistics 
    # calculate average loss over an epoch
    # Add epoch metrics
    loss = train_loss / len(dataloader)
    metric_epoch = (total_metrics / len(dataloader)).tolist()

    writer.add_scalar('loss', loss)
    for i, metric in enumerate(metrics):
        writer.add_scalar("%s"%metric.__name__, metric_epoch[i])

    log = {
        'loss': loss,
        'metrics': metric_epoch
    }
    
    print('train')
    print(log)
    print('test')
    
    val_log = valid_epoch(epoch, model, metrics, writer, test_dataloader, loss_fn)
    print(val_log)
    log = {**log, **val_log}

    if lr_scheduler is not None:
        lr_scheduler.step()
    
    c_lr = optimizer.param_groups[0]['lr']
        
    
    tot_log = {'epoch': epoch}
    for key, value in log.items():
        if key == 'metrics':
            tot_log.update({mtr.__name__ : value[i] for i, mtr in enumerate(metrics)})
        elif key == 'val_metrics':
            tot_log.update({'val_' + mtr.__name__ : value[i] for i, mtr in enumerate(metrics)})
        else:
            tot_log[key] = value
            
    writer.add_scalar('lr', c_lr)
    
    
    
    
    best = False
    try:
        # check whether model performance improved or not, according to specified metric(mnt_metric)
        improved = (mnt_mode == 'min' and tot_log[mnt_metric] < mnt_best) or                    (mnt_mode == 'max' and tot_log[mnt_metric] > mnt_best)
    except KeyError:
        logger.warning("Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(mnt_metric))
        mnt_mode = 'off'
        improved = False
        not_improved_count = 0

    if improved:
        mnt_best = tot_log[mnt_metric]
        not_improved_count = 0
        best = True
    else:
        not_improved_count += 1

    if not_improved_count > early_stop:
        logger.info("Validation performance didn\'t improve for {} epochs. Training stops.".format(early_stop))
        break

    if len(writer) > 0:
        logger.info(
            '\nRun TensorboardX:\ntensorboard --logdir={}\n'.format(log_dir))

    if epoch % 1 == 0:
        save_checkpoint(epoch, save_best=best)