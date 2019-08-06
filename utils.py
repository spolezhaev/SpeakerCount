import torch
import os, errno
import numpy as np

def pad_seq(batch):
    sort_ind = 0
    sorted_batch = sorted(batch, key=lambda x: x[0].size(sort_ind), reverse=True)
    seqs, srs, labels = zip(*sorted_batch)
    
    lengths, srs, labels = map(torch.LongTensor, [[x.size(sort_ind) for x in seqs], srs, labels])

    # seqs_pad -> (batch, time, channel) 
    seqs_pad = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    return seqs_pad, lengths, srs, labels

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def list_dir(path):
    filter_dir = lambda x: os.path.isdir(os.path.join(path,x))
    filter_file = lambda x: os.path.isfile(os.path.join(path,x)) and not x.startswith('.')     and not x.split('.')[-1] in ['pyc', 'py','txt']

    ret = [n for n in os.listdir(path) if filter_dir(n) or filter_file(n)]
    
    return ret


def eval_metrics(output, target, metrics, writer):
    acc_metrics = np.zeros(len(metrics))
    for i, metric in enumerate(metrics):
        #import pdb;pdb.set_trace()
        acc_metrics[i] += metric(output, target)

        writer.add_scalar("%s"%metric.__name__, acc_metrics[i])
        writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
    return acc_metrics


def valid_epoch(epoch, model, metrics, writer, test_dataloader, loss_fn):
    model.eval()
    total_val_loss = 0
    total_val_metrics = np.zeros(len(metrics))


    writer.set_step(epoch, 'valid')        

    with torch.no_grad():

        for batch_idx, batch in enumerate(test_dataloader):
            batch = [b.to("cuda") for b in batch]
            data, target = batch[:-1], batch[-1]
            data = data if len(data) > 1 else data[0] 

            output = model(data)
            loss = loss_fn.forward(output, target)

            total_val_loss += loss.item()
            total_val_metrics += eval_metrics(output, target, metrics, writer)

        # Add epoch metrics
        val_loss = total_val_loss / len(test_dataloader)
        val_metrics = (total_val_metrics / len(test_dataloader)).tolist()
        
        writer.add_scalar('loss', val_loss)
        for i, metric in enumerate(metrics):
            writer.add_scalar("%s"%metric.__name__, val_metrics[i])

    model.train()
    return {
        'val_loss': val_loss,
        'val_metrics':val_metrics
        }


