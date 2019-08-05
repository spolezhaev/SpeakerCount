import torch

def accuracy(output, target, percent=0.1):
    with torch.no_grad():

        assert output.shape[0] == len(target)
        preds = torch.argmax(output,dim=1)
        tp = 0
        tp = torch.sum(preds == target).item()

    return tp / len(target)