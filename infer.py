import os
from net_config.audio import MelspectrogramStretch
from transforms import AudioTransforms
from model import AudioCRNN
import librosa
import torch
import argparse

def accuracy(output, target, percent=0.1):
    with torch.no_grad():

        assert output.shape[0] == len(target)
        preds = torch.argmax(output,dim=1)
        tp = 0
        tp = torch.sum(preds == target).item()

    return tp / len(target)


def _get_model_att(checkpoint):
    m_name = 'AudioRCNN'
    sd = checkpoint['state_dict']
    classes = checkpoint['classes']
    return m_name, sd, classes


class AudioInference:

    def __init__(self, model, transforms):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.transform = transforms

        self.mel = MelspectrogramStretch(norm='db')
        self.mel.eval()

    def infer(self, path):
        data, sr = librosa.load(str(path))
        data = data.reshape(-1, 1)   
        
        if self.transform is not None:
            sig_t = self.transform.apply(data)

        length = torch.tensor(sig_t.size(0))
        sr = torch.tensor(sr)
        data = [d.unsqueeze(0).to(self.device) for d in [sig_t, length, sr]]
        label, conf = self.model.predict( data )

        return label, conf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', metavar='N', type=str, nargs='+',
                   help='an integer for the accumulator')
    args = parser.parse_args()
    checkpoint = torch.load('saved_cv/0802_134545/checkpoints/model_best.pth')
    batch_size = 64
    transform = AudioTransforms("val", {"noise":[0.3, 0.001], "crop":[0.4, 0.25]})

    m_name, sd, classes = _get_model_att(checkpoint)

    model = AudioCRNN(classes)

    model.load_state_dict(checkpoint['state_dict'])

    inference = AudioInference(model, transforms=transform)
    label, conf = inference.infer(args.filepath[0])
    print("Number of speakers: {}, confidence: {}".format(label+1, conf)) #as labels start from 0


