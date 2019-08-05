import torch
from torch import nn

import torch.nn.functional as F

from net_config.audio import MelspectrogramStretch


class AudioCRNN(nn.Module):
    def __init__(self, classes, state_dict=None):
        super(AudioCRNN, self).__init__()
        
        in_chan = 1

        self.classes = classes
        self.lstm_units = 64
        self.lstm_layers = 2
        self.spec = MelspectrogramStretch(hop_length=None, 
                                num_mels=128, 
                                fft_length=2048, 
                                norm='whiten', 
                                stretch_param=[0.4, 0.4])

        # shape -> (channel, freq, token_time)
        self.net = nn.ModuleDict({
            'convs' : nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ELU(alpha=1.0),
                nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False),
                nn.Dropout(p=0.1),
                nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ELU(alpha=1.0),
                nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False),
                nn.Dropout(p=0.1),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ELU(alpha=1.0),
                nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False),
                nn.Dropout(p=0.1)
            ),
            'recur' : nn.LSTM(128, 64, num_layers=2),
            'dense' : nn.Sequential(
                nn.Dropout(p=0.3),
                nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Linear(in_features=64, out_features=4, bias=True)
            )
        })
        #self.net = parse_cfg(config['cfg'], in_shape=[in_chan, self.spec.num_mels, 400])

    def _many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths - 1]

    def modify_lengths(self, lengths):
        def safe_param(elem):
            return elem if isinstance(elem, int) else elem[0]
        
        for name, layer in self.net['convs'].named_children():
            #if name.startswith(('conv2d','maxpool2d')):
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                p, k, s = map(safe_param, [layer.padding, layer.kernel_size,layer.stride]) 
                lengths = (lengths + 2*p - k)//s + 1

        return torch.where(lengths > 0, lengths, torch.tensor(1, device=lengths.device))

    def forward(self, batch):    
        # x-> (batch, time, channel)
        #print(batch)
        x, lengths, _ = batch # unpacking seqs, lengths and srs

        # x-> (batch, channel, time)
        xt = x.float().transpose(1,2)
        # xt -> (batch, channel, freq, time)
        xt, lengths = self.spec(xt, lengths)                

        # (batch, channel, freq, time)
        xt = self.net['convs'](xt)
        lengths = self.modify_lengths(lengths)

        # xt -> (batch, time, freq, channel)
        x = xt.transpose(1, -1)

        # xt -> (batch, time, channel*freq)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
    
        # x -> (batch, time, lstm_out)
        x_pack, hidden = self.net['recur'](x_pack)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        
        # (batch, lstm_out)
        x = self._many_to_one(x, lengths)
        # (batch, classes)
        x = self.net['dense'](x)

        x = F.log_softmax(x, dim=1)

        return x

    def predict(self, x):
        with torch.no_grad():
            out_raw = self.forward( x )
            out = torch.exp(out_raw)
            max_ind = out.argmax().item()        
            return self.classes[max_ind], out[:,max_ind].item()