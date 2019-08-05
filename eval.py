import os
import torch
from tqdm import tqdm

from utils import pad_seq

from net_config.audio import MelspectrogramStretch
#from utils import plot_heatmap, mkdir_p
from torch.utils.data import DataLoader
from transforms import AudioTransforms

from SoundSet import SoundSet


from model import AudioCRNN


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

class ClassificationEvaluator(object):

    def __init__(self, data_loader, model):

        self.data_loader = data_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.mel = MelspectrogramStretch(norm='db').to(self.device)

    def evaluate(self, metrics, debug=False):
        with torch.no_grad():
            total_metrics = torch.zeros(len(metrics))
            for batch_idx, batch in enumerate(tqdm(self.data_loader)):
                batch = [b.to(self.device) for b in batch]
                data, target = batch[:-1], batch[-1]
                
                output = self.model(data)

                self.model.classes
                batch_size = data[0].size(0)
                
                if debug:
                    self._store_batch(data, batch_size, output, target)
                
                for i, metric in enumerate(metrics):
                    total_metrics[i] += metric(output, target) * batch_size

            size = len(self.data_loader.dataset)
            ret = {met.__name__ : "%.3f"%(total_metrics[i].item() / size) for i, met in enumerate(metrics)}
            return ret




    def _store_batch(self, data, batch_size, output, target):

        path = 'eval_batch'
        mkdir_p(path)
        sig, lengths, _ = data

        inds = output.argmax(1)
        confs = torch.exp(output)[torch.arange(batch_size), inds]

        spec, lengths = self.mel(sig.transpose(1,2).float(), lengths)
        
        for i in range(batch_size):
            if inds[i] == target[i]:
                label = self.model.classes[inds[i]]
                pred_txt = "%s (%.1f%%)"%(label, 100*confs[inds[i]])
                out_path = os.path.join(path, '%s.png'%i)      
                plot_heatmap(spec[i][...,:lengths[i]].cpu().numpy(), 
                            out_path, 
                            pred=pred_txt)


checkpoint = torch.load('saved_cv/0802_134545/checkpoints/model_best.pth')
#data_config = config['data']

#tsf = _get_transform(config, 'val')

#data_manager = getattr(data_module, config['data']['type'])(config['data'])
batch_size = 64
test_loader = DataLoader(
        SoundSet(mode="test", transform=AudioTransforms("val", {"noise":[0.3, 0.001], "crop":[0.4, 0.25]})),
        batch_size=batch_size,
        shuffle=True, 
        num_workers=0,
        collate_fn=pad_seq)

m_name, sd, classes = _get_model_att(checkpoint)
model = AudioCRNN(classes)


print(model)

model.load_state_dict(checkpoint['state_dict'])

num_classes = len(classes)
metrics = [accuracy]

evaluation = ClassificationEvaluator(test_loader, model)
ret = evaluation.evaluate(metrics)
print(ret)