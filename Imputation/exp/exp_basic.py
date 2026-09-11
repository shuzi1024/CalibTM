import os
import torch
from models import ARI_LLM


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'ARI-LLM': ARI_LLM
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)


    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            existing_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                device_index = self.args.gpu
            elif existing_visible:
                device_index = 0
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                device_index = 0
            device = torch.device('cuda:{}'.format(device_index))
            print('Use GPU: cuda:{} visible={}'.format(device_index, os.environ.get("CUDA_VISIBLE_DEVICES", "")))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
