import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from mmengine import Config
from mmengine.device import get_device
from mmengine.infer import BaseInferencer
from mmengine.model import BaseModel
from mmengine.registry import INFERENCERS, MODELS
from mmengine.visualization import Visualizer


@MODELS.register_module()
class MMResNet50(BaseModel):

    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        print(imgs[0].size())
        x = self.resnet(imgs[0])
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            print(x.size())
            return x, labels


@INFERENCERS.register_module()
class CustomInferencer(BaseInferencer):

    def _init_visualizer(self, cfg):
        """Return custom visualizer.

        The returned visualizer will be set as ``self.visualzier``.
        """
        if cfg.get('visualizer') is not None:
            visualizer = cfg.visualizer
            visualizer.setdefault('name', 'mmengine_template')
            return Visualizer.get_instance(**cfg.visualizer)
        return Visualizer(name='mmengine_template')

    def _init_pipeline(self, cfg):
        """Return a pipeline to process input data.

        The returned pipeline should be a callable object and will be set as
        ``self.visualizer``

        This default implementation will read the image and convert it to a
        Tensor with shape (C, H, W) and dtype torch.float32. Also, users can
        build the pipeline from the ``cfg``.
        """
        device = get_device()

        def naive_pipeline(image):
            print(type(image))
            #            image = np.float32(imread(image))
            image = np.float32(image)
            image = image.transpose(2, 0, 1)
            print(image.shape)
            image = torch.from_numpy(image).to(device)
            image = image.unsqueeze(0)
            print(image.size())
            return dict(imgs=image, labels=torch.tensor(99))

        return naive_pipeline

    def visualize(self, inputs, preds, show=False):
        """Visualize the predictions on the original inputs."""
        visualization = []
        for image_path, pred in zip(inputs, preds):
            #            image = imread(image_path)
            image = image_path
            self.visualizer.set_image(image)
            # NOTE The implementation of visualization is left to the user.
            ...
            if show:
                self.visualizer.show()
            vis_result = self.visualizer.get_image()
            # Return the visualization for post process.
            visualization.append(dict(image=vis_result, filename='xxx'))
        return visualization

    def postprocess(self, preds, visualization, return_datasample=False):
        """Apply post process to the predictions and visualization.

        For example, you can save the predictions or visualization to files in
        this method.

        Note:
            The parameter ``return_datasample`` should only be used when
            ``model.forward`` output a list of datasample instance.
        """
        ...
        return dict(predictions=preds, visualization=visualization)


def unpickle(file):
    import pickle
    with open(file, 'rb') as f:
        _dict = pickle.load(f, encoding='bytes')
    res = _dict[b'data'][510]
    res = res.reshape(32, 32, 3)
    return res


cfg = Config(dict(model=dict(type='MMResNet50')))
weight = './work_dir/epoch_5.pth'
inferencer = CustomInferencer(model=cfg, weights=weight)
img = unpickle('/data/public/cifar/cifar-10-batches-py/test_batch')
print(type(img), img.shape)

result = inferencer(img)
print(type(result))
print(result.keys())
print(result['predictions'][0].size())
print(torch.argmax(result['predictions'][0]), result['predictions'][1])
print(result['predictions'][1][0].size())
