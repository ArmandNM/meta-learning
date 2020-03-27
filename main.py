import torch
import higher
import torchmeta

from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

from time import time

from models.metaconv import MetaConv
from models.metaconv_contextual import MetaConvContextual
from learners.meta_trainer import MetaTrainer


def start_experiment():
    # Setting benchmark = True should improve performance for constant shape input
    torch.backends.cudnn.benchmark = True

    meta_trainer = MetaTrainer()
    meta_trainer.train(training=True)
    # Test using best checkpoint saved
    meta_trainer.test(resume=True)
    meta_trainer.writer.close()


def main():
    print(torch.cuda.is_available())
    print(torchmeta.__version__)

    start_experiment()


if __name__ == '__main__':
    main()
