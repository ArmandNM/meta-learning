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
    # torch.backends.cudnn.benchmark = True

    meta_trainer = MetaTrainer()

    if meta_trainer.args.test_only:
        print("Skipping training.")
        # Reinitialize with given seed to make testing deterministic
        meta_trainer = MetaTrainer(test_seed=42)
        checkpoint_name = meta_trainer.args.checkpoint_name if meta_trainer.args.checkpoint_name else "best_checkpoint"
        meta_trainer.test(checkpoint=checkpoint_name)
    else:
        meta_trainer.train(training=True, checkpoint=meta_trainer.args.checkpoint_name)
        # Test using best checkpoint saved
        meta_trainer = MetaTrainer(test_seed=42)
        meta_trainer.test(checkpoint="best_checkpoint")

    # Close summary writer
    meta_trainer.writer.close()


def main():
    print(torch.cuda.is_available())
    print(torchmeta.__version__)

    start_experiment()


if __name__ == '__main__':
    main()
