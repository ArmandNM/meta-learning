import torch
import higher
import unittest
import argparse
import copy

from tests.generic_test import GenericTest
from models.metaconv_support import MetaConvSupport

from learners.grifon import GRIFON


class TestGRIFON(unittest.TestCase, GenericTest):
    def __init__(self, *args, **kwargs):
        super(TestGRIFON, self).__init__(*args, **kwargs)
        self._initialize_args()
        self._load_batch(n_ways=self.args.n_ways, tasks_num=self.args.tasks_num)

    def _initialize_args(self):
        self.argparser = argparse.ArgumentParser()
        self.argparser.add_argument('--n_ways', type=int, default=5)
        self.argparser.add_argument('--k_spt', type=int, default=1)
        self.argparser.add_argument('--tasks_num', type=int, default=4)
        self.argparser.add_argument('--model', type=str, default='meta_conv_support')
        self.argparser.add_argument('--inner_steps_train', type=int, default=2)
        self.argparser.add_argument('--inner_steps_test', type=int, default=2)
        self.args, _ = self.argparser.parse_known_args()

    def test_inner_loop(self):
        # Create new model
        model = MetaConvSupport()

        # Compute support embeddings
        model.set_support_params(self.inputs)

        # Save initial model weights
        original_model = copy.deepcopy(model)

        inner_optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2)
        inner_optimizer.zero_grad()

        # Make one training iteration
        logits = model(self.inputs)
        loss = torch.nn.functional.cross_entropy(logits, self.labels)
        loss.backward()
        inner_optimizer.step()

        # Check that only film fc layer is updated
        exceptions = list(map(lambda p: p.data_ptr(), original_model.fc.parameters()))
        self._equal_parameters(model.parameters(), original_model.parameters(), exceptions)

    def test_inner_loop_higher(self):
        # Create new model
        model = MetaConvSupport()

        # Compute support embeddings
        model.set_support_params(self.inputs)

        # Save initial model weights
        original_model = copy.deepcopy(model)

        inner_optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2)
        inner_optimizer.zero_grad()

        # Testing for track_higher_grads = True
        with higher.innerloop_ctx(model, opt=inner_optimizer,
                                  copy_initial_weights=False, track_higher_grads=True) as (fmodel, diffopt):
            logits = fmodel(self.inputs)
            loss = torch.nn.functional.cross_entropy(logits, self.labels)
            diffopt.step(loss)

            # Check that only film fc layer is updated
            exceptions = list(map(lambda p: p.data_ptr(), original_model.fc.parameters()))
            self._equal_parameters(fmodel.parameters(), original_model.parameters(), exceptions)

        # Testing for track_higher_grads = False
        with higher.innerloop_ctx(model, opt=inner_optimizer,
                                  copy_initial_weights=False, track_higher_grads=False) as (fmodel, diffopt):
            logits = fmodel(self.inputs)
            loss = torch.nn.functional.cross_entropy(logits, self.labels)
            diffopt.step(loss)

            # Check that only film fc layer is updated
            exceptions = list(map(lambda p: p.data_ptr(), original_model.fc.parameters()))
            self._equal_parameters(fmodel.parameters(), original_model.parameters(), exceptions)

        # Check that original model is unchanged
        self._equal_parameters(model.parameters(), original_model.parameters())

    def test_train_iteration(self):
        learner = GRIFON(self.args)
        super(TestGRIFON, self).test_train_iteration(learner)

    def test_batch_overfit(self):
        model = MetaConvSupport(out_channels=self.args.n_ways)
        model.set_support_params(self.inputs)
        super(TestGRIFON, self).test_batch_overfit(model=model, learnable_params=model.fc.parameters())


if __name__ == '__main__':
    unittest.main()
