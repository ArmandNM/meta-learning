import torch
import higher
import unittest
import argparse
import copy

from tests.generic_test import GenericTest
from models.metaconv_contextual import MetaConvContextual

from learners.cavia import CAVIA


class TestCAVIA(unittest.TestCase, GenericTest):
    def __init__(self, *args, **kwargs):
        super(TestCAVIA, self).__init__(*args, **kwargs)
        self._initialize_args()
        self._load_batch(n_ways=self.args.n_ways, tasks_num=self.args.tasks_num)

    def _initialize_args(self):
        self.argparser = argparse.ArgumentParser()
        self.argparser.add_argument('--n_ways', type=int, default=5)
        self.argparser.add_argument('--tasks_num', type=int, default=4)
        self.argparser.add_argument('--model', type=str, default='meta_conv_contextual')
        self.argparser.add_argument('--inner_steps_train', type=int, default=2)
        self.argparser.add_argument('--inner_steps_test', type=int, default=2)
        self.args, _ = self.argparser.parse_known_args()

    def test_inner_loop(self):
        # Create new model
        model = MetaConvContextual()
        # Save initial model weights
        original_model = copy.deepcopy(model)

        inner_optimizer = torch.optim.SGD([model.context_params], lr=1e-2)
        inner_optimizer.zero_grad()

        # Make one training iteration
        logits = model(self.inputs)
        loss = torch.nn.functional.cross_entropy(logits, self.labels)
        loss.backward()
        inner_optimizer.step()

        # Check that only context params are updated
        self._equal_parameters(model.parameters(), original_model.parameters(),
                              [original_model.context_params.data.data_ptr()])

    def test_inner_loop_higher(self):
        # Create new model
        model = MetaConvContextual()
        # Save initial model weights
        original_model = copy.deepcopy(model)

        inner_optimizer = torch.optim.SGD([model.context_params], lr=1e-2)
        inner_optimizer.zero_grad()

        # Testing for track_higher_grads = True
        with higher.innerloop_ctx(model, opt=inner_optimizer,
                                  copy_initial_weights=False, track_higher_grads=True) as (fmodel, diffopt):
            logits = fmodel(self.inputs)
            loss = torch.nn.functional.cross_entropy(logits, self.labels)
            diffopt.step(loss)

            # Check that only context params are updated
            self._equal_parameters(fmodel.parameters(), original_model.parameters(),
                                  [original_model.context_params.data.data_ptr()])

        # Testing for track_higher_grads = False
        with higher.innerloop_ctx(model, opt=inner_optimizer,
                                  copy_initial_weights=False, track_higher_grads=False) as (fmodel, diffopt):
            logits = fmodel(self.inputs)
            loss = torch.nn.functional.cross_entropy(logits, self.labels)
            diffopt.step(loss)

            # Check that only context params are updated
            self._equal_parameters(fmodel.parameters(), original_model.parameters(),
                                  [original_model.context_params.data.data_ptr()])

        # Check that original model is unchanged
        self._equal_parameters(model.parameters(), original_model.parameters())

    def test_train_iteration(self):
        learner = CAVIA(self.args)
        super(TestCAVIA, self).test_train_iteration(learner)

    def test_batch_overfit(self):
        model = MetaConvContextual(out_channels=self.args.n_ways)
        super(TestCAVIA, self).test_batch_overfit(model=model, learnable_params=[model.context_params])


if __name__ == '__main__':
    unittest.main()
