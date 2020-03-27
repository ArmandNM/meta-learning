import torch
import higher
import unittest
import argparse
import copy

from models.metaconv_contextual import MetaConvContextual
from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

from learners.cavia import CAVIA


class TestCAVIA(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCAVIA, self).__init__(*args, **kwargs)
        self._initialize_args()
        self._load_batch(n_ways=self.args.n_ways, tasks_num=self.args.tasks_num)

    def _load_batch(self, n_ways=5, k_spt=1, k_qry=15, tasks_num=4):
        train_dataset = miniimagenet('../datasets', ways=n_ways, shots=k_spt, test_shots=k_qry,
                                     meta_train=True, download=True)
        self.dataloader = BatchMetaDataLoader(train_dataset, batch_size=tasks_num, num_workers=4)
        for meta_batch in self.dataloader:
            self.meta_batch = meta_batch
            meta_train_inputs, meta_train_labels = meta_batch["train"]
            self.inputs = meta_train_inputs[0]
            self.labels = meta_train_labels[0]
            return

    def _initialize_args(self):
        self.argparser = argparse.ArgumentParser()
        self.argparser.add_argument('--n_ways', type=int, default=5)
        self.argparser.add_argument('--tasks_num', type=int, default=4)
        self.argparser.add_argument('--model', type=str, default='meta_conv_contextual')
        self.argparser.add_argument('--inner_steps_train', type=int, default=2)
        self.argparser.add_argument('--inner_steps_test', type=int, default=2)
        self.args, _ = self.argparser.parse_known_args()

    def _eqal_parameters(self, params1, params2, exceptions=None):
        for param1, param2 in zip(params1, params2):
            if exceptions is not None and param2.data_ptr() in exceptions:
                self.assertTrue(param1.data.ne(param2.data).sum() > 0)
                continue

            self.assertTrue(param1.data.ne(param2.data).sum() == 0)

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
        self._eqal_parameters(model.parameters(), original_model.parameters(),
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
            self._eqal_parameters(fmodel.parameters(), original_model.parameters(),
                                  [original_model.context_params.data.data_ptr()])

        # Testing for track_higher_grads = False
        with higher.innerloop_ctx(model, opt=inner_optimizer,
                                  copy_initial_weights=False, track_higher_grads=False) as (fmodel, diffopt):
            logits = fmodel(self.inputs)
            loss = torch.nn.functional.cross_entropy(logits, self.labels)
            diffopt.step(loss)

            # Check that only context params are updated
            self._eqal_parameters(fmodel.parameters(), original_model.parameters(),
                                  [original_model.context_params.data.data_ptr()])

        # Check that original model is unchanged
        self._eqal_parameters(model.parameters(), original_model.parameters())

    def test_train_iteration(self):
        learner = CAVIA(self.args)
        original_learner = copy.deepcopy(learner)
        optimizer = torch.optim.Adam(params=learner.get_outer_trainable_params(), lr=1e-3)

        # Run one meta-training iteration
        optimizer.zero_grad()
        _, _ = learner.run_iteration(self.meta_batch, training=True)
        optimizer.step()

        # Check that outer trainable params changed
        exceptions = list(map(lambda p: p.data_ptr(), original_learner.get_outer_trainable_params()))
        self._eqal_parameters(learner.model.parameters(), original_learner.model.parameters(), exceptions)


if __name__ == '__main__':
    unittest.main()