import torch
import argparse
import copy

from time import time

from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader


class GenericTest:
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

    def _equal_parameters(self, params1, params2, exceptions=None):
        for param1, param2 in zip(params1, params2):
            if exceptions is not None and param2.data_ptr() in exceptions:
                self.assertTrue(param1.data.ne(param2.data).sum() > 0)
                continue

            self.assertTrue(param1.data.ne(param2.data).sum() == 0)

    def test_batch_overfit(self, model, learnable_params, verbose=False, print_step=10):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(learnable_params, lr=1e-3)

        model.train()

        running_loss = 0.0
        running_accuracy = 0.0
        start = time()

        accuracies = []
        losses = []

        for i in range(500):
            optimizer.zero_grad()

            outputs = model(self.inputs)

            loss = criterion(outputs, self.labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, dim=1)
            accuracy = (predicted == self.labels).sum().item() / self.labels.size(0)

            running_loss += loss
            running_accuracy += accuracy

            if i % print_step == 0:
                end = time()

                losses.append(running_loss / print_step)
                accuracies.append(100 * running_accuracy / print_step)

                if verbose:
                    print(f'Iteration {i} loss: {losses[-1] :.10f} '
                          f'accuracy: {accuracies[-1]}% '
                          f'speed: {print_step / (end - start) :.2f} iter/s')

                running_loss = 0.0
                running_accuracy = 0.0
                start = time()

        window_size = 3
        if verbose:
            print("Accuracies:")
            print(accuracies)
        self.assertEqual(sum(accuracies[-window_size:]) / window_size, 100)

    def test_train_iteration(self, learner):
        original_learner = copy.deepcopy(learner)
        optimizer = torch.optim.Adam(params=learner.get_outer_trainable_params(), lr=1e-3)

        # Run one meta-training iteration
        optimizer.zero_grad()
        _, _ = learner.run_iteration(self.meta_batch, training=True)
        optimizer.step()

        # Check that outer trainable params changed
        exceptions = list(map(lambda p: p.data_ptr(), original_learner.get_outer_trainable_params()))
        self._equal_parameters(learner.model.parameters(), original_learner.model.parameters(), exceptions)