import torch
import higher

from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

from torch.optim.optimizer import Optimizer
from time import time

from models.MetaConv import MetaConv


class MAML:
    def __init__(self, dataloader, model: torch.nn.Module, optimizer: Optimizer,
                 meta_batch_size, ways, shots, test_shots, inner_steps):
        self.dataloader = dataloader
        self.model = model
        self.optimizer= optimizer
        self.meta_batch_size = meta_batch_size
        self.n_way = ways
        self.k_spt = shots
        self.k_qry = test_shots
        self.inner_steps = inner_steps

    def train(self):
        self.model.train()

        running_loss = 0.0
        running_accuracy = 0.0
        print_step = 5
        start = time()

        for it, meta_batch in enumerate(self.dataloader):
            meta_train_inputs, meta_train_labels = meta_batch["train"]
            meta_test_inputs, meta_test_labels = meta_batch["test"]

            inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)

            self.optimizer.zero_grad()
            for task_idx in range(self.meta_batch_size):

                with higher.innerloop_ctx(self.model, opt=inner_optimizer,
                                          copy_initial_weights=False) as (fmodel, diffopt):
                    train_inputs = meta_train_inputs[task_idx].cuda()
                    train_labels = meta_train_labels[task_idx].cuda()

                    # Inner loop
                    for _ in range(self.inner_steps):
                        train_logits = fmodel(train_inputs)
                        train_loss = torch.nn.functional.cross_entropy(train_logits, train_labels)
                        diffopt.step(train_loss)

                    # Query the trained model
                    test_inputs = meta_test_inputs[task_idx].cuda()
                    test_labels = meta_test_labels[task_idx].cuda()

                    test_logits = fmodel(test_inputs)
                    test_loss = torch.nn.functional.cross_entropy(test_logits, test_labels)
                    _, test_predictions = torch.max(test_logits, dim=1)

                    test_accuracy = (test_predictions == test_labels).sum().item() / test_labels.size(0)

                    # Compute metrics
                    running_loss += test_loss
                    running_accuracy += test_accuracy

                    # Propagate task loss through inner loop rollup
                    test_loss.backward()

            self.optimizer.step()
            if it % print_step == 0:
                end = time()
                print(f'Iteration {it} loss: {running_loss / (self.meta_batch_size * print_step) :.5f} '
                      f'accuracy: {100 * running_accuracy / (self.meta_batch_size * print_step) :.2f}% '
                      f'speed: {print_step / (end - start) :.2f} iter/s')
                running_loss = 0.0
                running_accuracy = 0.0
                start = time()

    def eval(self):
        pass
