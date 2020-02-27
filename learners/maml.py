import torch
import higher

from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

from torch.optim.optimizer import Optimizer
from time import time


class MAML:
    def __init__(self, train_dataloader, val_dataloader, model: torch.nn.Module, optimizer: Optimizer,
                 meta_batch_size, ways, shots, test_shots, inner_steps, outer_steps):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.optimizer= optimizer
        self.meta_batch_size = meta_batch_size
        self.n_way = ways
        self.k_spt = shots
        self.k_qry = test_shots
        self.inner_steps = inner_steps
        self.outer_steps = outer_steps

    def train(self, training=False, validation=False, testing=False):
        # Check that exactly one of trainig/validating/testing parameters is set to True
        assert int(training) + int(validation) + int(testing) == 1

        # Select dataset split
        if training:
            self.model.train()
            dataloader = self.train_dataloader
        if validation:
            self.model.eval()
            dataloader = self.val_dataloader

        running_loss = 0.0
        running_accuracy = 0.0
        train_print_step = 5
        val_print_step = 30
        start = time()

        for it, meta_batch in enumerate(dataloader):
            # Stop conditions
            if training and it > self.outer_steps:
                break
            if not training and it > val_print_step:
                break

            meta_train_inputs, meta_train_labels = meta_batch["train"]
            meta_test_inputs, meta_test_labels = meta_batch["test"]

            inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)

            # Only care about meta-optimizer at meta-training time
            if training:
                self.optimizer.zero_grad()

            # Run the inner loop for all tasks in meta-batch
            for task_idx in range(self.meta_batch_size):
                # Create inner loop context using higher library
                with higher.innerloop_ctx(self.model, opt=inner_optimizer,
                                          copy_initial_weights=False, track_higher_grads=training) as (fmodel, diffopt):

                    # Extract examples and labels for current task
                    train_inputs = meta_train_inputs[task_idx].cuda()
                    train_labels = meta_train_labels[task_idx].cuda()

                    test_inputs = meta_test_inputs[task_idx].cuda()
                    test_labels = meta_test_labels[task_idx].cuda()

                    # Inner loop
                    for _ in range(self.inner_steps):
                        train_logits = fmodel(train_inputs)
                        train_loss = torch.nn.functional.cross_entropy(train_logits, train_labels)

                        # _, train_predictions = torch.max(train_logits, dim=1)
                        # train_accuracy = (train_predictions == train_labels).sum().item() / train_labels.size(0)
                        # print(train_accuracy)

                        diffopt.step(train_loss)

                    # Query the trained model
                    test_logits = fmodel(test_inputs)
                    test_loss = torch.nn.functional.cross_entropy(test_logits, test_labels)
                    _, test_predictions = torch.max(test_logits, dim=1)

                    test_accuracy = (test_predictions == test_labels).sum().item() / test_labels.size(0)

                    # Compute metrics
                    running_loss += test_loss.detach()
                    running_accuracy += test_accuracy

                    # Propagate task loss through inner loop rollup
                    if training:
                        test_loss.backward()

            # Use the meta-optimizer to update parameters
            if training:
                self.optimizer.step()

            # Print intermediate results for training or val/test results
            if (training and it % train_print_step == 0) or (not training and it == val_print_step):
                end = time()

                if training:
                    print_step = train_print_step
                    phase = '[TRAIN]'
                if validation:
                    print_step = val_print_step
                    phase = '[VAL]'

                print(f'{phase} Iteration {it} loss: {running_loss / (self.meta_batch_size * print_step) :.5f} '
                      f'accuracy: {100 * running_accuracy / (self.meta_batch_size * print_step) :.2f}% '
                      f'speed: {print_step / (end - start) :.2f} iter/s')

                # [TODO] Save best model based on val accuracy

                running_loss = 0.0
                running_accuracy = 0.0

                start = time()

            # Run validation to select the best model
            if training and it > int(0.00005 * self.outer_steps) and it % val_print_step == 0:
                self.model.eval()
                self.eval()
                # Model must be set back to training mode after an evaluation
                self.model.train()
                start = time()

    def eval(self):
        self.train(validation=True)
