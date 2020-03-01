import torch
import higher
import os

from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

from torch.optim.optimizer import Optimizer
from time import time


class MAML:
    def __init__(self, train_dataloader, val_dataloader, test_dataloader, model: torch.nn.Module, optimizer: Optimizer,
                 meta_batch_size, ways, shots, test_shots, inner_steps, outer_steps,
                 experiment_name):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.optimizer = optimizer
        self.meta_batch_size = meta_batch_size
        self.n_way = ways
        self.k_spt = shots
        self.k_qry = test_shots
        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        self.experiment_name = experiment_name

        self.best_model = None

    def train(self, training=False, validation=False, testing=False, resume=False, train_iter=None):
        # Check that exactly one of training/validating/testing parameters is set to True
        assert int(training) + int(validation) + int(testing) == 1
        # Check that resume is True only if training/testing in True
        # assert int(training) ^ int(resume) == 0 or resume is False

        # Select dataset split
        if training:
            self.model.train()
            dataloader = self.train_dataloader
        if validation:
            self.model.eval()
            dataloader = self.val_dataloader
        if testing:
            if resume:
                self.load_checkpoint(checkpoint_name='best_checkpoint')
            self.model.eval()
            dataloader = self.test_dataloader

        running_loss = 0.0
        running_accuracy = 0.0
        train_print_step = 10
        val_print_step = 50
        test_num_iters = 50
        if resume:
            test_num_iters = 1000
        checkpoint_step = 500
        start = time()

        for it, meta_batch in enumerate(dataloader):
            # Stop conditions
            if training and it > self.outer_steps:
                break
            if validation and it > val_print_step:
                break
            if testing and it > test_num_iters:
                break

            # Load immediately after saving model only for developing purposes
            if training and it > 0 and (it - 1) % checkpoint_step == 0:
                self.load_checkpoint(checkpoint_name=f'checkpoint_{it - 1}')

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
            if (training and it % train_print_step == 0) or (validation and it == val_print_step)\
                    or (testing and it == test_num_iters):
                end = time()

                if training:
                    print_step = train_print_step
                    phase = '[TRAIN]'
                if validation:
                    print_step = val_print_step + 1
                    phase = '[VAL]'
                if testing:
                    print_step = test_num_iters + 1
                    phase = '[TEST]'

                print(f'{phase} Iteration {it} loss: {running_loss / (self.meta_batch_size * print_step) :.5f} '
                      f'accuracy: {100 * running_accuracy / (self.meta_batch_size * print_step) :.2f}% '
                      f'speed: {print_step / (end - start) :.2f} iter/s')

                # [TODO] Save best model based on val accuracy
                if validation and (self.best_model is None or
                                   100 * running_accuracy / (self.meta_batch_size * print_step) > self.best_model):
                    self.best_model = 100 * running_accuracy / (self.meta_batch_size * print_step)
                    self.save_checkpoint(it=train_iter, checkpoint_name='best_checkpoint')

                running_loss = 0.0
                running_accuracy = 0.0

                start = time()

            # Run validation to select the best model
            if training and it > int(0.0000005 * self.outer_steps) and it % val_print_step == 0:
                self.model.eval()
                self.eval(train_iter=it)
                self.test(resume=False)
                # Model must be set back to training mode after an evaluation
                self.model.train()
                start = time()

            # Save checkpoint of model and optimizer
            if training and it % checkpoint_step == 0:
                self.save_checkpoint(it=it, checkpoint_name=f'checkpoint_{it}')

    def eval(self, train_iter=None):
        self.train(validation=True, train_iter=train_iter)

    def test(self, resume):
        self.train(testing=True, resume=resume)

    def save_checkpoint(self, it, checkpoint_name):
        if not os.path.isdir(f'experiments/{self.experiment_name}'):
            os.makedirs(f'experiments/{self.experiment_name}')
        print(f'Saving {checkpoint_name} ...')
        torch.save({
            'iteration': it,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, f'experiments/{self.experiment_name}/{checkpoint_name}')

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(f'experiments/{self.experiment_name}/{checkpoint_name}')
        print(f'Loading {checkpoint_name} from iteration {checkpoint["iteration"]} ...')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])