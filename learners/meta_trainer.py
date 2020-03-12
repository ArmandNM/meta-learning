import torch
import argparse
import os

from learners.maml import MAML

from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torch.utils.tensorboard import SummaryWriter

from time import time


class MetaTrainer:
    def __init__(self):
        self.argparser = argparse.ArgumentParser()

        self.argparser.add_argument('--n_ways', type=int, help='num of classes', default=5)
        self.argparser.add_argument('--k_spt', type=int, help='samples per class in support set', default=1)
        self.argparser.add_argument('--k_qry', type=int, help='samples per class in query set', default=15)
        self.argparser.add_argument('--tasks_num', type=int, help='meta batch size', default=5)
        self.argparser.add_argument('--meta_learner', type=str, help='name of meta learning method', default='maml')
        self.argparser.add_argument('--meta_optimizer', type=str, help='meta optimizer', default='adam')
        self.argparser.add_argument('--meta_iters', type=int, help='num of meta iterations', default=60000)
        self.argparser.add_argument('--meta_lr', type=float, help='meta learning rate', default=1e-3)
        self.argparser.add_argument('--dataset', type=str, help='meta dataset to use', default='miniimagenet')
        self.argparser.add_argument('--experiment_name', type=str, help='name of current experiment', default='maml')

        self.argparser.add_argument('--train_print_step', type=int,
                                    help='number of meta iterations before printing metrics', default=10)
        self.argparser.add_argument('--val_print_step', type=int,
                                    help='number of meta iterations before running validation', default=50)
        self.argparser.add_argument('--val_num_iters', type=int,
                                    help='number of meta iterations to run for validation', default=50)
        self.argparser.add_argument('--test_num_iters', type=int,
                                    help='number of meta iterations to run for testing', default=1000)
        self.argparser.add_argument('--checkpoint_step', type=str,
                                    help='number of meta iterations before saving checkpoint', default=500)

        # Parse general meta-learning args
        self.args, additional_args = self.argparser.parse_known_args()

        # Selected meta-learner constructor
        learner_constructor = None
        print(self.args)
        if self.args.meta_learner == 'maml':
            learner_constructor = MAML
        assert learner_constructor is not None

        # Parse specific parameters of the selected meta-learner
        learner_argparser = argparse.ArgumentParser()
        learner_constructor.add_arguments(argparser=learner_argparser)
        self.args, _ = learner_argparser.parse_known_args(additional_args, namespace=self.args)
        print(self.args)

        # Create dataset loaders
        dataset = None
        if self.args.dataset == 'miniimagenet':
            dataset = miniimagenet
        assert dataset is not None

        train_dataset = dataset('datasets', ways=self.args.n_ways, shots=self.args.k_spt, test_shots=self.args.k_qry,
                                meta_train=True, download=True)
        self.train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=self.args.tasks_num, num_workers=4)

        val_dataset = dataset('datasets', ways=self.args.n_ways, shots=self.args.k_spt, test_shots=self.args.k_qry,
                              meta_val=True, download=True)
        self.val_dataloader = BatchMetaDataLoader(val_dataset, batch_size=self.args.tasks_num, num_workers=4)

        test_dataset = dataset('datasets', ways=self.args.n_ways, shots=self.args.k_spt, test_shots=self.args.k_qry,
                               meta_test=True, download=True)
        self.test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=self.args.tasks_num, num_workers=4)

        # Create meta-learner object
        self.learner = learner_constructor(args=self.args)

        # Create optimizer
        self.optimizer = None
        if self.args.meta_optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params=self.learner.get_trainable_params(), lr=self.args.meta_lr)
        assert self.optimizer is not None

        # Used to save best checkpoint, initially None
        self.best_score = None

        # Initialize summary writer to save logs during training that can be visualized in Tensorboard
        self.writer = SummaryWriter(f'experiments/{self.args.experiment_name}/{self.args.experiment_name}')
        # Add layout for combined metrics on custom scalars page
        layout = {
            'Accuracies': {
                'all_accuracies':
                    ['Multiline', ['Accuracies/train_accuracy', 'Accuracies/val_accuracy', 'Accuracies/test_accuracy']]},
            'Losses': {
                'all_losses':
                    ['Multiline', ['Losses/train_loss', 'Losses/val_loss', 'Losses/test_loss']]}
        }
        self.writer.add_custom_scalars(layout)

    def train(self, training=False, validation=False, testing=False, resume=False, train_iter=None):
        # Check that exactly one of training/validation/testing parameters is set to True
        assert int(training) + int(validation) + int(testing) == 1

        # Select dataset split and stopping conditions based on training/validation/testing
        dataloader = None
        num_meta_iterations = None
        if training:
            num_meta_iterations = self.args.meta_iters
            dataloader = self.train_dataloader
        elif validation:
            num_meta_iterations = self.args.val_num_iters
            dataloader = self.val_dataloader
        elif testing:
            num_meta_iterations = self.args.val_num_iters
            if resume:
                self.load_checkpoint(checkpoint_name='best_checkpoint')
                num_meta_iterations = self.args.test_num_iters
                train_iter = self.args.meta_iters + 1
            dataloader = self.test_dataloader
        assert dataloader is not None
        assert num_meta_iterations is not None

        running_loss = 0.0
        running_accuracy = 0.0
        start = time()

        for it, meta_batch in enumerate(dataloader):
            # Stop conditions
            if it > num_meta_iterations:
                break

            # Load immediately after saving model only for developing purposes
            if training and it > 0 and (it - 1) % self.args.checkpoint_step == 0:
                self.load_checkpoint(checkpoint_name=f'checkpoint_{it - 1}')

            # Only care about meta-optimizer at meta-training time
            if training:
                self.optimizer.zero_grad()

            # Run one iteration and get resulting metrics
            loss, accuracy = self.learner.run_iteration(meta_batch=meta_batch, training=training)
            # print(f'res loss {loss}')
            # print(f'res acc {accuracy}')

            # Update running metrics
            running_loss += loss
            running_accuracy += accuracy

            # Update learnable parameters
            if training:
                self.optimizer.step()

            # Print intermediate results for training or val/test results
            if (training and it % self.args.train_print_step == 0) or (not training and it == num_meta_iterations):
                end = time()

                print_step = None
                phase = None
                train_it = None
                if training:
                    print_step = self.args.train_print_step
                    phase = 'train'
                    train_it = it
                if validation:
                    print_step = num_meta_iterations + 1
                    phase = 'val'
                    train_it = train_iter
                if testing:
                    print_step = num_meta_iterations + 1
                    phase = 'test'
                    train_it = train_iter
                assert print_step is not None and phase is not None and train_it is not None

                log_loss = running_loss / (self.args.tasks_num * print_step)
                log_accuracy = 100 * running_accuracy / (self.args.tasks_num * print_step)

                # Print console logs
                print(f'[{str.upper(phase)}] Iteration {train_it} loss: {log_loss :.5f} '
                      f'accuracy: {log_accuracy :.2f}% '
                      f'speed: {print_step / (end - start) :.2f} iter/s')

                # Save Tensorboard logs
                self.writer.add_scalar(f'Losses/{phase}_loss', log_loss, train_it)
                self.writer.add_scalar(f'Accuracies/{phase}_accuracy', log_accuracy, train_it)

                # Save combined logs
                # self.writer.add_scalars('Losses/all_losses', {
                #     f'{phase}_loss': log_loss
                # }, train_it)
                # self.writer.add_scalars('Accuracies/all_accuracies', {
                #     f'{phase}_accuracy': log_accuracy
                # }, train_it)

                # Save best model based on val accuracy
                if validation and (self.best_score is None or
                                   100 * running_accuracy / (self.args.tasks_num * print_step) > self.best_score):
                    self.best_score = 100 * running_accuracy / (self.args.tasks_num * print_step)
                    self.save_checkpoint(it=train_iter, checkpoint_name='best_checkpoint')

                running_loss = 0.0
                running_accuracy = 0.0

                start = time()

            # Run validation to select the best model
            if training and it > int(0.0000005 * self.args.meta_iters) and it % self.args.val_print_step == 0:
                self.learner.set_eval_mode()
                self.eval(train_iter=it)
                self.test(train_iter=it, resume=False)
                # Model must be set back to training mode after an evaluation
                # [TODO] implement and set_train_mode() for the learner instead of accessing member
                self.learner.set_train_mode()
                start = time()

            # Save checkpoint of model and optimizer
            if training and it % self.args.checkpoint_step == 0:
                self.save_checkpoint(it=it, checkpoint_name=f'checkpoint_{it}')

    def eval(self, train_iter=None):
        self.learner.set_eval_mode()
        self.train(validation=True, train_iter=train_iter)

    def test(self, train_iter=None, resume=False):
        self.learner.set_train_mode()
        self.train(testing=True, train_iter=train_iter, resume=resume)

    def save_checkpoint(self, it, checkpoint_name):
        if not os.path.isdir(f'experiments/{self.args.experiment_name}'):
            os.makedirs(f'experiments/{self.args.experiment_name}')
        print(f'Saving {checkpoint_name} ...')
        torch.save({
            'iteration': it,
            'learner_state_dict': self.learner.get_state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, f'experiments/{self.args.experiment_name}/{checkpoint_name}')

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(f'experiments/{self.args.experiment_name}/{checkpoint_name}')
        print(f'Loading {checkpoint_name} from iteration {checkpoint["iteration"]} ...')
        self.learner.load_checkpoint(state_dict=checkpoint['learner_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == '__main__':
    # Setting benchmark = True should improve performance for constant shape input
    torch.backends.cudnn.benchmark = True
    meta_trainer = MetaTrainer()
    meta_trainer.train(training=True)
    meta_trainer.test(resume=True)
