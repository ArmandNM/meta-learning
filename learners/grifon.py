import torch
import higher
import argparse

from models.metaconv import MetaConv
from models.metaconv_contextual import MetaConvContextual
from models.metaconv_support import MetaConvSupport


class GRIFON:
    def __init__(self, args):
        self.args = args

        self.model = None
        if args.model == 'meta_conv':
            self.model = MetaConv(out_channels=self.args.n_ways)
            self.model.cuda()
        if args.model == 'meta_conv_contextual':
            self.model = MetaConvContextual(out_channels=self.args.n_ways)
            self.model.cuda()
        if args.model == 'meta_conv_support':
            self.model = MetaConvSupport(out_channels=self.args.n_ways, k_spt=self.args.k_spt)
            self.model.cuda()
        assert self.model is not None

        self.best_model = None

    @staticmethod
    def add_arguments(argparser: argparse.ArgumentParser):
        argparser.add_argument('--inner_steps_train', type=int, help='number of iters in inner loop ar train time', default=5)
        argparser.add_argument('--inner_steps_test', type=int, help='number of iters in inner loop ar test time', default=10)
        argparser.add_argument('--model', type=str, help='model optimized in inner loop', default='meta_conv_support')

    def get_inner_trainable_params(self):
        return self.model.fc.parameters()

    def get_outer_trainable_params(self):
        return self.model.parameters()

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    def run_iteration(self, meta_batch, training=False):
        if training:
            self.model.train()
            inner_steps = self.args.inner_steps_train
        else:
            self.model.eval()
            inner_steps = self.args.inner_steps_test

        meta_batch_loss = 0.0
        meta_batch_accuracy = 0.0

        meta_train_inputs, meta_train_labels = meta_batch["train"]
        meta_test_inputs, meta_test_labels = meta_batch["test"]

        inner_optimizer = torch.optim.SGD(self.get_inner_trainable_params(), lr=1e-2)

        for task_idx in range(self.args.tasks_num):
            # Extract examples and labels for current task
            train_inputs = meta_train_inputs[task_idx].cuda()
            train_labels = meta_train_labels[task_idx].cuda()

            test_inputs = meta_test_inputs[task_idx].cuda()
            test_labels = meta_test_labels[task_idx].cuda()

            # self.model.set_support_params(train_inputs)

            # Create inner loop context using higher library
            with higher.innerloop_ctx(self.model, opt=inner_optimizer,
                                      copy_initial_weights=False, track_higher_grads=training) as (fmodel, diffopt):

                # Inner loop
                for _ in range(inner_steps):
                    train_logits = fmodel.forward(train_inputs, is_support=True)

                    train_loss = torch.nn.functional.cross_entropy(train_logits, train_labels)

                    # _, train_predictions = torch.max(train_logits, dim=1)
                    # train_accuracy = (train_predictions == train_labels).sum().item() / train_labels.size(0)
                    # print(train_accuracy)

                    diffopt.step(train_loss)

                # One extra iteration to compute the support set intermediary features using current weights
                fmodel.forward(train_inputs, is_support=True)

                # Query the trained model
                test_logits = fmodel(test_inputs)
                test_loss = torch.nn.functional.cross_entropy(test_logits, test_labels)
                _, test_predictions = torch.max(test_logits, dim=1)

                test_accuracy = (test_predictions == test_labels).sum().item() / test_labels.size(0)

                # Compute metrics
                meta_batch_loss += test_loss.detach()
                meta_batch_accuracy += test_accuracy

                # Propagate task loss through inner loop rollup
                if training:
                    test_loss.backward()

        return meta_batch_loss, meta_batch_accuracy

    def get_state_dict(self):
        return {'model_state_dict': self.model.state_dict()}

    def load_checkpoint(self, state_dict):
        self.model.load_state_dict(state_dict['model_state_dict'])