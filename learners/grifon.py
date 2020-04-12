import torch
import higher
import argparse

import torch.nn.functional as F

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
            self.model = MetaConvSupport(out_channels=self.args.n_ways, k_spt=self.args.k_spt,
                                         hidden_size=self.args.num_filters)
            self.model.cuda()
        assert self.model is not None

        self.best_model = None

    @staticmethod
    def add_arguments(argparser: argparse.ArgumentParser):
        argparser.add_argument('--inner_steps_train', type=int, help='number of iters in inner loop ar train time', default=5)
        argparser.add_argument('--inner_steps_test', type=int, help='number of iters in inner loop ar test time', default=10)
        argparser.add_argument('--inner_lr', type=float, help='number of iters in inner loop ar test time', default=1e-2)
        argparser.add_argument('--model', type=str, help='model optimized in inner loop', default='meta_conv_support')
        argparser.add_argument('--num_filters', type=int, help='number of conv filters', default=32)

    def get_inner_trainable_params(self):
        return self.model.fc_update.parameters()

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

        inner_optimizer = torch.optim.SGD(self.get_inner_trainable_params(), lr=self.args.inner_lr)

        for task_idx in range(self.args.tasks_num):
            # Extract examples and labels for current task
            train_inputs = meta_train_inputs[task_idx].cuda()
            train_labels = meta_train_labels[task_idx].cuda()

            test_inputs = meta_test_inputs[task_idx].cuda()
            test_labels = meta_test_labels[task_idx].cuda()

            support_embeddings = self.model.forward(train_inputs, is_support=True)
            support_embeddings = support_embeddings.view(self.args.n_ways, self.args.k_spt, -1)
            train_labels = train_labels.view(self.args.n_ways, self.args.k_spt, -1).max(axis=1)[0][:, 0]
            # Average over all examples in the same class
            support_prototypes = support_embeddings.mean(axis=1)
            support_prototypes = F.normalize(support_prototypes, p=2, dim=1)

            query_embeddings = self.model.forward(test_inputs, is_support=False)
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

            prediction = torch.matmul(query_embeddings, support_prototypes.transpose(dim0=0, dim1=1))
            prediction = self.model.temp * prediction
            train_labels_oh = torch.nn.functional.one_hot(train_labels, self.args.n_ways)
            prediction = torch.matmul(prediction, train_labels_oh.t().float())
            # prediction = torch.gather(prediction, dim=1, index=train_labels)

            test_loss = F.cross_entropy(prediction, test_labels)

            _, test_predictions = torch.max(prediction, dim=1)

            test_accuracy = (test_predictions == test_labels).sum().item() / test_labels.size(0)

            # Compute metrics
            meta_batch_loss += test_loss.detach()
            meta_batch_accuracy += test_accuracy

        return meta_batch_loss, meta_batch_accuracy

    def get_state_dict(self):
        return {'model_state_dict': self.model.state_dict()}

    def load_checkpoint(self, state_dict):
        self.model.load_state_dict(state_dict['model_state_dict'])