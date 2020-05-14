import torch
import higher
import argparse

import torch.nn.functional as F

from models.metaconv import MetaConv
from models.metaconv_contextual import MetaConvContextual
from models.metaconv_support import MetaConvSupport
from models.resnet12 import resnet12
from models.resnet12 import resnet12_narrow128



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
        if args.model == "resnet12":
            self.model = resnet12(n_ways=self.args.n_ways, k_spt=self.args.k_spt)
            self.model.cuda()
        if args.model == "resnet12_narrow128":
            self.model = resnet12_narrow128(n_ways=self.args.n_ways, k_spt=self.args.k_spt)
            self.model.cuda()
        assert self.model is not None

        self.best_model = None

    @staticmethod
    def add_arguments(argparser: argparse.ArgumentParser):
        argparser.add_argument('--prediction_type', type=str, help='fc/proto', default='proto')
        argparser.add_argument('--inner_steps_train', type=int, help='number of iters in inner loop ar train time', default=5)
        argparser.add_argument('--inner_steps_test', type=int, help='number of iters in inner loop ar test time', default=10)
        argparser.add_argument('--inner_lr', type=float, help='number of iters in inner loop ar test time', default=1e-2)
        argparser.add_argument('--model', type=str, help='model optimized in inner loop', default='meta_conv_support')
        argparser.add_argument('--num_filters', type=int, help='number of conv filters', default=32)

    def get_inner_trainable_params(self):
        return self.model.fc.parameters()
        # return self.model.fc_update.parameters()

    def get_outer_trainable_params(self):
        return self.model.parameters()

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    def fc_prediction(self, x, model, is_support):
        e_embeddings = model.forward(x, is_support=is_support)
        prediction = model.predict(e_embeddings)

        return prediction

    def proto_prediction(self, x, model, train_inputs, train_labels):
        support_embeddings = model.forward(train_inputs, is_support=True)
        support_embeddings = support_embeddings.view(self.args.n_ways, self.args.k_spt, -1)
        train_labels = train_labels.view(self.args.n_ways, self.args.k_spt, -1).max(axis=1)[0][:, 0]

        # Average over all examples in the same class
        support_prototypes = support_embeddings.mean(axis=1)
        support_prototypes = F.normalize(support_prototypes, p=2, dim=1)

        x_embeddings = model.forward(x, is_support=False)
        x_embeddings = F.normalize(x_embeddings, p=2, dim=1)

        prediction = torch.matmul(x_embeddings, support_prototypes.transpose(dim0=0, dim1=1))
        prediction = model.temp * prediction
        train_labels_oh = torch.nn.functional.one_hot(train_labels, self.args.n_ways)
        prediction = torch.matmul(prediction, train_labels_oh.float())

        return prediction

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

            # Create inner loop context using higher library
            with higher.innerloop_ctx(self.model, opt=inner_optimizer,
                                      copy_initial_weights=False, track_higher_grads=training) as (fmodel, diffopt):
                # Inner loop
                for _ in range(inner_steps):
                    # Determine prediction type
                    if self.args.prediction_type == 'fc':
                        train_logits = self.fc_prediction(train_inputs, fmodel, is_support=True)
                    if self.args.prediction_type == 'proto':
                        train_logits = self.proto_prediction(train_inputs, fmodel, train_inputs, train_labels)

                    train_loss = torch.nn.functional.cross_entropy(train_logits, train_labels)

                    diffopt.step(train_loss)

            # Query the trained model
            if self.args.prediction_type == 'fc':
                test_logits = self.fc_prediction(test_inputs, fmodel, is_support=False)
            if self.args.prediction_type == 'proto':
                test_logits = self.proto_prediction(test_inputs, fmodel, train_inputs, train_labels)
            test_loss = F.cross_entropy(test_logits, test_labels)

            _, test_predictions = torch.max(test_logits, dim=1)

            test_accuracy = (test_predictions == test_labels).sum().item() / test_labels.size(0)

            # Compute metrics
            meta_batch_loss += test_loss.detach().item()
            meta_batch_accuracy += test_accuracy

            # Propagate task loss through inner loop rollup
            if training:
                test_loss.backward()

        return meta_batch_loss / self.args.tasks_num, meta_batch_accuracy / self.args.tasks_num

    def get_state_dict(self):
        return {'model_state_dict': self.model.state_dict()}

    def load_checkpoint(self, state_dict):
        self.model.load_state_dict(state_dict['model_state_dict'])