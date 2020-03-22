from learners.maml import MAML


class CAVIA(MAML):
    def __init__(self, args):
        super().__init__(args)

    def get_inner_trainable_params(self):
        return [self.model.context_params]

    def get_outer_trainable_params(self):
        return self.model.parameters()

    def run_iteration(self, meta_batch, training=False):
        self.model.reset_context_params()
        return super(CAVIA, self).run_iteration(meta_batch, training)