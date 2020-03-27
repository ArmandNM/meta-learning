import argparse

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