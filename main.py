import torch
import higher
import torchmeta

from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

from time import time
from datetime import datetime

from models.metaconv import MetaConv
from learners.maml import MAML


def mock_train(train_inputs, train_labels):
    model = MetaConv(in_size=train_inputs.size(2),
                     in_channels=train_inputs.size(1),
                     out_channels=train_labels.size(0))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    running_loss = 0.0
    running_accuracy = 0.0
    start = time()

    print_step = 50

    for i in range(100):
        optimizer.zero_grad()

        outputs = model(train_inputs)

        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, dim=1)
        accuracy = (predicted == train_labels).sum().item() / train_labels.size(0)

        running_loss += loss
        running_accuracy += accuracy

        if i % print_step == 0:
            end = time()
            print(f'Iteration {i} loss: {running_loss / print_step :.10f} '
                  f'accuracy: {100 * running_accuracy / print_step}% '
                  f'speed: {print_step / (end - start) :.2f} iter/s')
            running_loss = 0.0
            running_accuracy = 0.0
            start = time()


def mock_train_maml():
    train_dataset = miniimagenet('datasets', ways=5, shots=1, test_shots=15, meta_train=True, download=True)
    train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=5, num_workers=4)

    val_dataset = miniimagenet('datasets', ways=5, shots=1, test_shots=15, meta_val=True, download=True)
    val_dataloader = BatchMetaDataLoader(val_dataset, batch_size=5, num_workers=4)

    test_dataset = miniimagenet('datasets', ways=5, shots=1, test_shots=15, meta_test=True, download=True)
    test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=5, num_workers=4)

    model = MetaConv()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters())

    now = datetime.now()
    maml = MAML(train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                model=model,
                optimizer=optimizer,
                meta_batch_size=5,
                ways=5,
                shots=1,
                test_shots=15,
                inner_steps_train=5,
                inner_steps_test=10,
                outer_steps=60000,
                experiment_name=f'maml__{now.strftime("%d_%B_%Y__%H_%M_%S")}')

    maml.train(training=True)
    # Test using best checkpoint saved
    maml.test(resume=True)


def main():
    print(torch.cuda.is_available())
    print(torchmeta.__version__)

    dataset = miniimagenet('datasets', ways=5, shots=1, test_shots=15, meta_train=True, download=True)
    dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)

    for batch in dataloader:
        train_inputs, train_labels = batch["train"]
        print(f'Train inputs shape: {train_inputs.shape}')
        print(f'Train labels shape: {train_labels.shape}')

        test_inputs, test_labels = batch["test"]
        print(f'Test inputs shape: {test_inputs.shape}')
        print(f'Test labels shape: {test_labels.shape}')

        mock_train(train_inputs[0], train_labels[0])

        break

    mock_train_maml()


if __name__ == '__main__':
    main()
