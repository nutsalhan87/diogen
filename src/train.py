import argparse
import os
from sys import stderr
import torch
import torch.nn.grad
import torch.utils.data.dataset
import diogen
from step import DetectNN
from step import NumberNN
from step import TypeNN
from step.detect import DetectDataset
from step.number import NumberDataset
from step.type import TypeDataset

device = diogen.device



# src: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation
def train_loop(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn, optimizer):
    if dataloader.batch_size is None:
        dataloader.batch_size = 1
    size = len(dataloader.dataset) # type: ignore
    
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader: torch.utils.data.DataLoader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset) # type: ignore
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class CustomFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

parser = argparse.ArgumentParser(
    prog="train.py",
    description="Диоген учащийся.",
    formatter_class=CustomFormatter
)
parser.add_argument("-s", "--step", choices=["detect", "type", "number"], required=True, help="Модель, которую необходимо обучить/дообучить.")
parser.add_argument("-r", "--retrain", action='store_true')
parser.add_argument('-l', "--learning-rate", type=float, default=0.001)
parser.add_argument('-b', "--batch-size", type=int, default=64)
parser.add_argument('-e', "--epochs", type=int, default=20)
parser.add_argument('-p', "--part", type=float, default=0.8, help="Доля данных, которые станут тренировочными. Значение должно быть в пределах (0; 1].")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.part <= 0 or args.part > 1:
        print(f"Значение PART должно быть в пределах (0; 1], но равно {args.part}", file=stderr)
        exit(1)

    match args.step:
        case "detect":
            model = DetectNN()
            dataset = DetectDataset()
        case "type":
            model = TypeNN()
            dataset = TypeDataset()
        case "number":
            model = NumberNN()
            dataset = NumberDataset()
        case _:
            raise Exception("Выберите модель из списка: detect, type, number.")
    
    if not args.retrain:
        state_dict = torch.load(os.path.join("models", f"{args.step}.pth"), weights_only=True)
        model.load_state_dict(state_dict)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [args.part, 1 - args.part])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, args.batch_size, True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")