import argparse
from sys import stderr
from typing import Callable
from matplotlib import pyplot as plt
import torch
from torch import Tensor
from step import DetectNN
from step import NumberNN
from step import TypeNN
from step.detect import DetectDataset
from step.number import NumberDataset
from step.type import TypeDataset

device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)


# src: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation
def train_loop(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: torch.optim.Optimizer,
):
    if dataloader.batch_size is None:
        dataloader.batch_size = 1
    size = len(dataloader.dataset)  # type: ignore

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
) -> tuple[float, float]:
    model.eval()
    size = len(dataloader.dataset)  # type: ignore
    num_batches = len(dataloader)
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

    return correct, test_loss


class CustomFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass


parser = argparse.ArgumentParser(
    prog="train.py", description="Диоген учащийся.", formatter_class=CustomFormatter
)
parser.add_argument(
    "-s",
    "--step",
    choices=["detect", "type", "number"],
    required=True,
    help="Модель, которую необходимо обучить/дообучить.",
)
parser.add_argument(
    "-r",
    "--retrain",
    action="store_true",
    help="Необходимо ли переообучить модель вместо дообучения.",
)
parser.add_argument(
    "-l", "--learning-rate", type=float, default=0.001, help="Темп обучения."
)
parser.add_argument("-b", "--batch-size", type=int, default=64, help="Размер батча.")
parser.add_argument(
    "-e", "--epochs", type=int, default=20, help="Количество эпох обучения."
)
parser.add_argument(
    "-p",
    "--part",
    type=float,
    default=0.8,
    help="Доля данных, которые станут тренировочными. Значение должно быть в пределах (0; 1].",
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.part <= 0 or args.part > 1:
        print(
            f"Значение PART должно быть в пределах (0; 1], но равно {args.part}",
            file=stderr,
        )
        exit(1)

    match args.step:
        case "detect":
            print("step is detect")
            model = DetectNN()
            dataset = DetectDataset()
        case "type":
            print("step is type")
            model = TypeNN()
            dataset = TypeDataset()
        case "number":
            print("step is number")
            model = NumberNN()
            dataset = NumberDataset()
        case _:
            raise Exception("Выберите модель из списка: detect, type, number.")

    if not args.retrain:
        state_dict = torch.load(f"models/{args.step}.pth", weights_only=True)
        model.load_state_dict(state_dict)

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [args.part, 1 - args.part]
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, args.batch_size, True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    epochs = range(1, args.epochs + 1)
    accuracies = []
    losses = []
    for t in epochs:
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        accuracy, loss = test_loop(test_dataloader, model, loss_fn)
        accuracies.append(accuracy)
        losses.append(loss)

    print("Done!")

    torch.save(model.state_dict(), f"models/{args.step}.pth")

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("epochs")

    color = "tab:red"
    ax1.set_ylabel("accuracy", color=color)
    ax1.plot(epochs, accuracies, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("loss", color=color)
    ax2.plot(epochs, losses, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.show()
