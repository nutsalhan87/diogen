import json
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from difflib import SequenceMatcher
from pathlib import Path
from diogen.model.plate import CRNN, resize_plate, ALPHABET, BLANK_IDX

# --------------------- CONFIG --------------------- #
DATASET_ROOT = Path("data/plates_cropped")
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
TEST_DIR = DATASET_ROOT / "test"
OUTPUT_DIR = Path("train_crnn")
PLOT_DIR = OUTPUT_DIR / "plot"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LAST_MODEL_PATH = OUTPUT_DIR / "last.pt"
BEST_MODEL_PATH = OUTPUT_DIR / "best.pt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------- ALPHABET MAPPING --------------------- #
LAT2CYR = {
    'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М',
    'H': 'Н', 'O': 'О', 'P': 'Р', 'C': 'С', 'T': 'Т',
    'Y': 'У', 'X': 'Х',
}

def convert_to_cyrillic(text: str) -> str:
    result = []
    for ch in text.upper():
        if ch in ALPHABET:
            result.append(ch)
        elif ch in LAT2CYR:
            result.append(LAT2CYR[ch])
        else:
            raise ValueError(f"Недопустимый символ '{ch}' в номере: {text}")
    return ''.join(result)

# --------------------- DATASET --------------------- #
class PlatesDataset(Dataset):
    def __init__(self, root: str):
        self.img_dir = Path(root) / "img"
        self.ann_dir = Path(root) / "ann"
        self.samples = list(self.ann_dir.glob("*.json"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ann_path = self.samples[idx]
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        plate_str = convert_to_cyrillic(data["description"])
        name = data["name"]
        img_path = self.img_dir / f"{name}.png"
        img = read_image(str(img_path)) / 255.  # 3xHxW
        img = resize_plate(img)  # 1x32xW
        return img, plate_str

# --------------------- ENCODING --------------------- #
def encode_label(label: str) -> torch.Tensor:
    idxs = [ALPHABET.index(c) for c in label]
    return torch.tensor(idxs, dtype=torch.long)

def collate_fn(batch):
    images, labels = zip(*batch)
    targets = [encode_label(label) for label in labels]

    image_batch = torch.stack(images)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets)

    return image_batch, targets_concat, target_lengths, labels

# --------------------- METRICS --------------------- #
def cer(s1: str, s2: str) -> float:
    return 1 - SequenceMatcher(None, s1, s2).ratio()

# --------------------- TRAIN FUNCTION --------------------- #
def train_epoch(model: CRNN, loader: DataLoader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, targets, target_lengths, label_texts in loader:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        logits = model(images)  # TxBxC
        input_lengths = torch.full(size=(logits.size(1),), fill_value=logits.size(0), dtype=torch.long).to(device)

        loss = criterion(logits, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = model.decode_with_confidence(logits)
        for (pred, _), label in zip(predictions, label_texts):
            if pred == label:
                correct += 1
            total += 1

    acc = correct / total
    print(f"Train Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")
    return total_loss, acc

# --------------------- VALIDATION FUNCTION --------------------- #
def validate_epoch(model: CRNN, loader: DataLoader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    total_cer = 0.0

    with torch.no_grad():
        for images, targets, target_lengths, label_texts in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            input_lengths = torch.full(size=(logits.size(1),), fill_value=logits.size(0), dtype=torch.long).to(device)

            loss = criterion(logits, targets, input_lengths, target_lengths)
            total_loss += loss.item()

            predictions = model.decode_with_confidence(logits)
            for (pred, _), ref in zip(predictions, label_texts):
                if pred == ref:
                    correct += 1
                total_cer += cer(ref, pred)
                total += 1

    acc = correct / total
    avg_cer = total_cer / total
    print(f"Val Loss: {total_loss:.4f}, Accuracy: {acc:.4f}, CER: {avg_cer:.4f}")
    return total_loss, acc, avg_cer

# --------------------- TEST FUNCTION --------------------- #
def test(model: CRNN, loader: DataLoader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    total_cer = 0.0

    with torch.no_grad():
        for images, targets, target_lengths, label_texts in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            input_lengths = torch.full(size=(logits.size(1),), fill_value=logits.size(0), dtype=torch.long).to(device)

            loss = criterion(logits, targets, input_lengths, target_lengths)
            total_loss += loss.item()

            predictions = model.decode_with_confidence(logits)
            for (pred, _), ref in zip(predictions, label_texts):
                if pred == ref:
                    correct += 1
                total_cer += cer(ref, pred)
                total += 1

    acc = correct / total
    avg_cer = total_cer / total
    print(f"Test Loss: {total_loss:.4f}, Accuracy: {acc:.4f}, CER: {avg_cer:.4f}")

# --------------------- CLI --------------------- #
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--val', action='store_true', help="Запустить валидацию")
    group.add_argument('--test', action='store_true', help="Запустить тестирование")

    parser.add_argument('--epochs', type=int, default=50, help="Количество эпох")
    parser.add_argument('--batch', type=int, default=8, help="Размер батча")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Устройство")
    parser.add_argument('--lr', type=float, default=1e-4, help="Шаг обучения")
    args = parser.parse_args()

    device = torch.device(args.device)
    num_classes = len(ALPHABET) + 1  # +1 for CTC blank
    model = CRNN(num_classes=num_classes).to(device)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    if BEST_MODEL_PATH.exists():
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print(f"Loaded model from {BEST_MODEL_PATH}")

    train_ds = PlatesDataset(str(TRAIN_DIR))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_ds = PlatesDataset(str(VAL_DIR))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    test_ds = PlatesDataset(str(TEST_DIR))
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    if args.val:
        validate_epoch(model, val_loader, criterion, device)
        return
    elif args.test:
        test(model, test_loader, criterion, device)
        return

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_losses, train_accuracies, val_accuracies, val_cers = [], [], [], [], []
    best_acc = 0.0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, avg_cer = validate_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_cers.append(avg_cer)

        torch.save(model.state_dict(), LAST_MODEL_PATH)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"model_{epoch + 1}.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("Best model updated.")

    # Saving plots
    plt.figure()
    plt.plot(range(4, args.epochs), train_losses[4:], label='Train Loss')
    plt.plot(range(4, args.epochs), val_losses[4:], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_DIR / 'loss.png')

    plt.figure()
    plt.plot(range(4, args.epochs), train_losses, label='Train Loss')
    plt.plot(range(4, args.epochs), val_losses, label='Val Loss')
    plt.yscale('log')
    plt.title('Loss over Epochs (log scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_DIR / 'loss_log.png')

    plt.figure()
    plt.plot(train_accuracies[4:], label='Train Accuracy')
    plt.plot(val_accuracies[4:], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_DIR / 'accuracy.png')

    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.yscale('log')
    plt.title('Accuracy over Epochs (log scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (log)')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_DIR / 'accuracy_log.png')

    plt.figure()
    plt.plot(val_cers, label='Validation CER', color='purple')
    plt.yscale('log')
    plt.title('Validation CER over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('CER')
    plt.grid(True)
    plt.savefig(PLOT_DIR / 'val_cer.png')

if __name__ == "__main__":
    main()

