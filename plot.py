import matplotlib.pyplot as plt
import re

def parse_logs_and_plot(log_data):
    # Инициализация списков для хранения данных
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    val_cer = []
    best_epochs = []  # Для отметок лучших моделей
    
    # Парсинг данных
    current_epoch = 0
    for line in log_data.split('\n'):
        if line.startswith('Epoch'):
            current_epoch = int(re.search(r'Epoch (\d+)', line).group(1))
            epochs.append(current_epoch)
        elif 'Train Loss:' in line:
            t_loss = float(re.search(r'Train Loss: ([\d.-]+)', line).group(1))
            t_acc = float(re.search(r'Accuracy: ([\d.-]+)', line).group(1))
            train_loss.append(t_loss)
            train_acc.append(t_acc)
        elif 'Val Loss:' in line:
            v_loss = float(re.search(r'Val Loss: ([\d.-]+)', line).group(1))
            v_acc = float(re.search(r'Accuracy: ([\d.-]+)', line).group(1))
            v_cer = float(re.search(r'CER: ([\d.-]+)', line).group(1))
            val_loss.append(v_loss)
            val_acc.append(v_acc)
            val_cer.append(v_cer)
        elif 'Best model updated' in line:
            best_epochs.append(current_epoch)
    
    # Создание графиков
    # plt.figure(figsize=(15, 10))
    
    # График Loss
    # plt.subplot(2, 2, 1)
    plt.figure()
    plt.plot(epochs[4:], train_loss[4:], label='Train Loss')
    plt.plot(epochs[4:], val_loss[4:], label='Val Loss')
    for epoch in best_epochs:
        plt.axvline(x=epoch, color='gray', linestyle='--', alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # График Loss (логарифмическая шкала)
    # plt.subplot(2, 2, 4)
    plt.figure()
    plt.plot(epochs[4:], train_loss[4:], label='Train Loss')
    plt.plot(epochs[4:], val_loss[4:], label='Val Loss')
    for epoch in best_epochs:
        plt.axvline(x=epoch, color='gray', linestyle='--', alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss (Logarithmic Scale)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # График Accuracy
    # plt.subplot(2, 2, 2)
    plt.figure()
    plt.plot(epochs[4:], train_acc[4:], label='Train Accuracy')
    plt.plot(epochs[4:], val_acc[4:], label='Val Accuracy')
    for epoch in best_epochs:
        plt.axvline(x=epoch, color='gray', linestyle='--', alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # График Accuracy (логарифмическая шкала)
    # plt.subplot(2, 2, 2)
    plt.figure()
    plt.plot(epochs[4:], train_acc[4:], label='Train Accuracy')
    plt.plot(epochs[4:], val_acc[4:], label='Val Accuracy')
    for epoch in best_epochs:
        plt.axvline(x=epoch, color='gray', linestyle='--', alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (log scale)')
    plt.title('Accuracy (Logarithmic Scale)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # График CER
    # plt.subplot(2, 2, 3)
    plt.figure()
    plt.plot(epochs[4:], val_cer[4:], label='Val CER', color='red')
    for epoch in best_epochs:
        plt.axvline(x=epoch, color='gray', linestyle='--', alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('CER')
    plt.title('Validation Character Error Rate')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Логарифмическая шкала для CER
    
    plt.tight_layout()
    plt.show()

log_data = """
Epoch 1/100
Train Loss: 1102.1610, Accuracy: 0.0000
Val Loss: 125.5289, Accuracy: 0.0000, CER: 1.0000
Epoch 2/100
Train Loss: 1071.0259, Accuracy: 0.0000
Val Loss: 74.2346, Accuracy: 0.0000, CER: 0.7741
Epoch 3/100
Train Loss: 293.4055, Accuracy: 0.4605
Val Loss: 12.2875, Accuracy: 0.9659, CER: 0.0026
Best model updated.
Epoch 4/100
Train Loss: 80.8356, Accuracy: 0.9870
Val Loss: 6.7010, Accuracy: 0.9931, CER: 0.0006
Best model updated.
Epoch 5/100
Train Loss: 54.9613, Accuracy: 0.9959
Val Loss: 4.8466, Accuracy: 0.9937, CER: 0.0006
Best model updated.
Epoch 6/100
Train Loss: 40.0554, Accuracy: 0.9968
Val Loss: 3.2109, Accuracy: 0.9943, CER: 0.0005
Best model updated.
Epoch 7/100
Train Loss: 30.2535, Accuracy: 0.9972
Val Loss: 1.8167, Accuracy: 0.9965, CER: 0.0003
Best model updated.
Epoch 8/100
Train Loss: 23.9782, Accuracy: 0.9976
Val Loss: 2.7544, Accuracy: 0.9924, CER: 0.0008
Epoch 9/100
Train Loss: 19.8650, Accuracy: 0.9968
Val Loss: 1.9340, Accuracy: 0.9963, CER: 0.0004
Epoch 10/100
Train Loss: 16.0441, Accuracy: 0.9979
Val Loss: 1.7255, Accuracy: 0.9969, CER: 0.0003
Best model updated.
Epoch 11/100
Train Loss: 13.1318, Accuracy: 0.9981
Val Loss: 0.7851, Accuracy: 0.9971, CER: 0.0003
Best model updated.
Epoch 12/100
Train Loss: 10.8213, Accuracy: 0.9980
Val Loss: 1.3747, Accuracy: 0.9973, CER: 0.0002
Best model updated.
Epoch 13/100
Train Loss: 8.8418, Accuracy: 0.9985
Val Loss: 0.0748, Accuracy: 0.9969, CER: 0.0003
Epoch 14/100
Train Loss: 7.3085, Accuracy: 0.9981
Val Loss: 0.4863, Accuracy: 0.9963, CER: 0.0004
Epoch 15/100
Train Loss: 5.8477, Accuracy: 0.9987
Val Loss: 0.5420, Accuracy: 0.9992, CER: 0.0001
Best model updated.
Epoch 16/100
Train Loss: 4.6083, Accuracy: 0.9990
Val Loss: -0.1553, Accuracy: 0.9961, CER: 0.0004
Epoch 17/100
Train Loss: 3.7919, Accuracy: 0.9989
Val Loss: 0.3810, Accuracy: 0.9961, CER: 0.0004
Epoch 18/100
Train Loss: 3.0773, Accuracy: 0.9989
Val Loss: 0.1718, Accuracy: 0.9980, CER: 0.0002
Epoch 19/100
Train Loss: 2.7503, Accuracy: 0.9981
Val Loss: 0.4842, Accuracy: 0.9963, CER: 0.0004
Epoch 20/100
Train Loss: 2.2519, Accuracy: 0.9985
Val Loss: 0.4618, Accuracy: 0.9965, CER: 0.0003
Epoch 21/100
Train Loss: 2.0130, Accuracy: 0.9984
Val Loss: -0.1212, Accuracy: 0.9967, CER: 0.0003
Epoch 22/100
Train Loss: 1.5876, Accuracy: 0.9988
Val Loss: 0.1805, Accuracy: 0.9978, CER: 0.0002
Epoch 23/100
Train Loss: 1.1516, Accuracy: 0.9995
Val Loss: -0.0299, Accuracy: 0.9980, CER: 0.0002
Epoch 24/100
Train Loss: 0.9663, Accuracy: 0.9995
Val Loss: 0.0448, Accuracy: 0.9975, CER: 0.0002
Epoch 25/100
Train Loss: 0.9527, Accuracy: 0.9991
Val Loss: -0.0915, Accuracy: 0.9965, CER: 0.0004
Epoch 26/100
Train Loss: 1.1619, Accuracy: 0.9981
Val Loss: 0.1092, Accuracy: 0.9959, CER: 0.0005
Epoch 27/100
Train Loss: 0.9519, Accuracy: 0.9985
Val Loss: 0.1121, Accuracy: 0.9980, CER: 0.0002
Epoch 28/100
Train Loss: 0.6027, Accuracy: 0.9996
Val Loss: -0.0026, Accuracy: 0.9984, CER: 0.0002
Epoch 29/100
Train Loss: 0.5049, Accuracy: 0.9994
Val Loss: 0.0996, Accuracy: 0.9984, CER: 0.0001
Epoch 30/100
Train Loss: 0.7001, Accuracy: 0.9987
Val Loss: 0.2647, Accuracy: 0.9980, CER: 0.0002
Epoch 31/100
Train Loss: 0.6549, Accuracy: 0.9988
Val Loss: 0.0005, Accuracy: 0.9982, CER: 0.0002
Epoch 32/100
Train Loss: 0.3875, Accuracy: 0.9995
Val Loss: 0.2852, Accuracy: 0.9978, CER: 0.0003
Epoch 33/100
Train Loss: 0.3026, Accuracy: 0.9997
Val Loss: 0.2297, Accuracy: 0.9980, CER: 0.0002
Epoch 34/100
Train Loss: 0.2885, Accuracy: 0.9996
Val Loss: 0.1530, Accuracy: 0.9975, CER: 0.0003
Epoch 35/100
Train Loss: 0.3001, Accuracy: 0.9995
Val Loss: -0.0970, Accuracy: 0.9988, CER: 0.0001
Epoch 36/100
Train Loss: 0.3309, Accuracy: 0.9991
Val Loss: 0.1940, Accuracy: 0.9975, CER: 0.0002
Epoch 37/100
Train Loss: 0.3690, Accuracy: 0.9992
Val Loss: -0.0130, Accuracy: 0.9973, CER: 0.0003
Epoch 38/100
Train Loss: 0.2116, Accuracy: 0.9996
Val Loss: 0.0381, Accuracy: 0.9988, CER: 0.0001
Epoch 39/100
Train Loss: 0.1123, Accuracy: 1.0000
Val Loss: 0.0573, Accuracy: 0.9988, CER: 0.0001
Epoch 40/100
Train Loss: 0.0982, Accuracy: 1.0000
Val Loss: 0.0337, Accuracy: 0.9990, CER: 0.0001
Epoch 41/100
Train Loss: 0.0871, Accuracy: 0.9999
Val Loss: 0.1634, Accuracy: 0.9982, CER: 0.0001
Epoch 42/100
Train Loss: 0.6075, Accuracy: 0.9978
Val Loss: 0.1121, Accuracy: 0.9978, CER: 0.0002
Epoch 43/100
Train Loss: 0.2558, Accuracy: 0.9992
Val Loss: 0.0812, Accuracy: 0.9982, CER: 0.0002
Epoch 44/100
Train Loss: 0.2323, Accuracy: 0.9992
Val Loss: 0.1822, Accuracy: 0.9967, CER: 0.0003
Epoch 45/100
Train Loss: 0.2130, Accuracy: 0.9993
Val Loss: 0.1121, Accuracy: 0.9975, CER: 0.0003
Epoch 46/100
Train Loss: 0.2106, Accuracy: 0.9994
Val Loss: -0.0814, Accuracy: 0.9986, CER: 0.0001
Epoch 47/100
Train Loss: 0.1300, Accuracy: 0.9996
Val Loss: 0.0136, Accuracy: 0.9988, CER: 0.0001
Epoch 48/100
Train Loss: 0.0650, Accuracy: 0.9999
Val Loss: 0.1060, Accuracy: 0.9988, CER: 0.0001
Epoch 49/100
Train Loss: 0.2550, Accuracy: 0.9991
Val Loss: 0.1816, Accuracy: 0.9980, CER: 0.0002
Epoch 50/100
Train Loss: 0.2387, Accuracy: 0.9992
Val Loss: -0.0631, Accuracy: 0.9988, CER: 0.0001
Epoch 51/100
Train Loss: 0.1637, Accuracy: 0.9995
Val Loss: 0.0692, Accuracy: 0.9988, CER: 0.0001
Epoch 52/100
Train Loss: 0.0952, Accuracy: 0.9998
Val Loss: 0.2269, Accuracy: 0.9973, CER: 0.0003
Epoch 53/100
Train Loss: 0.0480, Accuracy: 1.0000
Val Loss: 0.0894, Accuracy: 0.9992, CER: 0.0001
Epoch 54/100
Train Loss: 0.0365, Accuracy: 1.0000
Val Loss: 0.0095, Accuracy: 0.9994, CER: 0.0001
Best model updated.
Epoch 55/100
Train Loss: 0.0324, Accuracy: 1.0000
Val Loss: -0.0016, Accuracy: 0.9992, CER: 0.0001
Epoch 56/100
Train Loss: 0.0309, Accuracy: 1.0000
Val Loss: -0.0107, Accuracy: 0.9988, CER: 0.0001
Epoch 57/100
Train Loss: 0.0583, Accuracy: 0.9999
Val Loss: 0.0446, Accuracy: 0.9978, CER: 0.0002
Epoch 58/100
Train Loss: 0.3527, Accuracy: 0.9987
Val Loss: 0.0812, Accuracy: 0.9973, CER: 0.0003
Epoch 59/100
Train Loss: 0.2205, Accuracy: 0.9989
Val Loss: 0.0802, Accuracy: 0.9973, CER: 0.0002
Epoch 60/100
Train Loss: 0.0883, Accuracy: 0.9998
Val Loss: 0.0426, Accuracy: 0.9982, CER: 0.0002
Epoch 61/100
Train Loss: 0.0483, Accuracy: 0.9999
Val Loss: 0.0899, Accuracy: 0.9984, CER: 0.0002
Epoch 62/100
Train Loss: 0.1032, Accuracy: 0.9997
Val Loss: 0.0311, Accuracy: 0.9984, CER: 0.0002
Epoch 63/100
Train Loss: 0.2564, Accuracy: 0.9991
Val Loss: 0.1115, Accuracy: 0.9975, CER: 0.0003
Epoch 64/100
Train Loss: 0.0933, Accuracy: 0.9996
Val Loss: 0.0526, Accuracy: 0.9990, CER: 0.0001
Epoch 65/100
Train Loss: 0.0555, Accuracy: 0.9998
Val Loss: 0.0153, Accuracy: 0.9978, CER: 0.0002
Epoch 66/100
Train Loss: 0.1028, Accuracy: 0.9997
Val Loss: 0.0055, Accuracy: 0.9986, CER: 0.0001
Epoch 67/100
Train Loss: 0.0642, Accuracy: 0.9998
Val Loss: 0.0447, Accuracy: 0.9996, CER: 0.0000
Best model updated.
Epoch 68/100
Train Loss: 0.0265, Accuracy: 1.0000
Val Loss: -0.0825, Accuracy: 0.9994, CER: 0.0000
Epoch 69/100
Train Loss: 0.0174, Accuracy: 1.0000
Val Loss: 0.0496, Accuracy: 0.9996, CER: 0.0000
Epoch 70/100
Train Loss: 0.0748, Accuracy: 0.9997
Val Loss: 0.0804, Accuracy: 0.9933, CER: 0.0006
Epoch 71/100
Train Loss: 0.2810, Accuracy: 0.9987
Val Loss: 0.0421, Accuracy: 0.9992, CER: 0.0001
Epoch 72/100
Train Loss: 0.1012, Accuracy: 0.9996
Val Loss: 0.1255, Accuracy: 0.9986, CER: 0.0001
Epoch 73/100
Train Loss: 0.0862, Accuracy: 0.9997
Val Loss: 0.0012, Accuracy: 0.9988, CER: 0.0001
Epoch 74/100
Train Loss: 0.2455, Accuracy: 0.9988
Val Loss: 0.0262, Accuracy: 0.9984, CER: 0.0001
Epoch 75/100
Train Loss: 0.0744, Accuracy: 0.9998
Val Loss: 0.0572, Accuracy: 0.9980, CER: 0.0002
Epoch 76/100
Train Loss: 0.0310, Accuracy: 1.0000
Val Loss: 0.0158, Accuracy: 0.9990, CER: 0.0001
Epoch 77/100
Train Loss: 0.0168, Accuracy: 1.0000
Val Loss: 0.0309, Accuracy: 0.9988, CER: 0.0001
Epoch 78/100
Train Loss: 0.2071, Accuracy: 0.9991
Val Loss: 0.0363, Accuracy: 0.9978, CER: 0.0002
Epoch 79/100
Train Loss: 0.1240, Accuracy: 0.9996
Val Loss: -0.0142, Accuracy: 0.9975, CER: 0.0002
Epoch 80/100
Train Loss: 0.0567, Accuracy: 0.9998
Val Loss: 0.0812, Accuracy: 0.9986, CER: 0.0001
Epoch 81/100
Train Loss: 0.0183, Accuracy: 1.0000
Val Loss: -0.0012, Accuracy: 0.9990, CER: 0.0001
Epoch 82/100
Train Loss: 0.0151, Accuracy: 1.0000
Val Loss: 0.0076, Accuracy: 0.9990, CER: 0.0001
Epoch 83/100
Train Loss: 0.0118, Accuracy: 1.0000
Val Loss: 0.0693, Accuracy: 0.9990, CER: 0.0001
Epoch 84/100
Train Loss: 0.0141, Accuracy: 1.0000
Val Loss: -0.0438, Accuracy: 0.9990, CER: 0.0001
Epoch 85/100
Train Loss: 0.0096, Accuracy: 1.0000
Val Loss: 0.0164, Accuracy: 0.9992, CER: 0.0001
Epoch 86/100
Train Loss: 0.0094, Accuracy: 1.0000
Val Loss: 0.0480, Accuracy: 0.9994, CER: 0.0001
Epoch 87/100
Train Loss: 0.3199, Accuracy: 0.9986
Val Loss: 0.1570, Accuracy: 0.9978, CER: 0.0002
Epoch 88/100
Train Loss: 0.1425, Accuracy: 0.9993
Val Loss: 0.1099, Accuracy: 0.9980, CER: 0.0002
Epoch 89/100
Train Loss: 0.0764, Accuracy: 0.9997
Val Loss: -0.0092, Accuracy: 0.9988, CER: 0.0001
Epoch 90/100
Train Loss: 0.0407, Accuracy: 0.9998
Val Loss: 0.1262, Accuracy: 0.9986, CER: 0.0001
Epoch 91/100
Train Loss: 0.0831, Accuracy: 0.9996
Val Loss: 0.0144, Accuracy: 0.9978, CER: 0.0003
Epoch 92/100
Train Loss: 0.1301, Accuracy: 0.9994
Val Loss: -0.0006, Accuracy: 0.9990, CER: 0.0001
Epoch 93/100
Train Loss: 0.0417, Accuracy: 0.9999
Val Loss: 0.0518, Accuracy: 0.9982, CER: 0.0002
Epoch 94/100
Train Loss: 0.0703, Accuracy: 0.9998
Val Loss: 0.0946, Accuracy: 0.9975, CER: 0.0002
Epoch 95/100
Train Loss: 0.0176, Accuracy: 1.0000
Val Loss: 0.0258, Accuracy: 0.9980, CER: 0.0002
Epoch 96/100
Train Loss: 0.0093, Accuracy: 1.0000
Val Loss: 0.0281, Accuracy: 0.9978, CER: 0.0002
Epoch 97/100
Train Loss: 0.0490, Accuracy: 0.9998
Val Loss: 0.1196, Accuracy: 0.9980, CER: 0.0002
Epoch 98/100
Train Loss: 0.2844, Accuracy: 0.9987
Val Loss: -0.0432, Accuracy: 0.9967, CER: 0.0003
Epoch 99/100
Train Loss: 0.0922, Accuracy: 0.9997
Val Loss: 0.0070, Accuracy: 0.9980, CER: 0.0002
Epoch 100/100
Train Loss: 0.0596, Accuracy: 0.9998
Val Loss: 0.0400, Accuracy: 0.9986, CER: 0.0001
"""

parse_logs_and_plot(log_data)