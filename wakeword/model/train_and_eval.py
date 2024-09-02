import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
from augments import WakeWordDataset
from model import CNNGRU
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np

def load_data():
    dataset_df = pd.read_csv('my_data/df.csv')
    paths = dataset_df['path'].values
    labels = dataset_df['label'].values
    noise_dir = 'C:/Users/Jonah/Documents/Code/pyprogs/VoiceAssistedFanController/my_data/noise_recordings'
    data = WakeWordDataset(paths, labels, noise_dir, sample_rate=8000, max_length=10000)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader

def initialize_model():
    model = CNNGRU(num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    return model, device

def initialize_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # type: ignore
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion

def initialize_tensorboard():
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = f'results/logs/{current_time}'
    writer = SummaryWriter(log_dir)
    return writer

def train_one_epoch(model, train_loader, optimizer, criterion, device, writer, epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Log every 10 batches
        if i % 10 == 0: 
            writer.add_scalar('training_loss', running_loss / 10, epoch * len(train_loader) + i)
            running_loss = 0.0

def test(model, val_loader, criterion, device, writer, epoch):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate overall accuracy
    accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
    
    # Calculate precision, recall, and F1-score for each class
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    f1 = f1_score(all_labels, all_preds, average=None)
    
    # Log metrics to TensorBoard for each class
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        writer.add_scalar(f'validation_precision_class_{i}', p, epoch)
        writer.add_scalar(f'validation_recall_class_{i}', r, epoch)
        writer.add_scalar(f'validation_f1_score_class_{i}', f, epoch)

    # Log overall metrics
    writer.add_scalar('validation_loss', running_loss / len(val_loader), epoch)
    writer.add_scalar('validation_accuracy', accuracy, epoch)

    # Print the classification report
    print(classification_report(all_labels, all_preds, target_names=[f'Class {i}' for i in range(len(precision))]))

    print(f'Epoch {epoch}: Accuracy={accuracy:.2f}%')

def main():
    torch.manual_seed(42)

    train_loader, val_loader = load_data()

    # Initialize model, optimizer, and tensorboard
    model, device = initialize_model()
    optimizer, criterion = initialize_optimizer(model)
    writer = initialize_tensorboard()

    num_epochs = 50
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device, writer, epoch)
        print(f'Finished epoch {epoch}')
        
    test(model, val_loader, criterion, device, writer, epoch)

    writer.close()

    print('Finished Training')

if __name__ == "__main__":
    main()
