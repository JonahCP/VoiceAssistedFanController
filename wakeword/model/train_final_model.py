import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
from augments import WakeWordDataset
from model import CNNGRU
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime

def initialize_model():
    model = CNNGRU(num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, device

def initialize_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion

def initialize_tensorboard():
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = f'results/logs/{current_time}'
    writer = SummaryWriter(log_dir)
    return writer

def train_full_model(model, train_loader, optimizer, criterion, device, writer, num_epochs):
    for epoch in range(num_epochs):
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

        # Save the model checkpoint at each epoch
        torch.save(model.state_dict(), f"results/checkpoints/model_epoch_{epoch}.pth")
        print(f"Epoch {epoch} completed and model saved.")

    writer.add_text('Training Summary', f'Training completed for {num_epochs} epochs.')

def main():
    dataset_df = pd.read_csv('my_data/df.csv')
    paths = dataset_df['path'].values
    labels = dataset_df['label'].values
    noise_dir = 'my_data/noise_recordings'

    data = WakeWordDataset(paths, labels, noise_dir, sample_rate=8000, max_length=10000)
    train_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

    # Initialize model, optimizer, and loss function
    model, device = initialize_model()  # Assuming initialize_model is defined
    optimizer, criterion = initialize_optimizer(model) 

    # Initialize TensorBoard
    writer = initialize_tensorboard() 

    # Train the full model
    num_epochs = 50  # or whatever you deem necessary
    train_full_model(model, train_loader, optimizer, criterion, device, writer, num_epochs)

if __name__ == "__main__":
    main()