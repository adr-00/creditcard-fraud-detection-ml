import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
    

def train_autoencoder(model, dataloader, criterion, optimizer, epochs=50):
    model.train()

    for epoch in range(epochs):
        train_loss = 0.0
        print(epoch)
        for batch in dataloader:
            optimizer.zero_grad() # Resetrar el gradiente acumulado a 0
            output = model(batch) # Forward Pass
            loss = criterion(output, batch) # Calculamos la perdida
            loss.backward() # Calculamos los gradientes
            optimizer.step() # Actualizamos los pesos
            train_loss += loss.item() # Acumulamos la perdida del batch
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(dataloader):.4f}')
        
    return model

def get_reconstruction_error(model, X_test_tensor):
    model.eval()
    with torch.no_grad():
        recon = model(X_test_tensor)
        error = ((recon - X_test_tensor)**2).mean(dim=1)
    return error.numpy() 