import torch
from torch import nn
import copy
from sklearn.metrics import average_precision_score


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
        return self.decoder(self.encoder(x))    

def train_autoencoder(model, dataloader_train, dataloader_val, criterion, 
                      optimizer, patience, epochs=50):
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0

    history_train = []
    history_val = []

    model.train()

    for epoch in range(epochs):
        train_loss = 0.0
        for batch in dataloader_train:
            optimizer.zero_grad() # Resetrar el gradiente acumulado a 0
            output = model(batch) # Forward Pass
            loss = criterion(output, batch) # Calculamos la perdida
            loss.backward() # Calculamos los gradientes
            optimizer.step() # Actualizamos los pesos
            train_loss += loss.item() # Acumulamos la perdida del batch
        history_train.append(loss.item())
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dataloader_val:
                recon = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item()
        history_val.append(loss_val.item())
        
        model.train()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(dataloader_train):.4f}',
              f'Val Loss: {val_loss/len(dataloader_val):.4f}')
    
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
    model.load_state_dict(best_model_wts)


def get_reconstruction_error(model, X_test_tensor):
    model.eval()
    with torch.no_grad():
        recon = model(X_test_tensor)
        error = ((recon - X_test_tensor)**2).mean(dim=1)
    return error.numpy() 

def evaluate_autoencoder(recon_error, y_test):
    threshold = recon_error.mean() + 3 * recon_error.std()

    y_pred = (recon_error >= threshold).astype(int)

    aucpr = average_precision_score(y_test, y_pred)

    return aucpr