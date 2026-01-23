import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np

from autoencoder import Autoencoder, train_autoencoder, get_reconstruction_error
from xgboost_model import train_xgboost, evaluate_xgboost

# Importo los datos
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')
X_train_normal = np.load('data/processed/X_train_normal.npy')
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')

# Autoencoder
ae = Autoencoder(input_dim=30)
train_normal_tensor = torch.FloatTensor(X_train_normal)
dataloader = DataLoader(train_normal_tensor, batch_size=256, shuffle=True)
criterion = nn.MSELoss()
optimizer = optim.Adam(ae.parameters(),lr=1e-3)

print('Empieza el entrenamiento de AE')
train_autoencoder(model=ae, dataloader=dataloader, criterion=criterion, optimizer=optimizer)
print('Finaliza el entrenamiento de AE')

# Entrenar Xgboost con recon_error
train_tensor = torch.FloatTensor(X_train)
recon_error = get_reconstruction_error(ae, train_tensor)
X_features = np.concatenate([X_train, recon_error.reshape(-1,1)], axis=1)

total_fraude = int(y_train.sum())
total_normal = int(len(y_train) - total_fraude)

params = {
    'objective': 'binary:logistic',  
    'eval_metric': 'aucpr',
    'scale_pos_weight': total_normal / total_fraude,
    'max_depth': 6,
    'learning_rate': 0.1
}
xgb_model = train_xgboost(X_features, y_train, X_val, y_val, params=params)

test_tensor = torch.FloatTensor(X_test)
recon_error_test = get_reconstruction_error(ae, test_tensor)
X_test_features = np.concatenate([X_test, recon_error_test.reshape(-1,1)], axis=1)

precision, recall, aucpr = evaluate_xgboost(xgb_model, X_test_features, y_test)
print("AUCPR XGBoost: ", aucpr)
print("Pracision: ", precision)
print("Recall: ", recall)