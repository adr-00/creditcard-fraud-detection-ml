import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

from autoencoder import Autoencoder, train_autoencoder, get_reconstruction_error
from xgboost_model import train_xgboost, evaluate_xgboost

# Importo los datos de AE
X_test_AE = np.load('data/processed/X_test_AE.npy')
X_val_normal = np.load('data/processed/X_val_normal.npy')
X_train_normal = np.load('data/processed/X_train_normal.npy')
y_test = np.load('data/processed/y_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_val = np.load('data/processed/y_val.npy')

# Autoencoder
ae = Autoencoder(input_dim=29)
train_normal_tensor = torch.FloatTensor(X_train_normal)
val_normal_tensor = torch.FloatTensor(X_val_normal)

train_loader = DataLoader(train_normal_tensor, batch_size=64, shuffle=True)
val_loader = DataLoader(val_normal_tensor, batch_size=512, shuffle=True)

epochs = 100
criterion = nn.MSELoss()
optimizer = optim.Adam(ae.parameters(),lr=1e-4)
patience = 10

print('Empieza el entrenamiento de AE')
train_autoencoder(model=ae, dataloader_train=train_loader, dataloader_val=val_loader,criterion=criterion, 
                  optimizer=optimizer, patience=patience, epochs=epochs)
print('Finaliza el entrenamiento de AE')

# Entrenar Xgboost con recon_error
X_train_AE = np.load('data/processed/X_train_AE.npy')
X_val_AE = np.load('data/processed/X_val_AE.npy')
X_test_AE = np.load('data/processed/X_test_AE.npy')

X_train_tensor = torch.FloatTensor(X_train_AE)
X_val_tensor = torch.FloatTensor(X_val_AE)
X_test_tensor = torch.FloatTensor(X_test_AE)

recon_error_train = get_reconstruction_error(ae, X_train_tensor)
recon_error_val = get_reconstruction_error(ae, X_val_tensor)
recon_error_test = get_reconstruction_error(ae, X_test_tensor)

# Escalamos los datos de reconstruccion del AE
scaler = StandardScaler()
recon_error_train_scaled = scaler.fit_transform(recon_error_train.reshape(-1,1))
recon_error_test_scaled = scaler.transform(recon_error_test.reshape(-1,1))
recon_error_val_scaled = scaler.transform(recon_error_val.reshape(-1,1))

# Añadimos el feature a los datos
X_train_feature = np.concatenate([X_train_AE, recon_error_train_scaled.reshape(-1,1)], axis=1)
X_val_feature = np.concatenate([X_val_AE, recon_error_val_scaled.reshape(-1,1)], axis=1)
X_test_feature = np.concatenate([X_test_AE, recon_error_test_scaled.reshape(-1,1)], axis=1)

# Parametros
total_fraude = int(y_train.sum())
total_normal = int(len(y_train) - total_fraude)

params = {
    'objective': 'binary:logistic',  
    'eval_metric': 'aucpr',
    'scale_pos_weight': total_normal / total_fraude,
    'max_depth': 6,
    'learning_rate': 0.1
}
# Entrenamos el modelo XGB
xgb_model = train_xgboost(X_train_feature, y_train, X_val_feature, y_val, params=params)

# Obtenemos las metricas
precision, recall, aucpr = evaluate_xgboost(xgb_model, X_test_feature, y_test)
print("AUCPR XGBoost: ", aucpr)
print("Pracision: ", precision)
print("Recall: ", recall)