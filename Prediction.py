import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# ========== Definition of Model Architecture ==========
class MultiHeadAttentionFeatureExtractor(nn.Module):
    def __init__(self, input_dim, num_heads=2):
        super().__init__()
        assert input_dim % num_heads == 0
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        attn_output, _ = self.multihead_attn(x, x, x)
        x = attn_output.squeeze(1)
        return self.feature_projection(x)

class MultiHeadAttentionFeatureExtractor1(nn.Module):
    def __init__(self, input_dim, num_heads=1):
        super().__init__()
        assert input_dim % num_heads == 0
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        attn_output, _ = self.multihead_attn(x, x, x)
        x = attn_output.squeeze(1)
        return self.feature_projection(x)

class DNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(512),
            nn.Dropout(0),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(512),
            nn.Dropout(0),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(512),
            nn.Dropout(0),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(512),
            nn.Dropout(0),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(256),
            nn.Dropout(0),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(128),
            nn.Dropout(0),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(64),
            nn.Dropout(0),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(32),
            nn.Dropout(0),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

class CombinedModel(nn.Module):
    def __init__(self, cnn_input_dim, dnn_input_dim, struct):
        super().__init__()
        self.attention = MultiHeadAttentionFeatureExtractor(cnn_input_dim)
        self.attentionstruct = MultiHeadAttentionFeatureExtractor1(struct)
        self.dnn = DNN(dnn_input_dim + 32)

    def forward(self, X_struct, x_cnn, x_dnn):
        attn_cnn = self.attention(x_cnn)
        attn_struct = self.attentionstruct(X_struct)
        combined = torch.cat([attn_cnn, attn_struct, x_dnn], dim=1)
        return self.dnn(combined)

# ========== Data Loading and Preprocessing ==========
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, delimiter=',', encoding='utf-8')

    categories1 = ['C1', 'C2', 'C3', 'RM1', 'RM2', 'S1', 'S2', 'S3', 'S4', 'S5', 'URM', 'W1', 'W2']
    df['strutype'] = pd.Categorical(df['strutype'], categories=categories1, ordered=False)
    df = pd.get_dummies(df, columns=['strutype'], drop_first=False)

    print(df.shape)
    df = df.astype('float64')
    df.loc[df['M_Bracketed Duration'] == 0, 'M_Bracketed Duration'] = 0.0001
    df.loc[df['Bracketed Duration'] == 0, 'Bracketed Duration'] = 0.0001
    df = df.dropna()
    df = df[(df >= 0).all(axis=1)]

    for i in range(41):
        df.iloc[:, i] = np.log(df.iloc[:, i])

    df = df.replace([float('inf'), float('-inf')], np.nan).dropna()
    print(df.shape)

    X_cnn = df.iloc[:, 3:41].values
    X_dnn = df.iloc[:, 0:3].values
    X_struct = df.iloc[:, 41:].values
    print("X_cnn shape：", X_cnn.shape)
    print("X_dnn shape：", X_dnn.shape)
    print("X_struct shape：", X_struct.shape)
    return torch.FloatTensor(X_struct), torch.FloatTensor(X_cnn), torch.FloatTensor(X_dnn)

# ========== Load Model and Make Predictions ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
X_struct, X_cnn, X_dnn = load_and_preprocess("data.csv")

# Initialize model and load weights
model = CombinedModel(
    cnn_input_dim=X_cnn.shape[1],
    dnn_input_dim=X_dnn.shape[1],
    struct=X_struct.shape[1]
).to(device)

model.load_state_dict(torch.load('.\model\model_PTFA.pth', map_location=device))
model.eval()

# Make predictions
with torch.no_grad():
    preds = model(X_struct.to(device), X_cnn.to(device), X_dnn.to(device)).cpu().numpy()

# Save the prediction results
np.savetxt('predictions.txt', preds, fmt='%.6f')
print("Prediction completed, results have been saved to predictions.txt")