import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense

# -----------------------------
# 1. Load Dataset
# -----------------------------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
data = pd.read_csv(url)
values = data['Temp'].values.reshape(-1,1)

# -----------------------------
# 2. Normalize Data
# -----------------------------
scaler = MinMaxScaler()
values = scaler.fit_transform(values)

# -----------------------------
# 3. Create Sequences
# -----------------------------
def create_dataset(dataset, time_steps=10):
    X, y = [], []
    for i in range(len(dataset)-time_steps):
        X.append(dataset[i:i+time_steps])
        y.append(dataset[i+time_steps])
    return np.array(X), np.array(y)

time_steps = 10
X, y = create_dataset(values, time_steps)

# -----------------------------
# 4. Train-Test Split
# -----------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# 5. Model Builder
# -----------------------------
def build_model(model_type):
    model = Sequential()
    
    if model_type == "RNN":
        model.add(SimpleRNN(32, input_shape=(time_steps, 1)))
    elif model_type == "LSTM":
        model.add(LSTM(32, input_shape=(time_steps, 1)))
    elif model_type == "GRU":
        model.add(GRU(32, input_shape=(time_steps, 1)))
    
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    return model

# -----------------------------
# 6. Train & Evaluate
# -----------------------------
results = {}

for model_type in ["RNN", "LSTM", "GRU"]:
    print(f"\nTraining {model_type}...")
    
    model = build_model(model_type)
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        verbose=0
    )
    
    training_time = time.time() - start_time
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    # Metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    
    results[model_type] = {
        "Loss": history.history['loss'][-1],
        "MSE": mse,
        "Training Time (sec)": training_time
    }

# -----------------------------
# 7. Print Comparison
# -----------------------------
print("\n📊 Final Comparison:")
for model, res in results.items():
    print(f"\n{model}")
    for k, v in res.items():
        print(f"{k}: {v:.4f}")