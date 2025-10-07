import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

normal_data = np.random.normal(loc=0.0, scale=1.0, size=(1000, 20))
outliers = np.random.normal(loc=5.0, scale=1.0, size=(50, 20))
X = np.vstack([normal_data, outliers])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

input_dim = X.shape[1]
encoding_dim = 10

input_layer = layers.Input(shape=(input_dim,))
encoder = layers.Dense(encoding_dim, activation='relu')(input_layer)
decoder = layers.Dense(input_dim, activation='linear')(encoder)

autoencoder = models.Model(input_layer, decoder)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X_scaled[:1000], X_scaled[:1000],
                epochs=50,
                batch_size=32,
                shuffle=True,
                validation_split=0.1)

reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)


threshold = np.percentile(mse, 95)  
outlier_pred = mse > threshold

print(f"Detected outliers: {np.sum(outlier_pred)} out of {len(X)}")
