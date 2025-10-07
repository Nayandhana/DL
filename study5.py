import tensorflow as tf
import numpy as np

# XOR dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Define the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),       # ✅ Correct way
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train (reduced epochs for faster training)
model.fit(X, Y, epochs=500, verbose=1)  # ✅ 500 instead of 5000

# Predict
preds = model.predict(X)
print("\nPredictions:")
print(np.round(preds))
