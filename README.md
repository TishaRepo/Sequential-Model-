   git clone <repository-url>
   cd Sequential-Model-
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow keras numpy pandas scikit-learn
   ```

## Usage

### Building a Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

### Compiling the Model

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Training the Model

```python
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[tensorboard_callback, early_stopping_callback]
)
```

### Evaluating the Model

```python
loss, accuracy = model.evaluate(test_dataset)
print(f'Test Accuracy: {accuracy:.4f}')
```

### Saving and Loading

```python
# Save model
model.save('my_model.h5')

# Load model
from tensorflow.keras.models import load_model
loaded_model = load_model('my_model.h5')
```

## Examples

### Image Classification (MNIST)

```python
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_split=0.1)
```

### Text Classification (IMDb)

```python
# Load IMDb dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Preprocess data
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

# Build RNN model
model = Sequential([
    Embedding(10000, 16, input_length=200),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_split=0.1)
```

## Model Architecture

The Sequential model allows you to stack layers linearly:

```
Input Layer
  ↓
Layer 1
  ↓
Layer 2
  ↓
Layer 3
  ↓
Output Layer
```

## Common Layers

### Dense Layers

```python
# Fully connected layer
Dense(64, activation='relu')
```

### Convolutional Layers

```python
# 2D Convolutional layer
Conv2D(32, (3, 3), activation='relu')

# Max Pooling
MaxPooling2D((2, 2))
```

### Recurrent Layers

```python
# LSTM layer
LSTM(64)

# GRU layer
GRU(64)
```

### Embedding Layers

```python
# Embedding layer for text data
Embedding(input_dim=10000, output_dim=16, input_length=200)
```

## Callbacks

### TensorBoard

```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
```

### Early Stopping

```python
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

### Model Checkpointing

```python
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_accuracy',
    save_best_only=True
)
```

## Model Evaluation

```python
# Evaluate on test data
loss, accuracy = model.evaluate(x_test, y_test)

# Get predictions
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
```

## Model Saving

```python
# Save entire model (architecture + weights + optimizer state)
model.save('my_model.h5')

# Save only weights
model.save_weights('my_model_weights.h5')

# Save architecture only
json_config = model.to_json()
with open('model_config.json', 'w') as f:
    f.write(json_config)
```

## Model Loading

```python
# Load entire model
from tensorflow.keras.models import load_model
loaded_model = load_model('my_model.h5')

# Load weights only
new_model = create_model()
new_model.load_weights('my_model_weights.h5')

# Load from JSON config
from tensorflow.keras.models import model_from_json
with open('model_config.json', 'r') as f:
    loaded_model = model_from_json(f.read())
```

## Model Summary

```python
model.summary()