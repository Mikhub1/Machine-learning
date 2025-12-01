import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------
# Paths
# ---------------------------
data_dir = "Animals"   # contains cats/, dogs/, snakes/
classes = ["cats", "dogs", "snakes"]
img_size = (256, 256) #This can be reduced to (128, 128) or (64,64) for faster training. If this is done, it should be noted in the report. 

# Lists to store data
X_train_list = []
y_train_list = []
X_test_list = []
y_test_list = []

# ---------------------------
# Read each class separately
# ---------------------------
for label, cls in enumerate(classes):
    folder = os.path.join(data_dir, cls)
    files = os.listdir(folder)
    
    images = []
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, f)
            img = load_img(path, target_size=img_size)
            img = img_to_array(img) / 255.0
            images.append(img)
    
    images = np.array(images)
    labels = np.full(len(images), label)

    # ---------------------------
    # Split 80% train / 20% test
    # ---------------------------
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        images, labels, test_size=0.20, shuffle=True, random_state=42
    )

    # ---------------------------
    # Collect into global lists
    # ---------------------------
    X_train_list.append(X_train_cls)
    y_train_list.append(y_train_cls)
    X_test_list.append(X_test_cls)
    y_test_list.append(y_test_cls)

# ---------------------------
# Concatenate all classes
# ---------------------------
X_train = np.concatenate(X_train_list)
y_train = np.concatenate(y_train_list)
X_test = np.concatenate(X_test_list)
y_test = np.concatenate(y_test_list)

# Final data shapes
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ---------------------------
# Prepare labels for categorical classification
# ---------------------------
num_classes = len(classes)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# ---------------------------
# Build the neural network
# ---------------------------
model = models.Sequential()

# ---------------------------
# Input Layer
# ---------------------------
model.add(layers.InputLayer(input_shape=(256, 256, 3))) # the first two dimensions must correspond to the dimensions in line 17

# ---------------------------
# ---- Pure Dense NN ----
# You can comment/uncomment to experiment this order
# ---------------------------
# model.add(layers.Flatten())  # Flatten input first for Dense

# model.add(layers.Dense(10, activation='relu')) # Dense Layer 1 - can be replicated to create more layers

# model.add(layers.Dense(64, activation='relu')) # Dense Layer 2

# model.add(layers.Dense(20, activation='relu')) # Dense Layer 3

# ---------------------------
# ---- Pure CNN ----
# You can comment/uncomment to experiment this order
# ---------------------------
# Conv Layer 1
# model.add(layers.Conv2D(6, (3,3), activation='relu', padding='same'))
# model.add(layers.MaxPooling2D(2))

# Conv Layer 2
# model.add(layers.Conv2D(4, (3,3), activation='relu', padding='same')) # To create more convolutional layers, replicate lines 106-112
# model.add(layers.MaxPooling2D(2))

# Flatten for output
# model.add(layers.Flatten())

# ---------------------------
# ---- CNN + Dense  ----
# You can comment/uncomment layers to experiment
# ---------------------------
# Conv Layer 1
model.add(layers.Conv2D(6, (3,3), activation='relu', padding='same'))  # <-- edit filters/activation
model.add(layers.MaxPooling2D(2))

# Conv Layer 2
model.add(layers.Conv2D(4, (3,3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(2))

# Conv Layer 3
model.add(layers.Conv2D(2, (3,3), activation='relu', padding='same')) # To create more convolutional layers, replicate lines 122-131
model.add(layers.MaxPooling2D(2))

# Flatten
model.add(layers.Flatten())

# Dense Layer 1
model.add(layers.Dense(5, activation='relu'))  # <-- edit neurons/activation
# model.add(layers.Dropout(0.3)) #--->> optional: line 138 can also be used in the purely dense NN

# Dense Layer 2
model.add(layers.Dense(5, activation='relu'))

# Dense Layer 3 
# model.add(layers.Dense(64, activation='tanh'))



# ---------------------------
# Output layer
# ---------------------------
model.add(layers.Dense(num_classes, activation='softmax')) # keep the activation function as softmax


# ---------------------------
# Compile the model
# ---------------------------
# Edit optimizer and learning rate directly
optimizer = optimizers.Adam(learning_rate=1e-4)  # <-- try different optimizers and learning rates
# optimizer = optimizers.SGD(learning_rate=1e-3)
# optimizer = optimizers.RMSprop(learning_rate=5e-5)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

train_flag = True
test_flag = True 

if train_flag:
    # ---------------------------
    # Train the model
    # ---------------------------
    # Edit batch size and epochs directly
    history = model.fit(
        X_train, y_train_cat,
        validation_split=0.25,
        batch_size=32,  # <-- change batch size
        epochs=10       # <-- change epochs
    )
    
    # Plot training & validation accuracy
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    plt.figure(figsize=(8,10))
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy', marker='x')
    
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)  # Optional: show integer ticks for each epoch
    plt.legend()
    plt.grid(True)
    plt.show()
    
if test_flag:
    # ---------------------------
    # Evaluate
    # ---------------------------
    loss, acc = model.evaluate(X_test, y_test_cat)
    print(f"Test Accuracy: {acc*100:.2f}%")
    # Confusion matrix
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    
    # Optional: detailed metrics
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))