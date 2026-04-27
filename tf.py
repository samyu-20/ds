import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------------------------------
# 1. Data Preprocessing
# ------------------------------------------------
train_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_data = train_datagen.flow_from_directory(
    'dataset/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# ------------------------------------------------
# 2. Build CNN Model
# ------------------------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# ------------------------------------------------
# 3. Compile Model
# ------------------------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ------------------------------------------------
# 4. Train Model
# ------------------------------------------------
model.fit(train_data, epochs=5, validation_data=test_data)

# ------------------------------------------------
# 5. Evaluate Model
# ------------------------------------------------
loss, acc = model.evaluate(test_data)

print("\nTest Accuracy:", acc)
