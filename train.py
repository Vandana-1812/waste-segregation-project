import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

# Paths
dataset_dir = "dataset"
model_save_path = "models/waste_model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10  # reduce if CPU is slow

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, "validation"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save model
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"Model saved at {model_save_path}")