import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from segmentation_models import Unet
from segmentation_models.metrics import iou_score

# Assume X_train and y_train are your data
# X_train = ...
# y_train = ...

# Create a Unet model
model = Unet('resnet34', classes=1, activation='sigmoid')

# Compile the model
model.compile(optimizer=Adam(), loss=binary_crossentropy, metrics=[iou_score])

# Train the model
model.fit(
    x=X_train,
    y=y_train,
    batch_size=32,
    epochs=10
)
