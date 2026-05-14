import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

class TrainingPipeline:
    def __init__(self, model, config):
        self.model = model
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 10)
    
    def get_data_genrator(self):
        """Create augmentation generators"""
        train_gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=False,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2]
        )

        val_gen = ImageDataGenerator()

        return train_gen, val_gen
    
    def get_callbacks(self, model_path):
        """Create training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ), 
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                min_lr= 0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, model_path):
        """Train the model"""
        # train_gen, val_gen = self.get_data_genrator()
        callbacks = self.get_callbacks(model_path)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

