import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

class TrainingPipeline:
    def __init__(self, model, config):
        self.model = model
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 50)
    
    def get_data_genrator(self):
        """Create augmentation generators"""
        train_gen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            zoom_range=0.1,
        )

        val_gen = ImageDataGenerator()

        return train_gen, val_gen
    
    def get_callbacks(self, model_path):
        """Create training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=50,
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

        # Get augmentation generators
        train_gen, val_gen = self.get_data_genrator()

        # Create training generator
        train_generator = train_gen.flow(
            X_train,
            y_train,
            batch_size=self.batch_size
        )

        # Validation generator
        val_generator = val_gen.flow(
            X_val,
            y_val,
            batch_size=self.batch_size
        )

        # Get callbacks
        callbacks = self.get_callbacks(model_path)

        # Train model
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )

        return history

