import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

class SignLanguageModel_ResNet50:
    def __init__(self, num_classes, input_shape=(224,224,3)):
        self.num_classes = num_classes
        self.input_shape = input_shape

    def build_model(self):
        """Build ResNet50 transfer learning model"""
        #load pretrain ResNet50 model
        backbone = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze early layers (only fine-tune last layers)
        for layer in backbone.layers[:100]:
            layer.trainable = False

        # Add custom classification head
        model = tf.keras.Sequential([
            backbone,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model

    def compile_model(self, model):
        """Compile with optimizer and loss"""
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
        )

        return model
        

# testing model building and compilation
model_builder_resnet50 = SignLanguageModel_ResNet50(num_classes=24)
model = model_builder_resnet50.build_model()
model = model_builder_resnet50.compile_model(model)
print(model.summary())