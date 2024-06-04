import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.utils import to_categorical

class CNNModel:
    def __init__(self, input_shape1, input_shape2, num_classes, activation, optimizer, loss):
        self.input_shape1 = input_shape1
        self.input_shape2 = input_shape2
        if activation == 'sigmoid':
            self.num_classes = 1
        else:
            self.num_classes = num_classes

        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss

        self.model = self.build_model()

    def build_model(self):
        # Input layers
        input1 = Input(shape=self.input_shape1)
        input2 = Input(shape=self.input_shape2)

        # First branch for signal1
        x1 = Conv1D(filters = 32, kernel_size = 3, activation='relu')(input1)
        x1 = MaxPooling1D(pool_size = 2)(x1)
        x1 = Conv1D(filters = 64, kernel_size = 3, activation='relu')(x1)
        x1 = MaxPooling1D(pool_size = 2)(x1)
        x1 = Flatten()(x1)

        # Second branch for signal2
        x2 = Conv1D(filters = 32, kernel_size = 3, activation='relu')(input2)
        x2 = MaxPooling1D(pool_size = 2)(x2)
        x2 = Conv1D(filters = 64, kernel_size = 3, activation='relu')(x2)
        x2 = MaxPooling1D(pool_size = 2)(x2)
        x2 = Flatten()(x2)

        # Concatenate both branches
        x = concatenate([x1, x2])
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes, activation = self.activation)(x)

        # Create model
        model = Model(inputs=[input1, input2], outputs=output)
        model.compile(optimizer = self.optimizer, loss = self.loss, metrics=['accuracy'])
        return model

    def train(self, train_data, train_labels, val_data, val_labels, epochs, batch_size):
        self.model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=epochs, batch_size=batch_size)

    def evaluate(self, test_data, test_labels):
        return self.model.evaluate(test_data, test_labels)

    def predict(self, new_data):
        return self.model.predict(new_data)