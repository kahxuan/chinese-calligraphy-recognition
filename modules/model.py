import tensorflow as tf
import yaml

config_path = 'config.yaml'
with open(config_path) as file:
    config = yaml.safe_load(file)


class CRModel(tf.keras.Model):

    def __init__(self, num_class):
        super(CRModel, self).__init__()
        self.num_class = num_class
        
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomRotation(config['model']['augmentation']['rotation']),
        ])
        
        self.preprocess_input = tf.keras.applications.efficientnet.preprocess_input

        self.base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            classes=self.num_class,
        )
        self.base_model.trainable = False
        self.pooling = tf.keras.layers.AveragePooling2D(
            pool_size=(4, 4), strides=(4, 4), padding="same")
        self.flat_layer = tf.keras.layers.Flatten()
        self.dense_layer = tf.keras.layers.Dense(config['model']['hidden_dense'], activation='relu')
        self.prediction_layer = tf.keras.layers.Dense(self.num_class, activation='softmax')
        

    def call(self, inputs, training=False):
        x = self.data_augmentation(inputs)
        x = self.preprocess_input(x)
        x = self.base_model(x, training=training)
        x = self.pooling(x)
        x = self.flat_layer(x)
        x = self.dense_layer(x)
        x = self.prediction_layer(x)
        return x


def load_cr_model(num_class, weights_path, optimizer=None):
    if not optimizer:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model = CRModel(num_class=100)
    model.build(input_shape=tuple([None] + config['model']['image_size'] + [3]))
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.load_weights(weights_path)
    return model