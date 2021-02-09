import tensorflow as tf
import yaml

config_path = 'config.yaml'
with open(config_path) as file:
    config = yaml.safe_load(file)['model']

IMG_SIZE = config['image_size']
IMG_SHAPE = IMG_SIZE + [3]
INPUT_SHAPE = tuple([None] + IMG_SHAPE)

class CRModel(tf.keras.Model):

    def __init__(self, num_class):
        super(CRModel, self).__init__()
        self.num_class = num_class
        
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomRotation(config['data_augmentation']['rotation']),
        ])
        self.preprocess_input = tf.keras.applications.vgg19.preprocess_input
        self.base_model = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            classes=self.num_class,
        )
        self.base_model.trainable = False
        self.fc_layer = tf.keras.layers.Flatten()
        self.dense_layer = tf.keras.layers.Dense(2048, activation='relu')
        self.prediction_layer = tf.keras.layers.Dense(self.num_class, activation='softmax')
        

    def call(self, inputs, training=False):
        x = self.data_augmentation(inputs)
        x = self.preprocess_input(x)
        x = self.base_model(x, training=training)
        x = self.fc_layer(x)
        x = self.dense_layer(x)
        x = self.prediction_layer(x)
        return x


def load_cr_model(num_class, weights_path, optimizer=None):
    if not optimizer:
        optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    model = CRModel(num_class=100)
    model.build(input_shape=INPUT_SHAPE)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.load_weights(weights_path)
    return model