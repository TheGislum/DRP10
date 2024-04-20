import keras.losses
import keras.metrics
import tensorflow as tf
import keras
from keras import regularizers
from keras.constraints import NonNeg
from keras.regularizers import OrthogonalRegularizer, L2
from keras import layers
from tensorflow.keras.layers import (
    Input,
    Dense,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from keras import backend as K


class minimum_volume(keras.constraints.Constraint):
    def __init__(self, dim=15, beta=0.001):
        self.beta = beta
        self.dim = dim
        self.eye = K.eye(self.dim)

    def __call__(self, weights):
        w_matrix = K.dot(weights, K.transpose(weights))
        log_det = tf.linalg.logdet(w_matrix + self.eye)
        return self.beta * log_det

    def get_config(self):
        return {"dim": self.dim, "beta": float(self.beta)}


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + z_log_var * epsilon


class MUSE_XVAE(keras.Model):
    def __init__(
        self,
        input_dim=96,
        l_1=128,
        z=17,
        beta_v=0.05,
        beta_r=0.001,
        activation="softplus",
        reg="min_vol",
        refit=False,
        **kwargs
    ):
        """hybrid autoencoder due to non linear encoder and linear decoder;
        NonNegativity constraint for the decoder"""

        if reg == "min_vol":
            regularizer = minimum_volume(beta=beta_r, dim=z)
        elif reg == "ortogonal":
            regularizer = keras.regularizers.OrthogonalRegularizer(beta_r)
        elif reg == "L2":
            regularizer = keras.regularizers.L2(beta_r)

        if refit == True:
            activation = "relu"

        encoder = self.build_encoder(input_dim, l_1, z, activation, refit)
        decoder = self.build_decoder(input_dim, z, regularizer)
        super().__init__(encoder.input, decoder(encoder.output[2]), **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta_v = beta_v
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def build_encoder(self, input_dim, l_1, z, activation, refit):
        encoder_input = keras.Input(shape=(input_dim,))

        latent_1 = layers.Dense(l_1, activation=activation)(encoder_input)
        latent_1 = layers.LayerNormalization()(latent_1)
        latent_1 = layers.Dense(l_1 / 2, activation=activation)(latent_1)
        latent_1 = layers.LayerNormalization()(latent_1)
        latent_1 = layers.Dense(l_1 / 4, activation=activation)(latent_1)
        latent_1 = layers.LayerNormalization()(latent_1)

        z_mean = layers.Dense(z, activation="softplus", name="z_mean")(latent_1)
        z_log_var = layers.Dense(z, activation="softplus", name="z_log_var")(latent_1)
        signatures = Sampling()([z_mean, z_log_var])
        signatures = layers.ReLU(name="encoder_layer")(signatures)

        return keras.Model(
            encoder_input, [z_mean, z_log_var, signatures], name="encoder"
        )

    def build_decoder(self, input_dim, z, regularizer):
        latent_input = keras.Input(shape=(z,))
        decode = layers.Dense(
            input_dim,
            activation="linear",
            use_bias=False,
            kernel_constraint=keras.constraints.NonNeg(),
            kernel_regularizer=regularizer,
        )(latent_input)

        return keras.Model(latent_input, decode, name="decoder")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data[0])
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.MSE(data[0], reconstruction))
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + self.beta_v * kl_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.MSE(data[0], reconstruction))
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta_v * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def MUSE_XAE(
    input_dim=96,
    l_1=128,
    z=17,
    beta=0.001,
    activation="softplus",
    reg="min_vol",
    refit=False,
):
    """hybrid autoencoder due to non linear encoder and linear decoder;
    NonNegativity constraint for the decoder"""

    if reg == "min_vol":
        regularizer = minimum_volume(beta=beta, dim=z)
    elif reg == "ortogonal":
        regularizer = OrthogonalRegularizer(beta)
    elif reg == "L2":
        regularizer = L2(beta)

    if refit == True:
        activation = "relu"

    encoder_input = Input(shape=(input_dim,))

    latent_1 = Dense(l_1, activation=activation)(encoder_input)
    latent_1 = BatchNormalization()(latent_1)
    latent_1 = Dense(l_1 / 2, activation=activation)(latent_1)
    latent_1 = BatchNormalization()(latent_1)
    latent_1 = Dense(l_1 / 4, activation=activation)(latent_1)
    latent_1 = BatchNormalization()(latent_1)

    if refit == True:
        signatures = Dense(
            z,
            activation=activation,
            activity_regularizer=regularizers.l1(1e-3),
            name="encoder_layer",
        )(latent_1)
    else:
        signatures = Dense(z, activation="softplus", name="encoder_layer")(latent_1)

    decoder = Dense(
        input_dim,
        activation="linear",
        use_bias=False,
        kernel_constraint=NonNeg(),
        kernel_regularizer=regularizer,
    )(signatures)
    hybrid_dae = Model(encoder_input, decoder)

    return hybrid_dae


if __name__ == "__main__":
    data = tf.random.normal((1000, 96))
    model = MUSE_XVAE()
    model.compile(optimizer="adam")
    model.fit(data, data, epochs=10)
