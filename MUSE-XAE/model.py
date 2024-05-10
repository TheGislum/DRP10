import keras.losses
import keras.metrics
import keras.utils
import tensorflow as tf
import keras
import numpy as np
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


class fn_Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, K=1, name="sampling", **kwargs):
        super(fn_Sampling, self).__init__(name=name, **kwargs)
        self.K = K

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        z_mean = tf.expand_dims(z_mean, 1)
        z_log_var = tf.expand_dims(z_log_var, 1)
        epsilon = tf.random.normal(shape=(batch, self.K, dim))
        return tf.abs(z_mean + tf.exp(0.5 * z_log_var) * epsilon)


class g_Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, K=1, name="sampling", **kwargs):
        super(g_Sampling, self).__init__(name=name, **kwargs)
        self.K = K

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        z_mean = tf.expand_dims(z_mean, 1)
        z_log_var = tf.expand_dims(z_log_var, 1)
        epsilon = tf.random.normal(shape=(batch, self.K, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class MUSE_XVAE(keras.Model):
    def __init__(
        self, input_dim=96, z=17, beta_v=1e-5, beta_r=0.001, Kx=10, dist="fn", **kwargs
    ):
        """hybrid autoencoder due to non linear encoder and linear decoder;
        NonNegativity constraint for the decoder"""

        self.input_dim = input_dim
        self.beta_r = beta_r
        self.Kx = Kx
        self.z = z
        self.activation = "gelu"
        self.dist = dist

        encoder = self.build_encoder()
        decoder = self.build_decoder()
        super().__init__(encoder.input, decoder(encoder.output[2]), **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta_v = beta_v
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def build_encoder(self):
        encoder_input = keras.Input(shape=(self.input_dim,))

        latent_1 = layers.Dense(self.input_dim, activation=self.activation)(
            encoder_input
        )
        latent_1 = layers.LayerNormalization()(latent_1)
        latent_1 = layers.Dense(int(self.input_dim / 2), activation=self.activation)(
            latent_1
        )
        latent_1 = layers.LayerNormalization()(latent_1)
        latent_1 = layers.Dense(int(self.input_dim / 4), activation=self.activation)(
            latent_1
        )
        latent_1 = layers.LayerNormalization()(latent_1)

        z_mean = layers.Dense(self.z, name="z_mean")(latent_1)
        z_log_var = layers.Dense(self.z, name="z_log_var")(latent_1)
        if self.dist == "fn":
            signatures = fn_Sampling(K=self.Kx, name="encoder_layer")(
                [z_mean, z_log_var]
            )
        else:
            signatures = g_Sampling(K=self.Kx, name="encoder_layer")(
                [z_mean, z_log_var]
            )

        return keras.Model(
            encoder_input, [z_mean, z_log_var, signatures], name="encoder"
        )

    def build_decoder(self):
        latent_input = keras.Input(
            shape=(
                self.Kx,
                self.z,
            )
        )
        decode = layers.Dense(
            self.input_dim,
            activation="linear",
            use_bias=False,
            kernel_constraint=keras.constraints.NonNeg(),
            kernel_regularizer=minimum_volume(beta=self.beta_r, dim=self.z),
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
        data = tf.expand_dims(data[0], 1)
        reconstruction_loss = tf.reduce_mean(keras.losses.poisson(data, reconstruction))
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
            data = tf.expand_dims(data[0], 1)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.poisson(data, reconstruction)
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

    def get_f_mean(self, data, k=1000):
        self.encoder.layers[-1].K = k
        mu, log_var, Z = self.encoder.predict(data)
        if self.dist == "fn":
            std = tf.exp(0.5 * log_var)
            f_mean = std * tf.sqrt(2 / np.pi) * tf.exp(
                -0.5 * (mu**2 / std**2)
            ) + mu * tf.math.erf(mu / (std * np.sqrt(2)))
        else:
            f_mean = mu
        return f_mean, Z, mu, log_var


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
