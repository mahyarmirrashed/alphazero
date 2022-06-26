"""Defines the AlphaZero Tensorflow model."""


import tensorflow as tf
from keras import Input, Model
from keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, ReLU, Reshape
from keras.optimizer_v2.gradient_descent import SGD
from pydantic import BaseModel, NonNegativeInt

from interface.problem.game import Game

DEFAULT_FILTERS = 256
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_MOMENTUM = 0.9
DEFAULT_RESIDUAL_LAYERS = 40

POLICY_HEAD_FILTERS = 2
VALUE_HEAD_FILTERS = 1
VALUE_HEAD_HIDDEN_LAYER_SIZE = 256


def _softmax_cross_entropy_with_logits(y_true, y_pred):
  """Define loss function for evaluating policy head."""
  return tf.nn.softmax_cross_entropy_with_logits(
    labels=y_true,
    logits=tf.where(
      tf.equal(y_true, tf.zeros(shape=tf.shape(y_true), dtype=tf.float32)),
      tf.fill(tf.shape(y_true), -100.0),
      y_pred,
    ),
  )


class AlphaZero(BaseModel):
  """AlphaZero model derived from schematic shown at https://adspassets.blob.core.windows.net/website/content/alpha_go_zero_cheat_sheet.png."""

  game: Game

  filters: NonNegativeInt = DEFAULT_FILTERS
  residual_layers: NonNegativeInt = DEFAULT_RESIDUAL_LAYERS

  def __init__(self, *args, **kwargs):
    """Initialize AlphaZero model."""
    super().__init__(*args, **kwargs)

    model_input = Input(shape=self.game.board.size())
    model = Reshape((*self.game.board.size(), 1))(model_input)

    # pass through convolutional layer
    model = self._convolution_layer(model)
    # pass through residual layers
    for _ in range(self.residual_layers):
      model = self._residual_layer(model)
    # diverge into policy and value heads
    pi = self._policy_head(model)
    v = self._value_head(model)

    model = Model(inputs=[model_input], outputs=[v, pi])
    model.compile(
      optimizer=SGD(learning_rate=DEFAULT_LEARNING_RATE, momentum=DEFAULT_MOMENTUM),
      loss={"v": "mean_squared_error", "pi": _softmax_cross_entropy_with_logits},
      loss_weights={"v": 0.5, "pi": 0.5},
    )

  def _convolution_layer(self, model):
    """Convolutional layer to start model."""
    model = Conv2D(
      filters=self.filters,
      kernel_size=3,
      strides=1,
      padding="same",
    )(model)
    model = BatchNormalization()(model)
    model = ReLU()(model)

    return model

  def _residual_layer(self, model):
    """Residual layer with skip connection."""
    skip = model

    model = Conv2D(
      filters=self.filters,
      kernel_size=3,
      strides=1,
      padding="same",
    )(model)
    model = BatchNormalization()(model)
    model = ReLU()(model)
    model = Conv2D(
      filters=self.filters,
      kernel_size=3,
      strides=1,
      padding="same",
    )(model)
    model = BatchNormalization()(model)
    model = Add()([model, skip])
    model = ReLU()(model)

    return model

  def _policy_head(self, model):
    """Policy head that decides which move to take."""
    model = Conv2D(
      filters=POLICY_HEAD_FILTERS,
      kernel_size=1,
      strides=1,
      padding="same",
    )(model)
    model = BatchNormalization()(model)
    model = ReLU()(model)
    model = Flatten()(model)
    model = Dense(
      units=self.game.total_actions(),
      activation="softmax",
      name="pi",
    )(model)

    return model

  def _value_head(self, model):
    """Value head that decides the value of a certain move."""
    model = Conv2D(
      filters=VALUE_HEAD_FILTERS,
      kernel_size=1,
      strides=1,
      padding="same",
    )(model)
    model = BatchNormalization()(model)
    model = ReLU()(model)
    model = Flatten()(model)
    model = Dense(units=VALUE_HEAD_HIDDEN_LAYER_SIZE)(model)
    model = ReLU()(model)
    model = Dense(units=1, activation="tanh", name="v")(model)

    return model
