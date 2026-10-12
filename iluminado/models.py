"""Transfer-learning models for fundus images.

Every model takes raw pixels in 0-255 and does its own backbone-specific
scaling inside the network. That fixes the old mismatch where EfficientNet got
0-1 inputs, ResNeXt never got its preprocessing, and the app fed something
different again. One input format, everywhere.
"""
from __future__ import annotations

import numpy as np

import keras
from keras import layers

IMAGENET_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_VAR = [(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2]


def _efficientnet(cls):
    def build(shape, weights):
        return cls(include_top=False, weights=weights, input_shape=shape), []
    return build


def _convnext(cls):
    def build(shape, weights):
        return cls(include_top=False, weights=weights, input_shape=shape, include_preprocessing=True), []
    return build


def _densenet(cls):
    # DenseNet expects "torch" style input: scale to 0-1, then ImageNet mean/std.
    def build(shape, weights):
        pre = [layers.Normalization(mean=IMAGENET_MEAN, variance=IMAGENET_VAR, name="imagenet_norm")]
        return cls(include_top=False, weights=weights, input_shape=shape), pre
    return build


BACKBONES = {
    "efficientnet_b0": _efficientnet(keras.applications.EfficientNetB0),
    "efficientnet_b3": _efficientnet(keras.applications.EfficientNetB3),
    "densenet121": _densenet(keras.applications.DenseNet121),
    "convnext_tiny": _convnext(keras.applications.ConvNeXtTiny),
}


@keras.saving.register_keras_serializable(package="iluminado")
class WeightedBCE(keras.losses.Loss):
    """Binary cross-entropy with a per-label weight on the positive class.

    Handles imbalance for both tasks with one loss:
    - disease risk: 79% of images are positive, so positives are down-weighted
    - DR / MH / ODC: each is rare, so positives are up-weighted
    Works for one output (binary) or several independent outputs (multi-label).
    """

    def __init__(self, pos_weight, name="weighted_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos_weight = [float(w) for w in np.atleast_1d(pos_weight)]

    def call(self, y_true, y_pred):
        y_pred = keras.ops.clip(y_pred, 1e-7, 1 - 1e-7)
        w = keras.ops.convert_to_tensor(self.pos_weight, dtype=y_pred.dtype)
        loss = -(w * y_true * keras.ops.log(y_pred) + (1 - y_true) * keras.ops.log(1 - y_pred))
        return keras.ops.mean(loss, axis=-1)

    def get_config(self):
        return {**super().get_config(), "pos_weight": self.pos_weight}


def positive_weights(labels) -> np.ndarray:
    """neg/pos ratio per label column, so each class contributes equally."""
    y = np.asarray(labels, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    pos = y.sum(axis=0)
    return (len(y) - pos) / np.maximum(pos, 1)


def build_model(backbone: str, img_size: int, n_outputs: int, weights: str | None = "imagenet",
                dropout: float = 0.3) -> keras.Model:
    """Backbone + global pooling + dropout + sigmoid head.

    Sigmoid (not softmax) for both tasks: one unit gives the disease-risk
    probability; three units give independent DR / MH / ODC probabilities, so an
    image can be positive for more than one disease.
    """
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}. Choose from {sorted(BACKBONES)}")
    shape = (img_size, img_size, 3)
    base, pre_layers = BACKBONES[backbone](shape, weights)
    base.trainable = False

    inputs = keras.Input(shape=shape, name="image_0_255")
    x = inputs
    for layer in pre_layers:
        x = layer(x)
    x = base(x, training=False)  # keeps BatchNorm statistics frozen during fine-tuning
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(n_outputs, activation="sigmoid", dtype="float32", name="head")(x)
    return keras.Model(inputs, outputs, name=f"{backbone}_{n_outputs}out")


def get_backbone(model: keras.Model):
    """The pretrained network inside our model (the only nested Model layer).

    Found by type rather than by name, because renaming a Keras application
    changes the pretrained-weights filename it downloads.
    """
    nested = [layer for layer in model.layers if isinstance(layer, keras.Model)]
    if len(nested) != 1:
        raise ValueError(f"Expected exactly one backbone, found {[l.name for l in nested]}")
    return nested[0]


def unfreeze_top(model: keras.Model, fraction: float = 0.3) -> None:
    """Unfreeze the top `fraction` of backbone layers for fine-tuning.

    BatchNorm layers stay frozen: with small batches their statistics get noisy
    and wreck the pretrained features.
    """
    base = get_backbone(model)
    base.trainable = True
    cutoff = int(len(base.layers) * (1 - fraction))
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cutoff and not isinstance(layer, layers.BatchNormalization)


def compile_model(model: keras.Model, lr: float, pos_weight, n_outputs: int) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=WeightedBCE(pos_weight),
        metrics=[
            keras.metrics.AUC(name="auc", multi_label=n_outputs > 1, num_labels=n_outputs if n_outputs > 1 else None),
            keras.metrics.AUC(name="pr_auc", curve="PR", multi_label=n_outputs > 1,
                              num_labels=n_outputs if n_outputs > 1 else None),
        ],
    )


def train(model: keras.Model, train_ds, val_ds, pos_weight, n_outputs: int, ckpt_path: str,
          head_epochs: int = 5, finetune_epochs: int = 25, head_lr: float = 1e-3,
          finetune_lr: float = 3e-5, unfreeze_fraction: float = 0.3, patience: int = 5):
    """Two-stage transfer learning, with early stopping on the VALIDATION set.

    Stage 1 trains only the new head on frozen features. Stage 2 unfreezes the
    top of the backbone at a low learning rate. The test set is never touched
    here, so the final test score is a genuine held-out estimate.
    """
    compile_model(model, head_lr, pos_weight, n_outputs)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, verbose=2)

    unfreeze_top(model, unfreeze_fraction)
    compile_model(model, finetune_lr, pos_weight, n_outputs)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.3,
                                          patience=max(2, patience // 2), verbose=1),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_auc", mode="max", save_best_only=True),
    ]
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=finetune_epochs, callbacks=callbacks, verbose=2)
    return h1, h2


def load(path: str) -> keras.Model:
    return keras.models.load_model(path, compile=False)
