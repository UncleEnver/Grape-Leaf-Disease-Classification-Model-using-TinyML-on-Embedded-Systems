import math, requests, sys
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import (
    Dense, InputLayer, Dropout, Flatten, Reshape, 
    BatchNormalization, ReLU, Conv2D, GlobalAveragePooling2D, 
    Add, Multiply, AveragePooling2D, GlobalMaxPooling2D,
    Activation, Concatenate, Resizing
)
import ei_tensorflow.training

# --- GrapeNet Modules Implementation ---

def cbam_block(x):
    channel = x.shape[-1]
    shared_layer_one = Dense(channel // 8, activation='relu', kernel_initializer='he_normal', use_bias=True)
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True)
    
    avg_pool = GlobalAveragePooling2D()(x)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = GlobalMaxPooling2D()(x)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    x = Multiply()([x, cbam_feature])
    
    # Spatial Attention
    avg_s = tf.keras.backend.mean(x, axis=-1, keepdims=True)
    max_s = tf.keras.backend.max(x, axis=-1, keepdims=True)
    concat = Concatenate(axis=-1)([avg_s, max_s])
    spatial_attention = Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    
    return Multiply()([x, spatial_attention])

def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    return ReLU()(x)

def rffb_module(x, filters):
    p_x = Conv2D(filters, 3, strides=2, padding='same')(x)
    p_x = BatchNormalization()(p_x)
    p_x = ReLU()(p_x)
    p_x = Conv2D(filters, 3, strides=1, padding='same')(p_x)
    p_x = BatchNormalization()(p_x)
    h_x = AveragePooling2D(pool_size=2, strides=2, padding='same')(x)
    return Concatenate()([p_x, h_x])

# --- Model Construction ---

INPUT_SHAPE = (160, 160, 3) 
inputs = Input(shape=INPUT_SHAPE, name='x_input')

x = Conv2D(64, 3, strides=2, padding='same')(inputs)
x = BatchNormalization()(x)
x = ReLU()(x)

# Stages
x = residual_block(x, 64)
x = rffb_module(x, 64)
x = cbam_block(x)

x = residual_block(x, 128)
x = residual_block(x, 128)
x = rffb_module(x, 128)
x = cbam_block(x)

x = residual_block(x, 256)
x = residual_block(x, 256)
x = rffb_module(x, 256)
x = cbam_block(x)

x = Conv2D(512, 1, strides=1)(x)
x = Conv2D(1024, 3, strides=2, padding='same')(x)
x = Conv2D(1280, 1, strides=1)(x)

x = GlobalAveragePooling2D()(x)
outputs = Dense(classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# --- Training Configuration ---

BATCH_SIZE = 16
INITIAL_EPOCHS = 60
LEARNING_RATE = 0.0002

# Safe Reshaping Function
def fix_shapes(images, labels):
    # Force the image to (160, 160, 3) just in case
    images = tf.ensure_shape(images, [160, 160, 3])
    return images, labels

# Prepare the datasets safely
# We map the fix_shapes function BEFORE batching
train_dataset = train_dataset.map(fix_shapes).batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.map(fix_shapes).batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# --- Phase 1: Initial Training ---
model.fit(train_dataset, 
          validation_data=validation_dataset, 
          epochs=INITIAL_EPOCHS, 
          verbose=2, 
          callbacks=callbacks)

print('\nInitial training done. Starting Fine-tuning...', flush=True)

# --- PHASE 2: FINE-TUNING ---
FINE_TUNE_EPOCHS = 20
FINE_TUNE_PERCENTAGE = 65 
FINE_TUNE_LR = 0.000045 

# Unfreeze the model 
model.trainable = True
model_layer_count = len(model.layers)
fine_tune_from = math.ceil(model_layer_count * ((100 - FINE_TUNE_PERCENTAGE) / 100))

# Freeze the early layers
for layer in model.layers[:fine_tune_from]:
    layer.trainable = False

# Re-compile is MANDATORY after changing 'trainable'
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(train_dataset,
          epochs=FINE_TUNE_EPOCHS,
          verbose=2,
          validation_data=validation_dataset,
          callbacks=callbacks,
          # Note: Y_train must be available in your scope for this helper
          class_weight=ei_tensorflow.training.get_class_weights(Y_train))
