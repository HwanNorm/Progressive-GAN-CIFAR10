# -*- coding: utf-8 -*-
# Import libraries
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from keras import Model, layers, models, callbacks, optimizers
from keras.datasets import cifar10
from keras.layers import (
    Dropout, Flatten, Dense, BatchNormalization, LeakyReLU, Reshape,
    Conv2D, Conv2DTranspose
)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

from numpy import zeros, ones, vstack, expand_dims
from numpy.random import randn, randint
from sklearn.model_selection import train_test_split

"""# First-level requirement
First-level requirement: train and generate images for ONE class of the dataset using vanilla GAN. You are free to choose which class of object you are interested in working on.
"""

# Load CIFAR-10 dataset more efficiently
cifar10_train = tfds.load(name="cifar10", split=tfds.Split.TRAIN, batch_size=-1)
x_train = tf.cast(cifar10_train['image'], tf.float32)
y_train = tf.cast(cifar10_train['label'], tf.int32)

# Print shapes of the entire training and test set of CIFAR 10
print("X_train shape: " + str(x_train.shape))
print("y_train shape: " + str(y_train.shape))

# Get samples  of a certain label
# Select only one class
selected_class = 1
mask = (y_train == selected_class)
x_train = tf.boolean_mask(x_train, mask)
x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]

# Visualize some data samples of class '1' (car)
# Convert tensor to numpy for visualization
x_train_np = x_train.numpy()

# Plot a grid of images
num_images = 16  # Number of images to display
plt.figure(figsize=(10, 10))
for i in range(num_images):
    plt.subplot(4, 4, i + 1)
    plt.imshow((x_train_np[i] + 1) / 2)  # Rescale to [0, 1] for visualization
    plt.axis('off')
    plt.title("Car")
plt.tight_layout()
plt.show()

"""## 1. Coding tasks

# Vanilla GAN
"""

BUFFER_SIZE = 50000
BATCH_SIZE = 64

# Create a dataset
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#####################################
# Task 1.2: Build GAN Architecture #
#####################################

# Generator model
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((8, 8, 256)))

    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (4,4), strides=(1,1), padding='same', activation='tanh'))

    return model

generator = build_generator()

# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (4,4), strides=(2,2), padding='same', input_shape=[32, 32, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = build_discriminator()

# Loss functions
def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(5e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(5e-5)

# Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

#################################################
# Task 1.3: Visualize Model Training Behaviors #
#################################################

# Track loss
gen_losses = []
disc_losses = []

def train(dataset, epochs=50):
    for epoch in range(epochs):
        epoch_gen_loss = 0
        epoch_disc_loss = 0

        for image_batch in dataset:
            train_step(image_batch)
            noise = tf.random.normal([BATCH_SIZE, 100])
            generated_images = generator(noise, training=True)
            real_output = discriminator(image_batch, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

            epoch_gen_loss += gen_loss.numpy()
            epoch_disc_loss += disc_loss.numpy()

        gen_losses.append(epoch_gen_loss / len(dataset))
        disc_losses.append(epoch_disc_loss / len(dataset))

        print(f'Epoch {epoch+1} completed')
        generate_and_save_images(generator, epoch+1)

    plot_loss()

def plot_loss():
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label="Generator Loss")
    plt.plot(disc_losses, label="Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    plt.savefig("gan_loss_plot.png")
    plt.show()

########################################
# Task 1.4: Visualize Generated Images #
########################################

def generate_and_save_images(model, epoch):
    noise = tf.random.normal([16, 100])
    generated_images = model(noise, training=False)
    generated_images = (generated_images + 1) / 2  # Rescale to [0,1] for visualization

    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i])
        ax.axis('off')

    plt.savefig(f'generated_image_at_epoch_{epoch}.png')
    plt.show()

################
# Start Training #
################

# Train the GAN
train(train_dataset, epochs=400)

"""# ProGAN"""

# Display some of the cat images
print("Displaying sample car images from the dataset:")
x_train_np = x_train.numpy()
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow((x_train_np[i] + 1) / 2)  # Rescale to [0, 1] for visualization
    plt.axis('off')
    plt.title("Car Sample")
plt.tight_layout()
plt.show()

# Constants
BUFFER_SIZE = x_train.shape[0]  # Use actual dataset size
BATCH_SIZE = 64  # Smaller batch size for better training with complex features
NOISE_DIM = 100

# Create downsampled datasets for each resolution
x_train_32 = x_train  # Original 32x32 images
x_train_16 = tf.image.resize(x_train, [16, 16])
x_train_8 = tf.image.resize(x_train, [8, 8])
x_train_4 = tf.image.resize(x_train, [4, 4])

train4 = tf.data.Dataset.from_tensor_slices(x_train_4).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train8 = tf.data.Dataset.from_tensor_slices(x_train_8).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train16 = tf.data.Dataset.from_tensor_slices(x_train_16).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train32 = tf.data.Dataset.from_tensor_slices(x_train_32).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Create directories to save results
for res in ['4x4', '8x8', '16x16', '32x32']:
    os.makedirs(res, exist_ok=True)

# STAGE 1: 4x4 Resolution
print("\n===== TRAINING 4x4 RESOLUTION =====")

# Generator for 4x4 resolution
def build_generator_4x4():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4 * 4 * 256, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Reshape((4, 4, 256)))
    model.add(layers.Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(3, (1, 1), padding='same', activation='tanh'))
    return model

# Discriminator for 4x4 resolution
def build_discriminator_4x4():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=[4, 4, 3]))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# Initialize generator and discriminator for 4x4 resolution
generator_4x4 = build_generator_4x4()
discriminator_4x4 = build_discriminator_4x4()

# Loss functions
def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Train the 4x4 model
EPOCHS_4x4 = 300  # You can adjust this
gen_losses = []
disc_losses = []

def train_4x4():
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(images):
        noise = tf.random.normal([images.shape[0], NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator_4x4(noise, training=True)
            real_output = discriminator_4x4(images, training=True)
            fake_output = discriminator_4x4(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator_4x4.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_4x4.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_4x4.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_4x4.trainable_variables))

        return gen_loss, disc_loss

    for epoch in range(EPOCHS_4x4):
        epoch_gen_loss, epoch_disc_loss = 0, 0
        num_batches = 0

        for image_batch in train4:
            g_loss, d_loss = train_step(image_batch)
            epoch_gen_loss += g_loss
            epoch_disc_loss += d_loss
            num_batches += 1

        epoch_gen_loss /= num_batches
        epoch_disc_loss /= num_batches
        gen_losses.append(epoch_gen_loss)
        disc_losses.append(epoch_disc_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS_4x4}, Gen Loss: {epoch_gen_loss:.4f}, Disc Loss: {epoch_disc_loss:.4f}")

            # Generate and save images
            noise = tf.random.normal([16, NOISE_DIM])
            generated_images = generator_4x4(noise, training=False)
            generated_images = (generated_images + 1) / 2  # Rescale to [0,1]

            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            for i, ax in enumerate(axes.flat):
                ax.imshow(generated_images[i])
                ax.axis('off')
            plt.savefig(f"4x4/generated_images_epoch_{epoch+1}.png")
            plt.show()
            plt.close()

    # Save trained models
    generator_4x4.save("4x4/generator_4x4_trained.h5")
    discriminator_4x4.save("4x4/discriminator_4x4_trained.h5")

    # Plot and save loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label="Generator Loss")
    plt.plot(disc_losses, label="Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("4x4 Resolution Training Loss")
    plt.savefig("4x4/loss_plot.png")
    plt.close()

# Train the 4x4 GAN
train_4x4()

# STAGE 2: 8x8 Resolution
print("\n===== TRAINING 8x8 RESOLUTION =====")

# Generator for 8x8 resolution
def build_generator_8x8():
    # Load the trained 4x4 generator
    base_model = tf.keras.models.load_model("4x4/generator_4x4_trained.h5")

    # Get all layers except the last (to-RGB) layer
    layers_to_keep = base_model.layers[:-1]

    # Create a new model with these layers
    input_layer = tf.keras.layers.Input(shape=(NOISE_DIM,))
    x = input_layer
    for layer in layers_to_keep:
        x = layer(x)

    # Add upsampling to 8x8
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Add final to-RGB layer
    outputs = layers.Conv2D(3, (1, 1), padding='same', activation='tanh')(x)

    return tf.keras.Model(inputs=input_layer, outputs=outputs)

# Discriminator for 8x8 resolution
def build_discriminator_8x8():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=[8, 8, 3]))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(128, (3, 3), padding='same', strides=(2, 2)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# Initialize generator and discriminator for 8x8 resolution
generator_8x8 = build_generator_8x8()
discriminator_8x8 = build_discriminator_8x8()

# Train the 8x8 model
EPOCHS_8x8 = 300
gen_losses_8x8 = []
disc_losses_8x8 = []

def train_8x8():
    generator_optimizer = tf.keras.optimizers.Adam(8e-5)  # Slightly reduced learning rate
    discriminator_optimizer = tf.keras.optimizers.Adam(8e-5)

    @tf.function
    def train_step(images):
        noise = tf.random.normal([images.shape[0], NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator_8x8(noise, training=True)
            real_output = discriminator_8x8(images, training=True)
            fake_output = discriminator_8x8(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator_8x8.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_8x8.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_8x8.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_8x8.trainable_variables))

        return gen_loss, disc_loss

    for epoch in range(EPOCHS_8x8):
        epoch_gen_loss, epoch_disc_loss = 0, 0
        num_batches = 0

        for image_batch in train8:
            g_loss, d_loss = train_step(image_batch)
            epoch_gen_loss += g_loss
            epoch_disc_loss += d_loss
            num_batches += 1

        epoch_gen_loss /= num_batches
        epoch_disc_loss /= num_batches
        gen_losses_8x8.append(epoch_gen_loss)
        disc_losses_8x8.append(epoch_disc_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS_8x8}, Gen Loss: {epoch_gen_loss:.4f}, Disc Loss: {epoch_disc_loss:.4f}")

            # Generate and save images
            noise = tf.random.normal([16, NOISE_DIM])
            generated_images = generator_8x8(noise, training=False)
            generated_images = (generated_images + 1) / 2  # Rescale to [0,1]

            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            for i, ax in enumerate(axes.flat):
                ax.imshow(generated_images[i])
                ax.axis('off')
            plt.savefig(f"8x8/generated_images_epoch_{epoch+1}.png")
            plt.show()
            plt.close()

    # Save trained models
    generator_8x8.save("8x8/generator_8x8_trained.h5")
    discriminator_8x8.save("8x8/discriminator_8x8_trained.h5")

    # Plot and save loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses_8x8, label="Generator Loss")
    plt.plot(disc_losses_8x8, label="Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("8x8 Resolution Training Loss")
    plt.savefig("8x8/loss_plot.png")
    plt.close()

# Train the 8x8 GAN
train_8x8()

# STAGE 3: 16x16 Resolution
print("\n===== TRAINING 16x16 RESOLUTION =====")

# Generator for 16x16 resolution
def build_generator_16x16():
    # Load the trained 8x8 generator
    base_model = tf.keras.models.load_model("8x8/generator_8x8_trained.h5")

    # Create a new model
    model = tf.keras.Sequential()

    # First add all layers except the last RGB layer from base model
    for layer in base_model.layers[:-1]:
        model.add(layer)

    # Add upsampling to 16x16
    model.add(layers.UpSampling2D(size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    # Add final to-RGB layer
    model.add(layers.Conv2D(3, (1, 1), padding='same', activation='tanh'))

    return model

# Discriminator for 16x16 resolution
def build_discriminator_16x16():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=[16, 16, 3]))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(64, (3, 3), padding='same', strides=(2, 2)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(128, (3, 3), padding='same', strides=(2, 2)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# Initialize generator and discriminator for 16x16 resolution
generator_16x16 = build_generator_16x16()
discriminator_16x16 = build_discriminator_16x16()

# Train the 16x16 model
EPOCHS_16x16 = 300
gen_losses_16x16 = []
disc_losses_16x16 = []

def train_16x16():
    generator_optimizer = tf.keras.optimizers.Adam(5e-5)  # Further reduced learning rate
    discriminator_optimizer = tf.keras.optimizers.Adam(5e-5)

    @tf.function
    def train_step(images):
        noise = tf.random.normal([images.shape[0], NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator_16x16(noise, training=True)
            real_output = discriminator_16x16(images, training=True)
            fake_output = discriminator_16x16(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator_16x16.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_16x16.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_16x16.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_16x16.trainable_variables))

        return gen_loss, disc_loss

    for epoch in range(EPOCHS_16x16):
        epoch_gen_loss, epoch_disc_loss = 0, 0
        num_batches = 0

        for image_batch in train16:
            g_loss, d_loss = train_step(image_batch)
            epoch_gen_loss += g_loss
            epoch_disc_loss += d_loss
            num_batches += 1

        epoch_gen_loss /= num_batches
        epoch_disc_loss /= num_batches
        gen_losses_16x16.append(epoch_gen_loss)
        disc_losses_16x16.append(epoch_disc_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS_16x16}, Gen Loss: {epoch_gen_loss:.4f}, Disc Loss: {epoch_disc_loss:.4f}")

            # Generate and save images
            noise = tf.random.normal([16, NOISE_DIM])
            generated_images = generator_16x16(noise, training=False)
            generated_images = (generated_images + 1) / 2  # Rescale to [0,1]

            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            for i, ax in enumerate(axes.flat):
                ax.imshow(generated_images[i])
                ax.axis('off')
            plt.savefig(f"16x16/generated_images_epoch_{epoch+1}.png")
            plt.show()
            plt.close()

    # Save trained models
    generator_16x16.save("16x16/generator_16x16_trained.h5")
    discriminator_16x16.save("16x16/discriminator_16x16_trained.h5")

    # Plot and save loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses_16x16, label="Generator Loss")
    plt.plot(disc_losses_16x16, label="Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("16x16 Resolution Training Loss")
    plt.savefig("16x16/loss_plot.png")
    plt.close()

# Train the 16x16 GAN
train_16x16()

# STAGE 4: 32x32 Resolution (Final)
print("\n===== TRAINING 32x32 RESOLUTION =====")

# Generator for 32x32 resolution
def build_generator_32x32():
    # Load the trained 16x16 generator
    base_model = tf.keras.models.load_model("16x16/generator_16x16_trained.h5")

    # Get all layers except the last (to-RGB) layer
    layers_to_keep = base_model.layers[:-1]

    # Create a new model with these layers
    input_layer = tf.keras.layers.Input(shape=(NOISE_DIM,))
    x = input_layer
    for layer in layers_to_keep:
        x = layer(x)

    # Add upsampling to 32x32
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Add final to-RGB layer
    outputs = layers.Conv2D(3, (1, 1), padding='same', activation='tanh')(x)

    return tf.keras.Model(inputs=input_layer, outputs=outputs)

# Discriminator for 32x32 resolution
def build_discriminator_32x32():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(16, (3, 3), padding='same', input_shape=[32, 32, 3]))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(32, (3, 3), padding='same', strides=(2, 2)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(64, (3, 3), padding='same', strides=(2, 2)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(128, (3, 3), padding='same', strides=(2, 2)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# Initialize generator and discriminator for 32x32 resolution
generator_32x32 = build_generator_32x32()
discriminator_32x32 = build_discriminator_32x32()

# Train the 32x32 model
EPOCHS_32x32 = 300  # More epochs for final resolution
gen_losses_32x32 = []
disc_losses_32x32 = []

def train_32x32():
    generator_optimizer = tf.keras.optimizers.Adam(2e-5)  # Even lower learning rate for final resolution
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-5)

    @tf.function
    def train_step(images):
        noise = tf.random.normal([images.shape[0], NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator_32x32(noise, training=True)
            real_output = discriminator_32x32(images, training=True)
            fake_output = discriminator_32x32(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator_32x32.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_32x32.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_32x32.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_32x32.trainable_variables))

        return gen_loss, disc_loss

    for epoch in range(EPOCHS_32x32):
        epoch_gen_loss, epoch_disc_loss = 0, 0
        num_batches = 0

        for image_batch in train32:
            g_loss, d_loss = train_step(image_batch)
            epoch_gen_loss += g_loss
            epoch_disc_loss += d_loss
            num_batches += 1

        epoch_gen_loss /= num_batches
        epoch_disc_loss /= num_batches
        gen_losses_32x32.append(epoch_gen_loss)
        disc_losses_32x32.append(epoch_disc_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS_32x32}, Gen Loss: {epoch_gen_loss:.4f}, Disc Loss: {epoch_disc_loss:.4f}")

            # Generate and save images
            noise = tf.random.normal([16, NOISE_DIM])
            generated_images = generator_32x32(noise, training=False)
            generated_images = (generated_images + 1) / 2  # Rescale to [0,1]

            fig, axes = plt.subplots(4, 4, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                ax.imshow(generated_images[i])
                ax.axis('off')
            plt.savefig(f"32x32/generated_images_epoch_{epoch+1}.png")
            plt.show()
            plt.close()

    # Save trained models
    generator_32x32.save("32x32/generator_32x32_trained.h5")
    discriminator_32x32.save("32x32/discriminator_32x32_trained.h5")

    # Plot and save loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses_32x32, label="Generator Loss")
    plt.plot(disc_losses_32x32, label="Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("32x32 Resolution Training Loss")
    plt.savefig("32x32/loss_plot.png")
    plt.close()

# Train the 32x32 GAN
train_32x32()

# Generate final showcase of results
print("\n===== GENERATING FINAL RESULTS =====")

# Load the final generator
final_generator = tf.keras.models.load_model("32x32/generator_32x32_trained.h5")

# Generate a larger grid of sample images
noise = tf.random.normal([25, NOISE_DIM])
generated_images = final_generator(noise, training=False)
generated_images = (generated_images + 1) / 2  # Rescale to [0,1]

plt.figure(figsize=(15, 15))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(generated_images[i])
    plt.axis('off')
plt.tight_layout()
plt.savefig("final_results.png")
plt.show()

print("Progressive GAN training complete!")

"""Open Discussion

# 2.1 Explain the architecture of your Generator and Discriminator. What specific layers or structures did you include, and why did you choose them?

For my vanilla GAN, I designed a Generator that transforms a 100-dimensional noise vector into a 32×32 image through a series of upsampling operations. I included dense layers to process the initial noise, followed by Conv2DTranspose layers with BatchNormalization and LeakyReLU activations to generate increasingly detailed features. My Discriminator uses a series of convolutional layers with increasing filter counts (64→128), LeakyReLU activations, and dropout for regularization, culminating in a binary classification output. In my ProGAN implementation, I took a fundamentally different approach by creating resolution-specific architectures that grow progressively. I started with a simple Generator for 4×4 images, then built increasingly complex networks for 8×8, 16×16, and finally 32×32 resolutions, with each model incorporating the trained weights from the previous resolution. This progressive approach allowed me to establish stable low-resolution features before attempting to generate finer details, though I found both approaches struggled with the complex features of car images at the limited 32×32 resolution.

# 2.2 How does the architecture of your vanilla GAN differ from the light version of ProGAN? What changes did you make to implement the ProGAN structure?

My vanilla GAN trained directly on full-resolution 32×32 images with a fixed architecture, whereas in my ProGAN implementation, I progressively increased resolution from 4×4 to 32×32 while growing model complexity. The key architectural differences I implemented include: sequential training across four distinct resolution stages, each with its own Generator and Discriminator; transfer of learned weights between resolution stages; resolution-specific hyperparameters with decreasing learning rates for higher resolutions; and increasing network depth and parameter count at each stage. In my ProGAN, I carefully designed the model transition between resolutions, making sure previous layers remained trainable while adding new capacity for higher-resolution details. I deliberately chose not to implement some of the original paper's more complex components like minibatch standard deviation or equalized learning rates to create a more computationally efficient version that still captures the core progressive growth concept, though I found this simplified approach still struggled with generating clear car images.

# 2.3 When working on the ProGAN, what specific issues did you face that differed from training the vanilla GAN? Did the training process improve or worsen with ProGAN?

I encountered several challenges unique to my ProGAN implementation that I didn't face with vanilla GAN. The most significant was correctly handling model architecture transitions between resolutions, which caused "too many positional arguments" errors when I tried to use the functional API with loaded models. I also struggled with training stability across resolution changes and needed to carefully tune learning rates for each stage. Despite these difficulties, I observed only modest improvements with ProGAN for the car class. In my vanilla GAN, after 300 epochs, the generated images showed vague car-like shapes with poor definition. Similarly, in my ProGAN implementation, I could hardly discern clear car structures even after training through multiple resolution stages. I observed occasional boxy shapes and sometimes what might be interpreted as wheels, but the results were far from realistic car representations. The progressive approach showed theoretical promise but would likely require significantly more training epochs at each resolution stage and a more sophisticated implementation to generate recognizable vehicles at this limited resolution.

# 2.4 How did hyperparameters (e.g., learning rate, batch size) influence the performance of your GAN? Did you adjust them during training to improve results?

I found hyperparameter selection to be crucial for successfully training both my GAN implementations, though even with extensive tuning, both models struggled with the car class. For learning rates, I observed that using 1e-4 for my vanilla GAN caused training instability, while my ProGAN benefited from a decreasing schedule (1e-4 → 8e-5 → 5e-5 → 2e-5) across resolution stages, which provided some stability but insufficient quality improvement. I reduced my batch size from 128 to 64, which introduced more variability but didn't substantially improve the clarity of car features. With both implementations, I noticed that although losses would stabilize, the visual quality of generated images remained poor, with cars barely recognizable in either approach. I experimented with different epoch counts, finding that even 300 epochs for vanilla GAN and 150 epochs per resolution stage for ProGAN were insufficient for generating clear car images. I believe the limited resolution (32×32) combined with the complex structure of cars presents a fundamental challenge that requires more sophisticated approaches or significantly more training time than I was able to implement.

# 2.5 What improvements would you make to your GAN or ProGAN implementation? Would you add any additional techniques or layers to improve the image generation quality?

Based on my experiments with both implementations on the car class, I would make several significant improvements to enhance image quality. First, I'd dramatically increase the number of epochs at each resolution stage, as my current results suggest both models need much more training time to develop recognizable car features. I'd implement Wasserstein loss with gradient penalty to better handle the geometric structures of vehicles, and add self-attention mechanisms to help capture the spatial relationships between car components. For my ProGAN specifically, I'd implement proper fade-in transitions between resolutions and add the minibatch standard deviation layer from the original paper to encourage diversity. I'd also consider working with higher final resolutions (64×64 or 128×128) as 32×32 may be too limited for capturing the distinctive details of cars. Additionally, I'd experiment with conditional GAN approaches to provide more guidance to the generator. While I could hardly see clear cars in my current ProGAN implementation, I believe these enhancements, particularly the increased training duration and higher resolution, would substantially improve the results for this challenging class of images.
"""
