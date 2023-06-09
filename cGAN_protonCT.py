import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import datetime

from keras.optimizers import Adam
from keras.layers import Input, Dropout, Concatenate
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D, Conv2D
from keras.models import Model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.python.client import device_lib
from data_loader import DataLoader

device_lib.list_local_devices()
tf.test.is_built_with_cuda()
tf.config.list_physical_devices()
tf.sysconfig.get_build_info()

autotune = tf.data.AUTOTUNE


class CycleGAN:
    def __init__(self):
        # Input shape
        self.img_rows = 128  # 이미지 가로 픽셀
        self.img_cols = 128  # 이미지 세로 픽셀
        self.channels = 1  # 이미지 채널 (흑백이면 1, 컬러면 3)
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN) discriminator가 patch단위로 판단하는데, 그 patch 크기
        patch = int(self.img_rows / 2 ** 4)  # 2**4 hyperparameter
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32  # hyperparameter
        self.df = 64  # hyperparameter

        # Loss weights, 둘다 hyperparameter
        self.lambda_cycle = 10.0  # Cycle-consistency loss (F(G(X))=x인지) 의 weight
        self.lambda_id = 0.1 * self.lambda_cycle  # Identity loss (Detail을 살리기 위한 loss, F(G(X))와 F(Y)의 차이)

        optimizer = Adam(learning_rate=0.0002, beta_1=0.5)  # 둘다 hyperparameter, 특히 beta_1이 너무 낮아서 특이함

        # Build and compile the discriminators
        self.d_pCT = self.build_discriminator()  # MC pCT와 synthetic pCT 구분하는 discriminator
        self.d_kVCT = self.build_discriminator()  # Real kVCT와 cyclic kVCT 구분하는 discriminator
        self.d_pCT.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.d_kVCT.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_pCT = self.build_generator()  # kVCT를 pCT로 변환하는 generator
        self.g_kVCT = self.build_generator()  # pCT를 kVCT로 변환하는 generator

        # Input images from both domains
        img_kVCT = Input(shape=self.img_shape)
        img_pCT = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_pCT = self.g_pCT(img_kVCT)
        fake_kVCT = self.g_kVCT(img_pCT)
        # Translate images back to original domain
        reconstr_kVCT = self.g_kVCT(fake_pCT)
        reconstr_pCT = self.g_pCT(fake_kVCT)
        # Identity mapping of images
        img_kVCT_id = self.g_kVCT(img_kVCT)  # g_kVCT는 kVCT를 얼마나 안 건드리나?
        img_pCT_id = self.g_pCT(img_pCT)  # g_pCT는 pCT를 얼마나 안 건드리나?

        # For the combined model we will only train the generators
        self.d_pCT.trainable = False
        self.d_kVCT.trainable = False

        # Discriminators determines validity of translated images
        valid_kVCT = self.d_kVCT(fake_kVCT)
        valid_pCT = self.d_pCT(fake_pCT)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_kVCT, img_pCT],
                              outputs=[valid_kVCT, valid_pCT, reconstr_kVCT, reconstr_pCT, img_kVCT_id, img_pCT_id])
        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                              loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf * 4)
        u2 = deconv2d(u1, d2, self.gf * 2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()
        # Adversarial loss ground truths
        valid = np.ones(shape=((batch_size,) + self.disc_patch))
        fake = np.zeros(shape=((batch_size,) + self.disc_patch))

        for epoch in range(epochs):
            for batch_i, (imgs_kVCT, imgs_pCT) in enumerate(self.data_loader.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_pCT = self.g_pCT.predict(imgs_kVCT)
                fake_kVCT = self.g_kVCT.predict(imgs_pCT)

                # Train the discriminators (original images = real / translated = Fake)
                dkVCT_loss_real = self.d_kVCT.train_on_batch(x=imgs_kVCT, y=valid)  # 실제 kvct는 valid (1) 이도록
                dkVCT_loss_fake = self.d_kVCT.train_on_batch(x=fake_kVCT, y=fake)  # 가짜 kvct는 fake (0) 이도록
                dkVCT_loss = 0.5 * np.add(dkVCT_loss_real, dkVCT_loss_fake)

                dpCT_loss_real = self.d_pCT.train_on_batch(x=imgs_pCT, y=valid)
                dpCT_loss_fake = self.d_pCT.train_on_batch(x=fake_pCT, y=fake)
                dpCT_loss = 0.5 * np.add(dpCT_loss_real, dpCT_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dkVCT_loss, dpCT_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch(x=[imgs_kVCT, imgs_pCT],
                                                      y=[valid, valid, imgs_kVCT, imgs_pCT, imgs_kVCT, imgs_pCT])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, "
                    "id: %05f] time: %s " \
                    % (epoch, epochs,
                       batch_i, self.data_loader.n_batches,
                       d_loss[0], 100 * d_loss[1],
                       g_loss[0],
                       np.mean(g_loss[1:3]),
                       np.mean(g_loss[3:5]),
                       np.mean(g_loss[5:6]),
                       elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        os.makedirs('images', exist_ok=True)
        r, c = 2, 3

        imgs_kVCT = self.data_loader.load_data(domain="kVCT", batch_size=1, is_testing=True)
        imgs_pCT = self.data_loader.load_data(domain="pCT", batch_size=1, is_testing=True)

        # Translate images to the other domain
        fake_pCT = self.g_pCT.predict(imgs_kVCT)
        fake_kVCT = self.g_kVCT.predict(imgs_pCT)
        # Translate back to original domain
        reconstr_kVCT = self.g_kVCT.predict(fake_pCT)
        reconstr_pCT = self.g_pCT.predict(fake_kVCT)

        gen_imgs = np.concatenate([imgs_kVCT, fake_pCT, reconstr_kVCT, imgs_pCT, fake_kVCT, reconstr_pCT])

        # Rescale images 0 - 1 (tanh 출력은 -1 - 1이므로 변환)
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d_%d.png" % (epoch, batch_i))
        plt.close()


if __name__ == '__main__':

    gan = CycleGAN()
    gan.train(epochs=200, batch_size=1, sample_interval=200)

