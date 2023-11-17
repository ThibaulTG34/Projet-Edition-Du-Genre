# This Python file uses the following encoding: utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import time
import argparse
import glob
import scipy

import cv2
from cv2 import imread, resize
import dlib

import tensorflow as tf
from keras import Input, Model
from keras.optimizers.legacy import Adam
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, ZeroPadding2D, BatchNormalization, Add
from keras_contrib.layers import InstanceNormalization

class CNN:
    def __init__(self):
        super().__init__()
        self.start_epochs = 0
        self.nepochs = 120
        self.decay_epochs = 10
        self.learning_decay = float(0.002)
        self.size = 256
        self.batch_size = 5
        self.inchannel = 3
        self.outchannel = 3
        self.cpu = 8
        self.gpu = False

        self.mode = 0
        self.source = None
        self.result = None

        self.fps = 15 # + vite - vite -> lecture [temps video]
        self.num_frames = 100 # + longue - longue -> interpolation [smoothing video]
        self.in_animation = str("test")

    def set_directory(self, val):
        v = int(max(min(int(val),1),0))
        self.mode = v

    def set_source(self, s):
        self.source = cv2.imread(str(s))

    def set_frames(self, f):
        self.num_frames = int(f)

    def set_fps(self, f):
        self.fps = int(f)

    def get_result(self):
        return self.result

    def set_animation(self,name):
        self.in_animation = str(name)

    def get_animation(self):
        #file__name, file_extension = os.path.splitext(os.path.basename(str(self.body_name)))
        #self.in_animation = str(file__name + "_anim")
        self.animation()

    def get_gif(self):
        #file__name, file_extension = os.path.splitext(os.path.basename(str(self.body_name)))
        #self.in_animation = str(file__name + "_anim")
        self.create_animation()

    def animation(self):
        if self.source.shape != self.result.shape:
            raise ValueError("Les deux images doivent avoir la même taille.")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.in_animation + ".mp4", fourcc, self.fps, (self.source.shape[1], self.source.shape[0]))

        for i in range(self.num_frames):
            alpha = i / (self.num_frames - 1)
            interpolated_image = cv2.addWeighted(self.source, 1 - alpha, self.result, alpha, 0)
            out.write(interpolated_image)

        print("Animation created in : " + self.in_animation + ".mp4")
        self.out_animation = self.in_animation + ".mp4"
        out.release()

    def create_animation(self):
        if self.source.shape != self.result.shape:
            raise ValueError("Les deux images doivent avoir la même taille.")

        frames = []

        for i in range(self.num_frames):
            alpha = i / (self.num_frames - 1)
            interpolated_image = cv2.addWeighted(self.source, 1 - alpha, self.result, alpha, 0)
            interpolated_image = cv2.cvtColor(interpolated_image, cv2.COLOR_BGR2RGB)
            frames.append(interpolated_image)

        self.out_animation = os.path.splitext(self.in_animation)[0] + ".gif"
        imageio.mimsave(self.out_animation, frames, fps=self.fps)

        print("Animation created in : " + self.out_animation)

    def print_parameters(self):
        s = ("mtf" if self.mode == 0 else "ftm")
        print("Datasets in : " + str(s))
        print(f"Start Epochs : {self.start_epochs}")
        print(f"Number of Epochs : {self.nepochs}")
        print(f"Decay Epochs : {self.decay_epochs}")
        print(f"Learning Rate Decay : {self.learning_decay}")
        print(f"Size : {self.size}")
        print(f"Batch Size : {self.batch_size}")
        print(f"Input Channels : {self.inchannel}")
        print(f"Output Channels : {self.outchannel}")
        print(f"Number of CPU Threads : {self.cpu}")
        print(f"Use GPU : {self.gpu}")

    def set(self, option, val):
        if option == 0:
            self.start_epochs = int(val)
        elif option == 1:
            self.nepochs = int(val)
        elif option == 2:
            self.decay_epochs = int(val)
        elif option == 3:
            self.learning_decay = float(val)
        elif option == 4:
            self.size = int(val)
        elif option == 5:
            self.batch_size = int(val)
        elif option == 6:
            v = int(min(max(int(val),0),3))
            self.inchannel = int(v)
        elif option == 7:
            v = int(min(max(int(val),0),3))
            self.outchannel = int(v)
        elif option == 8:
            self.cpu = int(val)
        elif option == 9:
            self.gpu = bool(val)

def load_train_images(data_dir):
    images_type_A = glob.glob(data_dir + '/trainA/*.jpg')
    images_type_B = glob.glob(data_dir + '/trainB/*.jpg')

    processed_imagesA = []
    processed_imagesB = []

    for i, filename in enumerate(images_type_A):
        imA = imread(filename)
        imB = imread(images_type_B[i])

        imA = resize(imA, (128, 128))
        imB = resize(imA, (128, 128))

        #Randomly flip some images
        if np.random.random() > 0.5:
            imA = np.fliplr(imA)
            imB = np.fliplr(imB)

        processed_imagesA.append(imA)
        processed_imagesB.append(imB)

    #Normalise image values between -1 and 1
    processed_imagesA = np.array(processed_imagesA)/127.5 - 1.0
    processed_imagesB = np.array(processed_imagesB)/127.5 - 1.0

    return processed_imagesA, processed_imagesB

def load_test_images(data_dir, num_images):
    images_type_A = glob.glob(data_dir + '/testA/*.jpg')
    images_type_B = glob.glob(data_dir + '/testB/*.jpg')

    images_type_A = np.random.choice(images_type_A, num_images)
    images_type_B = np.random.choice(images_type_B, num_images)

    processed_imagesA = []
    processed_imagesB = []

    for i in range(len(images_type_A)):
        imA = resize(imread(images_type_A[i]).astype(np.float32), (128, 128))
        imB = resize(imread(images_type_B[i]).astype(np.float32), (128, 128))

        processed_imagesA.append(imA)
        processed_imagesB.append(imB)

    #Normalise image values between -1 and 1
    processed_imagesA = np.array(processed_imagesA)/127.5 - 1.0
    processed_imagesB = np.array(processed_imagesB)/127.5 - 1.0

    return processed_imagesA, processed_imagesB

#Save the training losses to the tensorboard logs that can be used for visualization
def save_losses_tensorboard(callback, name, loss, batch_no):
    summary_writer = tf.summary.create_file_writer("./data/male_female")

    with summary_writer.as_default():
        tf.summary.scalar(name, loss, batch_no)

    summary_writer.flush()

def save_test_results(realA, realB, fakeA, fakeB, reconsA, reconsB, identityA, identityB):
    for i in range(len(realA)):

        realA[i] = cv2.cvtColor(realA[i], cv2.COLOR_BGR2RGB)
        fakeA[i] = cv2.cvtColor(fakeA[i], cv2.COLOR_BGR2RGB)
        #reconsA[i] = cv2.cvtColor(reconsA[i], cv2.COLOR_BGR2RGB)
        identityA[i] = cv2.cvtColor(identityA[i], cv2.COLOR_BGR2RGB)
        realB[i] = cv2.cvtColor(realB[i], cv2.COLOR_BGR2RGB)
        fakeB[i] = cv2.cvtColor(fakeB[i], cv2.COLOR_BGR2RGB)
        #reconsB[i] = cv2.cvtColor(reconsB[i], cv2.COLOR_BGR2RGB)
        identityB[i] = cv2.cvtColor(identityB[i], cv2.COLOR_BGR2RGB)

        fig = plt.figure()        
        plt.imshow(realA[i])
        plt.axis('off')
        plt.savefig("results/m2f/real_{}".format(i), bbox_inches='tight')
        plt.close(fig)
        fig2 = plt.figure()
        plt.imshow(fakeB[i])
        plt.axis('off')
        plt.savefig("results/m2f/fake_{}".format(i), bbox_inches='tight')
        plt.close(fig2)
        fig3 = plt.figure()
        plt.imshow(reconsA[i])
        plt.axis('off')
        plt.savefig("results/m2f/recons_{}".format(i, bbox_inches='tight'))
        plt.close(fig3)
        fig4 = plt.figure()
        plt.imshow(identityA[i])
        plt.axis('off')
        plt.savefig("results/m2f/identity_{}".format(i), bbox_inches='tight')
        plt.close(fig4)

        fig = plt.figure()            
        plt.imshow(realB[i])
        plt.axis('off')
        plt.savefig("results/f2m/real_{}".format(i), bbox_inches='tight')
        plt.close(fig)
        fig2 = plt.figure()
        plt.imshow(fakeA[i])
        plt.axis('off')
        plt.savefig("results/f2m/fake_{}".format(i), bbox_inches='tight')
        plt.close(fig2)
        fig3 = plt.figure()
        plt.imshow(reconsB[i])
        plt.axis('off')
        plt.savefig("results/f2m/recons_{}".format(i), bbox_inches='tight')
        plt.close(fig3)
        fig4 = plt.figure()
        plt.imshow(identityB[i])
        plt.axis('off')
        plt.savefig("results/f2m/identity_{}".format(i), bbox_inches='tight')
        plt.close(fig4)

def resnet_block(x):
    x2 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x2 = InstanceNormalization(axis=1)(x2)
    x2 = Activation('relu')(x2)

    x2 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x2)
    x2 = InstanceNormalization(axis=1)(x2)

    return Add()([x2, x])

def define_generator_network(num_resnet_blocks=9):
    input_size = (128,128,3)
    input_layer = Input(shape=input_size)

    x = Conv2D(filters=64, kernel_size=7, strides=1, padding="same")(input_layer)
    x = InstanceNormalization(axis=1)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(x)
    x = InstanceNormalization(axis=1)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=256, kernel_size=3, strides=2, padding="same")(x)
    x = InstanceNormalization(axis=1)(x)
    x = Activation("relu")(x)

    for i in range(num_resnet_blocks):
        x = resnet_block(x)

    #Upsampling to recover the transformed image
    #Conv2DTranspose with a stride 2 works like Conv2D with stride 1/2
    x = Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization(axis=1)(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization(axis=1)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=3, kernel_size=7, strides=1, padding="same")(x)
    output = Activation('tanh')(x) #tanh activation to get normalised output image

    model = Model(inputs=[input_layer], outputs=[output])
    return model

#Define the discriminator network based on the PatchGAN's architecture
def define_discriminator_network():
    input_size = (128, 128, 3)
    num_hidden_layers = 3
    input_layer = Input(shape=input_size)

    x = ZeroPadding2D(padding=(1, 1))(input_layer)

    x = Conv2D(filters=64, kernel_size=4, strides=2, padding="valid")(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = ZeroPadding2D(padding=(1, 1))(x)

    for i in range(1, num_hidden_layers + 1):
        x = Conv2D(filters=2 ** i * 64, kernel_size=4, strides=2, padding="valid")(x)
        x = InstanceNormalization(axis=1)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = ZeroPadding2D(padding=(1, 1))(x)

    #Sigmoid activation to normalise output values between 0 and 1 which will be used to train real or fake labels
    output = Conv2D(filters=1, kernel_size=4, strides=1, activation="sigmoid")(x) #This is the patch output

    model = Model(inputs=[input_layer], outputs=[output])
    return model

def define_adversarial_model(generatorA2B, generatorB2A, discriminatorA, discriminatorB, train_optimizer, lambda_cyc = 10, lambda_idt = 5):

    inA = Input(shape=(128, 128, 3))
    inB = Input(shape=(128, 128, 3))

    fakeB = generatorA2B(inA)
    fakeA = generatorB2A(inB)

    reconstructedA = generatorB2A(fakeB)
    reconstructedB = generatorA2B(fakeA)

    identityA = generatorB2A(inA)
    identityB = generatorA2B(inB)

    decisionA = discriminatorA(fakeA)
    decisionB = discriminatorB(fakeB)

    adversarial_model = Model(inputs = [inA, inB], outputs = [decisionA, decisionB, reconstructedA, reconstructedB, identityA, identityB])
    adversarial_model.compile(loss= ['mse', 'mse', 'mae', 'mae', 'mae', 'mae'], loss_weights= [1, 1, lambda_cyc, lambda_cyc, lambda_idt, lambda_idt],
                                optimizer = train_optimizer)
    print(adversarial_model.summary())
    return adversarial_model

def train():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batchSize', type=int, default=1, help='Batch Size to be used for training')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs that training should run')
    parser.add_argument('--lambda_cyc', type=int, default=10, help='lambda value for cycle consistency loss')
    parser.add_argument('--lambda_idt', type=int, default=5, help='lambda value for identity loss')
    parser.add_argument('--save_epoch_freq', type=int, default=5, help='The frequency at which model should be saved and evaluated')
    parser.add_argument('--num_resnet_blocks', type=int, default=9, help='Number of ResNet blocks for transformation in generator')
    parser.add_argument('--data_dir', type=str, default='data/male_female/', help='Directory where train and test images are present')

    opt, _ = parser.parse_known_args()

    data_dir = opt.data_dir #"data/male_female/"
    batch_size = opt.batchSize#2
    epochs = opt.epochs#40
    lambda_cyc = opt.lambda_cyc#10
    lambda_idt = opt.lambda_idt#5
    save_epoch_freq = opt.save_epoch_freq#5
    num_resnet_blocks = opt.num_resnet_blocks#9

    trainA, trainB = load_train_images(data_dir)
    train_optimizer = Adam(0.0002, 0.5)

    #Define the two discriminator models
    discA = define_discriminator_network()
    discB = define_discriminator_network()

    print(discA.summary())

    #The discriminators are trained on MSE loss on the patch output
    #Compile the model for dicriminators
    discA.compile(loss='mse', optimizer=train_optimizer, metrics= ['accuracy'])
    discB.compile(loss='mse', optimizer=train_optimizer, metrics= ['accuracy'])

    real_labels = np.ones((batch_size, 7, 7, 1))
    fake_labels = np.zeros((batch_size, 7, 7, 1))

    #Define the two generator models
    genA2B = define_generator_network(num_resnet_blocks=num_resnet_blocks)
    genB2A = define_generator_network(num_resnet_blocks=num_resnet_blocks)

    print(genA2B.summary())

    #make the dicriminators non-trainable in the adversarial model
    discA.trainable = False
    discB.trainable = False

    #Define the adversarial model
    gan_model = define_adversarial_model(genA2B, genB2A, discA, discB, train_optimizer, lambda_cyc=lambda_cyc, lambda_idt=lambda_idt)

    #Setup the tensorboard to store and visualise losses
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), write_images=True, write_grads=True,
                                  write_graph=True)
    tensorboard.set_model(genA2B)
    tensorboard.set_model(genB2A)
    tensorboard.set_model(discA)
    tensorboard.set_model(discB)
    print("Batch Size: {}".format(batch_size))
    print("Num of ResNet Blocks: {}".format(num_resnet_blocks))
    print("Starting training for {0} epochs with lambda_cyc = {1}, lambda_idt = {2}, num_resnet_blocks = {3}".format(epochs, lambda_cyc, lambda_idt, num_resnet_blocks))
    #Start training
    for epoch in range(epochs):
        print("Epoch:{}".format(epoch))
        start_time = time.time()

        dis_losses = []
        gen_losses = []

        num_batches = int(min(trainA.shape[0], trainB.shape[0]) / batch_size)
        print("Number of batches:{} in each epoch".format(num_batches))

        #for index in range(num_batches):
        for index in range(50):
            print("Batch:{}".format(index))

            # Sample images
            realA = trainA[index * batch_size:(index + 1) * batch_size]
            realB = trainB[index * batch_size:(index + 1) * batch_size]

            # Translate images to opposite domain
            fakeB = genA2B.predict(realA)
            fakeA = genB2A.predict(realB)

            # Train the discriminator A on real and fake images
            dLossA_real = discA.train_on_batch(realA, real_labels)
            dLossA_fake = discA.train_on_batch(fakeA, fake_labels)

            # Train the discriminator B on ral and fake images
            dLossB_real = discB.train_on_batch(realB, real_labels)
            dLossB_fake = discB.train_on_batch(fakeB, fake_labels)

            # Calculate the total discriminator loss
            mean_disc_loss = 0.5 * np.add(0.5 * np.add(dLossA_real, dLossA_fake), 0.5 * np.add(dLossB_real, dLossB_fake))

            print("Total Discriminator Loss:{}".format(mean_disc_loss))

            """
            Train the generator networks
            """
            g_loss = gan_model.train_on_batch([realA, realB],
                                                        [real_labels, real_labels, realA, realB, realA, realB])

            print("Adversarial Model losses:{}".format(g_loss))

            dis_losses.append(mean_disc_loss)
            gen_losses.append(g_loss)

        #Save losses to tensorboard for that epoch
        #Adding a smoothed out loss (by taking mean) to the tensorboard
        save_losses_tensorboard(tensorboard, 'discriminatorA_loss', np.mean(0.5 * np.add(dLossA_real, dLossA_fake)), epoch)
        save_losses_tensorboard(tensorboard, 'discriminatorB_loss', np.mean(0.5 * np.add(dLossB_real, dLossB_fake)), epoch)
        save_losses_tensorboard(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
        save_losses_tensorboard(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)
        if epoch % save_epoch_freq == 0:
            # Load Test images for seeing the results of the network
            testA, testB = load_test_images(data_dir=data_dir, num_images=2)

            # Generate images
            fakeB = genA2B.predict(testA)
            fakeA = genB2A.predict(testB)

            # Get reconstructed images
            reconsA = genB2A.predict(fakeB)
            reconsB = genA2B.predict(fakeA)

            identityA = genB2A.predict(testA)
            identityB = genA2B.predict(testB)

            genA2B.save('generatorAToB_temp_%d.h5'%epoch)
            genB2A.save('generatorBToA_temp_%d.h5'%epoch)
            discA.save('discriminatorA_temp_%d.h5'%epoch)
            discB.save('discriminatorB_temp_%d.h5'%epoch)
            save_test_results(testA, testB, fakeA, fakeB, reconsA, reconsA, identityA, identityB)

        print("--- %s seconds --- for epoch" % (time.time() - start_time))

    print("Training completed. Saving weights.")
    genA2B.save('generatorAToB.h5')
    genB2A.save('generatorBToA.h5')
    discA.save('discriminatorA.h5')
    discB.save('discriminatorB.h5')

def cnn_test():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batchSize', type=int, default=1, help='Batch Size to be used for training')
    parser.add_argument('--data_dir', type=str, default='data/male_female/', help='Directory where train and test images are present')

    opt, _ = parser.parse_known_args()

    data_dir = opt.data_dir
    batch_size = opt.batchSize

    print("Data_dir:{}".format(data_dir))
    print("BatchSize:{}".format(batch_size))

    genA2B = define_generator_network()
    genB2A = define_generator_network()

    genA2B.load_weights("generatorAToB.h5")
    genB2A.load_weights("generatorBToA.h5")

    testA, testB = load_test_images(data_dir=data_dir, num_images=batch_size)

    # Generate images
    fakeB = genA2B.predict(testA)
    fakeA = genB2A.predict(testB)

    # Get reconstructed images
    reconsA = genB2A.predict(fakeB)
    reconsB = genA2B.predict(fakeA)

    identityA = genB2A.predict(testA)
    identityB = genA2B.predict(testB)

    save_test_results(testA, testB, fakeA, fakeB, reconsA, reconsA, identityA, identityB)

#train()
#cnn_test()
