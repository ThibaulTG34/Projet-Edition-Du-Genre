# This Python file uses the following encoding: utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import time
import argparse
import glob
import scipy
import itertools
import os
import shutil

import cv2
from cv2 import imread, resize
import dlib

from collections import defaultdict
from datetime import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models.generator import Generator
from models.discriminator import Discriminator
from utils.utils import ReplayBuffer
from utils.utils import LambdaLR
from utils.utils import weights_init_normal
from datasets import ImageDataset
from tensorboardX import SummaryWriter
#tensorboard --logdir ./logs

class CNN:
    def __init__(self):
        super().__init__()
        self.root = str("./data")
        self.start_epochs = 0
        self.nepochs = 1
        self.decay_epochs = 0
        self.learning_decay = float(0.002)
        self.size = 256
        self.batch_size = 1
        self.inchannel = 3
        self.outchannel = 3
        self.cpu = 12
        self.gpu = True

        self.keras_dir = str("./keras")
        self.tensor_dir = str("./tensor")
        self.last_save_eochs = str("./keras")

        self.mode = 0
        self.source = None
        self.source_name = None
        self.target_name = None
        self.result = None

        self.fps = 15 # + vite - vite -> lecture [temps video]
        self.num_frames = 100 # + longue - longue -> interpolation [smoothing video]
        self.in_animation = str("test")

    def to_dict(self):
        attributes = vars(self)
        result_dict = {key: str(value) for key, value in attributes.items()}
        print(result_dict)
        #return result_dict

    def set_root(self, val):
        self.root = str(val)

    def set_keras_dir(self, val):
        self.keras_dir = str(val)

    def set_tensor_dir(self, val):
        self.tensor_dir = str(val)

    def set_directory(self, val):
        v = int(max(min(int(val),1),0))
        self.mode = v

    def set_source(self, s):
        self.source_name = str(s)
        self.source = cv2.imread(str(s))

    def set_mode(self, m):
        self.mode = int(m)

    def set_frames(self, f):
        self.num_frames = int(f)

    def set_fps(self, f):
        self.fps = int(f)

    def get_result(self):
        if self.target_name is None : return
        self.result = cv2.imread(self.target_name)
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

    def train(self):
        #if not (__name__ == "__main__") : return
        self.dataroot = str(self.root)
        self.start_epochs = int(self.start_epochs)
        self.nepochs = int(self.nepochs)
        self.decay_epochs = int(self.decay_epochs)
        self.learning_decay = float(self.learning_decay)
        self.size = int(self.size)
        self.batch_size = int(self.batch_size)
        self.inchannel = int(self.inchannel)
        self.outchannel = int(self.outchannel)
        self.cpu = int(self.cpu)
        self.gpu = bool(self.gpu)

        if torch.cuda.is_available() and not (self.gpu is True):
            print("WARNING: You have a CUDA device, so you should probably run with GPU")

        # Networks
        netG_A2B = Generator(self.inchannel, self.outchannel)
        netG_B2A = Generator(self.outchannel, self.inchannel)
        netD_A = Discriminator(self.inchannel)
        netD_B = Discriminator(self.outchannel)

        if self.gpu is True:
            netG_A2B.cuda()
            netG_B2A.cuda()
            netD_A.cuda()
            netD_B.cuda()

        netG_A2B.apply(weights_init_normal)
        netG_B2A.apply(weights_init_normal)
        netD_A.apply(weights_init_normal)
        netD_B.apply(weights_init_normal)

        # Lossess
        criterion_GAN = torch.nn.MSELoss()
        criterion_cycle = torch.nn.L1Loss()
        criterion_identity = torch.nn.L1Loss()

        # Optimizers & LR schedulers
        optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                        lr=self.learning_decay, betas=(0.5, 0.999))
        optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=self.learning_decay, betas=(0.5, 0.999))
        optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=self.learning_decay, betas=(0.5, 0.999))

        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(self.nepochs, self.start_epochs, self.decay_epochs).step)
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(self.nepochs, self.start_epochs, self.decay_epochs).step)
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(self.nepochs, self.start_epochs, self.decay_epochs).step)

        device = torch.device("cuda" if (self.gpu is True) else "cpu")

        input_A = torch.empty(self.batch_size, self.inchannel, self.size, self.size, dtype=torch.float32, device=device)
        input_B = torch.empty(self.batch_size, self.outchannel, self.size, self.size, dtype=torch.float32, device=device)
        target_real = torch.full((self.batch_size,), 1.0, dtype=torch.float32, device=device)
        target_fake = torch.full((self.batch_size,), 0.0, dtype=torch.float32, device=device)

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if (self.gpu is True) else torch.Tensor
        input_A = Tensor(self.batch_size, self.inchannel, self.size, self.size)
        input_B = Tensor(self.batch_size, self.outchannel, self.size, self.size)
        target_real = Variable(Tensor(self.batch_size).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(self.batch_size).fill_(0.0), requires_grad=False)

        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()

        # Dataset loader
        transforms_ = [ transforms.Resize(int(self.size*1.2), Image.BICUBIC),
                        transforms.CenterCrop(self.size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]


        dataloader = DataLoader(ImageDataset(self.root, transforms_=transforms_, unaligned=True),
                                batch_size=self.batch_size, shuffle=True, num_workers=self.cpu, drop_last=True)

        # Plot Loss and Images in Tensorboard
        experiment_dir = str(self.tensor_dir)
        os.makedirs(experiment_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

        metric_dict = defaultdict(list)
        n_iters_total = 0

        ###################################
        ###### Training ######
        for epoch in range(self.start_epochs, self.nepochs):
            for i, batch in enumerate(dataloader):

                # Set model input
                real_A = Variable(input_A.copy_(batch['A']))
                real_B = Variable(input_B.copy_(batch['B']))

                ###### Generators A2B and B2A ######
                optimizer_G.zero_grad()

                # Identity loss
                # G_A2B(B) should equal B if real B is fed
                same_B = netG_A2B(real_B)
                loss_identity_B = criterion_identity(same_B, real_B)*5.0 # [batchSize, 3, ImgSize, ImgSize]

                # G_B2A(A) should equal A if real A is fed
                same_A = netG_B2A(real_A)
                loss_identity_A = criterion_identity(same_A, real_A)*5.0 # [batchSize, 3, ImgSize, ImgSize]

                # GAN loss
                fake_B = netG_A2B(real_A)
                pred_fake = netD_B(fake_B).view(-1)
                loss_GAN_A2B = criterion_GAN(pred_fake, target_real) # [batchSize]

                fake_A = netG_B2A(real_B)
                pred_fake = netD_A(fake_A).view(-1)
                loss_GAN_B2A = criterion_GAN(pred_fake, target_real) # [batchSize]

                # Cycle loss
                recovered_A = netG_B2A(fake_B)
                loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0 # [batchSize, 3, ImgSize, ImgSize]

                recovered_B = netG_A2B(fake_A)
                loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0 # [batchSize, 3, ImgSize, ImgSize]

                # Total loss
                loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

                loss_G.backward()
                optimizer_G.step()
                ###################################

                ###### Discriminator A ######
                optimizer_D_A.zero_grad()

                # Real loss
                pred_real = netD_A(real_A).view(-1)
                loss_D_real = criterion_GAN(pred_real, target_real) # [batchSize]

                # Fake loss
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = netD_A(fake_A.detach()).view(-1)
                loss_D_fake = criterion_GAN(pred_fake, target_fake) # [batchSize]

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake)*0.5
                loss_D_A.backward()

                optimizer_D_A.step()
                ###################################

                ###### Discriminator B ######
                optimizer_D_B.zero_grad()

                # Real loss
                pred_real = netD_B(real_B).view(-1)
                loss_D_real = criterion_GAN(pred_real, target_real) # [batchSize]

                # Fake loss
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = netD_B(fake_B.detach()).view(-1)
                loss_D_fake = criterion_GAN(pred_fake, target_fake) # [batchSize]

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake)*0.5
                loss_D_B.backward()

                optimizer_D_B.step()
                ###################################

                metric_dict['loss_G'].append(loss_G.item())
                metric_dict['loss_G_identity'].append(loss_identity_A.item() + loss_identity_B.item())
                metric_dict['loss_G_GAN'].append(loss_GAN_A2B.item() + loss_GAN_B2A.item())
                metric_dict['loss_G_cycle'].append(loss_cycle_ABA.item() + loss_cycle_BAB.item())
                metric_dict['loss_D'].append(loss_D_A.item() + loss_D_B.item())

                for title, value in metric_dict.items():
                    writer.add_scalar('./tensor/{}'.format(title), value[-1], n_iters_total)

                n_iters_total += 1

            print("""
            -----------------------------------------------------------
            Epoch : {} Finished
            Loss_G : {}
            Loss_G_identity : {}
            Loss_G_GAN : {}
            Loss_G_cycle : {}
            Loss_D : {}
            -----------------------------------------------------------
            """.format(epoch, loss_G, loss_identity_A + loss_identity_B, loss_GAN_A2B + loss_GAN_B2A, loss_cycle_ABA + loss_cycle_BAB, loss_D_A + loss_D_B))


            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

            # Save models checkpoints

            if loss_G.item() < 2.5 :
                os.makedirs(os.path.join(self.keras_dir, str(epoch)), exist_ok=True)
                torch.save(netG_A2B.state_dict(), '{}/{}/netG_A2B.pth'.format(self.keras_dir, epoch))
                torch.save(netG_B2A.state_dict(), '{}/{}/netG_B2A.pth'.format(self.keras_dir, epoch))
                torch.save(netD_A.state_dict(), '{}/{}/netD_A.pth'.format(self.keras_dir, epoch))
                torch.save(netD_B.state_dict(), '{}/{}/netD_B.pth'.format(self.keras_dir, epoch))
            elif epoch >= 40 and epoch%40==0 :
                os.makedirs(os.path.join(self.keras_dir, str(epoch)), exist_ok=True)
                torch.save(netG_A2B.state_dict(), '{}/{}/netG_A2B.pth'.format(self.keras_dir, epoch))
                torch.save(netG_B2A.state_dict(), '{}/{}/netG_B2A.pth'.format(self.keras_dir, epoch))
                torch.save(netD_A.state_dict(), '{}/{}/netD_A.pth'.format(self.keras_dir, epoch))
                torch.save(netD_B.state_dict(), '{}/{}/netD_B.pth'.format(self.keras_dir, epoch))
            elif epoch == (self.nepochs - 1):
                self.last_save_eochs = str(os.path.join(self.keras_dir, str(epoch)))
                os.makedirs(os.path.join(self.keras_dir, str(epoch)), exist_ok=True)
                torch.save(netG_A2B.state_dict(), '{}/{}/netG_A2B.pth'.format(self.keras_dir, epoch))
                torch.save(netG_B2A.state_dict(), '{}/{}/netG_B2A.pth'.format(self.keras_dir, epoch))
                torch.save(netD_A.state_dict(), '{}/{}/netD_A.pth'.format(self.keras_dir, epoch))
                torch.save(netD_B.state_dict(), '{}/{}/netD_B.pth'.format(self.keras_dir, epoch))
            print(" \n ---- Saved in " + self.last_save_eochs)

            for title, value in metric_dict.items():
                writer.add_scalar("train/{}_epoch".format(title), np.mean(value), epoch)

        ###################################

    def morphing(self, option = 1):
        self.size = int(self.size)
        self.inchannel = int(self.inchannel)
        self.outchannel = int(self.outchannel)
        self.cpu = int(self.cpu)
        self.gpu = bool(self.gpu)

        gan_generator = (str(self.keras_dir + '/netG_A2B.pth') if option == 1 else str(self.keras_dir + '/netG_B2A.pth'))

        if torch.cuda.is_available() and not (self.gpu is True):
            print("WARNING: You have a CUDA device, so you should probably run with GPU")

        # Network
        netG = (Generator(self.inchannel, self.outchannel) if option == 1 else Generator(self.outchannel, self.inchannel))
        #Homme to femme if 1 else femme to homme

        if self.gpu is True:
            netG.cuda()

        # Load state dicts
        netG.load_state_dict(torch.load(gan_generator))

        # Set test mode
        netG.eval()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if self.gpu else torch.Tensor
        input_A = Tensor(self.batch_size, self.inchannel, self.size, self.size)
        input_B = Tensor(self.batch_size, self.outchannel, self.size, self.size)

        # Dataset loader
        transforms_ = [transforms.Resize((self.size, self.size), Image.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

        cheat_directory = "./cheat"
        os.makedirs(cheat_directory, exist_ok=True)
        os.makedirs(cheat_directory + "/test", exist_ok=True)
        os.makedirs(cheat_directory + "/test" + (str( "/A" if option == 1 else "/B")) , exist_ok = True)
        os.makedirs(cheat_directory + "/test" + (str( "/B" if option == 1 else "/A")) , exist_ok = True)
        cv2.imwrite(str(cheat_directory + "/test" + (str( "/A/" if option == 1 else "/B/")) + str(os.path.basename(self.source_name)) ), self.source)
        cv2.imwrite(str(cheat_directory + "/test" + (str( "/B/" if option == 1 else "/A/")) + str(os.path.basename(self.source_name)) ), self.source)
        dataloader = DataLoader(ImageDataset(cheat_directory, transforms_=transforms_, mode='test'),
                                batch_size=int(1), shuffle=False, num_workers=self.cpu)

        ###################################

        ###### Testing######

        # Create output dirs if they don't exist
        if (not (option == 1)) and (not os.path.exists('./output/ToMale')):
            os.makedirs('./output/ToMale')
        if (option == 1) and (not os.path.exists('./output/ToFemale')):
            os.makedirs('./output/ToFemale')

        for i, batch in enumerate(dataloader):
            if option == 1:
                #input
                real_A = Variable(input_A.copy_(batch['A']))
                fake_B = 0.5*(netG(real_A).data + 1.0)
                save_image(real_A, 'output/ToFemale/A2B_%04d_real.jpg' % (i+1))
                save_image(fake_B, 'output/ToFemale/A2B_%04d.jpg' % (i+1))
                self.target_name = str('output/ToFemale/A2B_%04d.jpg' % (i+1))
            else:
                #input
                real_B = Variable(input_B.copy_(batch['B']))
                fake_A = 0.5*(netG(real_B).data + 1.0)
                save_image(real_B, 'output/ToMale/B2A_%04d_real.jpg' % (i+1))
                save_image(fake_A, 'output/ToMale/B2A_%04d.jpg' % (i+1))
                self.target_name = str('output/ToMale/B2A_%04d.jpg' % (i+1))
            break

        shutil.rmtree(cheat_directory, ignore_errors=True)
        print("###########################")
        return
