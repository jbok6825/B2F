import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import torch.optim as optim
from training.configs import *
import numpy as np
from training.configs import *
from training.module.gating import Gating
from training.module.generator import MotionGenerator


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(103, 64)
        self.layer2 = nn.Linear(64, 32)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.float()

        seed = 12347
        self.rng = np.random.RandomState(seed)

        self.gatingNN = Gating(self.device, 32, NUM_EXPERTS, self.rng)
        self.generatorNN = MotionGenerator(self.device, 103, 51, self.rng)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(KEEP_PROB)




    def forward(self, x):

        x[:, 100:103] = x[:, 100:103] *100

        output = F.relu(self.bn1(self.layer1(x)))
        output = F.relu(self.bn2(self.layer2(output)))

        network_output = self.decode(output, x)



        return network_output
    
    def decode(self, latent, generator_input):
        gating_input = latent
        BC = self.gatingNN(gating_input) 
        prediction = self.generatorNN(generator_input, BC, generator_input.shape[0])

        network_out = prediction

        return network_out

    
    def epoch_process(self, epoch, dataloader, writer):
        # p의 확률로 실제 data를 사용하는거임
        # print('Epoch: {} / {}'.format(epoch, NUM_EPOCH))
        print("@@Epoch:", epoch)
        avg_error_train = 0.
        batch_num = len(dataloader)

        for training_data in dataloader:

            gc.collect()
            torch.cuda.empty_cache()
            
            # output = training_data['facial_feature']
            # vae_encoder_input = torch.cat((training_data['past_fullbody_feature'], training_data['fullbody_feature']), dim = 1)
            # generator_input = training_data['past_facial_feature']
            # gating_input = training_data['fullbody_feature']

            output = training_data['arkit'].float()
            input = training_data['flame'].float()
            prediction = self.forward(input)

            loss = F.mse_loss(prediction, output)
            loss_jaw =  F.mse_loss(prediction[:,24], output[:,24]) * 500
            loss = loss + loss_jaw
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Loss_jaw/train", loss_jaw, epoch)

            avg_error_train += loss.item() / batch_num

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

        print('avg error train in student mode : ', avg_error_train)
        print('------------------------------------')

    def train_network(self, dataloader, epoch_num, writer, save_dir, save_step):
        for param_tensor in self.state_dict():
            self.state_dict()[param_tensor] = self.state_dict()[param_tensor].to(self.device)
            self.state_dict()[param_tensor] = self.state_dict()[param_tensor].float()

        self.optimizer = optim.AdamW(self.parameters(),
                                     lr=INIT_LEARNING_RATE, weight_decay=INIT_WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, Te, Tmult)
        self.criterion = nn.MSELoss()
        self.train()

        for epoch in range(1, epoch_num+1):

            self.epoch_process(epoch, dataloader,writer)

            if (epoch > 50 and epoch % save_step == save_step - 1) or epoch == 1:
                save_path = save_dir + "/model"+"_"+str(epoch)+"epoch.pt"
                torch.save(self, save_path)

        save_path = save_dir + "/model"+"_"+str(epoch_num)+"epoch.pt"
        torch.save(self, save_path)
        
        writer.flush()
        writer.close()