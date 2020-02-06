import base_model
import os
import torch
import torch.nn as nn
import torch.optim as optim
from Loss import Loss

from model import Monodepth
#Need adaptive LR

class Trainer(nn.Module):
    def __init__(self, device, decay, batchnorm=True, pretrained=True, lr=1e-2, momentum=0.9):
        super(Trainer, self).__init__()

        self.model = Monodepth(batchnorm=True).to(device)

        self.optim = optim.SGD([{'params': self.model.parameters()}], lr=lr, momentum=momentum)
        self.criterion = Loss()
        self.lr_lambda = lambda epoch: decay ** epoch
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optim, self.lr_lambda)

    def forward(self, input_L, input_R):

        self.disp = self.model(input_L)
        loss, ap, lr, ds = self.criterion(self.disp, (input_L, input_R))

        self.model.zero_grad()
        loss.backward()
        self.optim.step()

        return loss, ap, lr, ds

    def load(self, state):
        self.model.load_state_dict(state['weight'])

    def save(self, dir, it):
        state_name = os.path.join(dir, 'resnet101_md_{}.pkl'.format(it))

        state = {
            'weight': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'iterations': it,
        }

        torch.save(state, state_name)

    def init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not isinstance(m, base_model.Encoder) and m.weight.requires_grad == True:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def get_disp_images(self):
        disp = self.disp[0] #Get highest level disparity

        disp_images_l = disp[:, 0].unsqueeze(1) #Get left disparity
        disp_images_r = disp[:, 1].unsqueeze(1) #Get right disparity

        return disp_images_l, disp_images_r
