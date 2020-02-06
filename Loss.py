import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import BilinearSampling as bs

class Loss(nn.Module):

    #alpha_AP = Appearance Matching Loss weight
    #alpha_DS = Disparity Smoothness Loss weight
    #alpha_LR = Left-Right Consistency Loss weight
    
    def __init__(self, n=4, alpha_AP=0.85, alpha_DS=0.1, alpha_LR=1.0):
        super(Loss, self).__init__()

        self.n = n
        self.alpha_AP = alpha_AP
        self.alpha_DS = alpha_DS
        self.alpha_LR = alpha_LR

    def build_pyramid(self, img, n):

        pyramid = [img]
        h = img.shape[2]
        w = img.shape[3]
        for i in range(n - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            pyramid.append(F.interpolate(pyramid[i], (nh, nw), mode='bilinear', align_corners=True))

        return pyramid

    def x_grad(self, img):

        img = F.pad(img, (0, 1, 0, 0), mode='replicate')
        grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]

        return grad_x

    def y_grad(self, img):

        img = F.pad(img, (0, 0, 0, 1), mode='replicate')
        grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]

        return grad_y

    def get_images(self, pyramid, disp, get):
        if get == 'left':
            return [bs.apply_disparity(pyramid[i], -disp[i]) for i in range(self.n)]
        elif get == 'right':
            return [bs.apply_disparity(pyramid[j], disp[j]) for j in range(self.n)]
        else:
            raise ValueError('Argument get must be either \'left\' or \'right\'')

    def disp_smoothness(self, disp, pyramid):

        disp_x_grad = [self.x_grad(i) for i in disp]
        disp_y_grad = [self.y_grad(j) for j in disp]

        image_x_grad = [self.x_grad(i) for i in pyramid]
        image_y_grad = [self.y_grad(j) for j in pyramid]

        #e^(-|x|) weights, gradient negatively exponential to weights
        #average over all pixels in C dimension
        #but supposed to be locally smooth?
        weights_x = [torch.exp(-torch.mean(torch.abs(i), 1, keepdim=True)) for i in image_x_grad]
        weights_y = [torch.exp(-torch.mean(torch.abs(j), 1, keepdim=True)) for j in image_y_grad]

        smoothness_x = [disp_x_grad[i] * weights_x[i] for i in range(self.n)]
        smoothness_y = [disp_y_grad[j] * weights_y[j] for j in range(self.n)]

        smoothness = [torch.mean(torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])) / 2 ** i for i in range(self.n)]

        return smoothness

    def DSSIM(self, x, y):

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        avgpool = nn.AvgPool2d(3, 1)

        mu_x = [avgpool(x[i]) for i in range(self.n)]
        mu_y = [avgpool(y[j]) for j in range(self.n)]
        mu_x_sq = [mu_x[i] ** 2 for i in range(self.n)]
        mu_y_sq = [mu_y[j] ** 2 for j in range(self.n)]

        #sigma = E[X^2] - E[X]^2
        sigma_x = [avgpool(x[i] ** 2) - mu_x_sq[i] for i in range(self.n)]
        sigma_y = [avgpool(y[j] ** 2) - mu_y_sq[j] for j in range(self.n)]
        #cov = E[XY] - E[X]E[Y]
        cov_xy = [avgpool(x[i] * y[i]) - (mu_x[i] * mu_y[i]) for i in range(self.n)]

        SSIM_top = [(2 * mu_x[i] * mu_y[i] + c1) * (2 * cov_xy[i] + c2) for i in range(self.n)]
        SSIM_bot = [(mu_x_sq[i] + mu_y_sq[i] + c1) * (sigma_x[i] + sigma_y[i] + c2) for i in range(self.n)]

        SSIM = [SSIM_top[i] / SSIM_bot[i] for i in range(self.n)]
        DSSIM = [torch.mean(torch.clamp((1 - SSIM[i]) / 2, 0, 1)) for i in range(self.n)]

        return DSSIM

    def L1(self, pyramid, est):

        L1_loss = [torch.mean(torch.abs(pyramid[i] - est[i])) for i in range(self.n)]

        return L1_loss

    def get_AP(self, left_pyramid, left_est, right_pyramid, right_est):

        #L1 Loss
        left_l1 = self.L1(left_pyramid, left_est)
        right_l1 = self.L1(right_pyramid, right_est)

        #DSSIM
        left_dssim = self.DSSIM(left_pyramid, left_est)
        right_dssim = self.DSSIM(right_pyramid, right_est)

        left_AP = [self.alpha_AP * left_dssim[i] + (1 - self.alpha_AP) * left_l1[i] for i in range(self.n)] 
        right_AP = [self.alpha_AP * right_dssim[j] + (1 - self.alpha_AP) * right_l1[j] for j in range(self.n)]

        AP_loss = sum(left_AP + right_AP)

        return AP_loss * self.alpha_AP

    def get_LR(self, disp_left, disp_right_to_left, disp_right, disp_left_to_right):

        left_LR = [torch.mean(torch.abs(disp_left[i] - disp_right_to_left[i])) for i in range(self.n)]
        right_LR = [torch.mean(torch.abs(disp_right[j] - disp_left_to_right[j])) for j in range(self.n)]

        LR_loss = sum(left_LR + right_LR)

        return LR_loss * self.alpha_LR

    def get_DS(self, disp_left, left_pyramid, disp_right, right_pyramid):

        left_DS = self.disp_smoothness(disp_left, left_pyramid)
        right_DS = self.disp_smoothness(disp_right, right_pyramid)
        
        DS_loss = sum(left_DS + right_DS)

        return DS_loss * self.alpha_DS

    def forward(self, disp, target):

        left, right = target

        #BUILD OUTPUTS
        #Raw data pyramid
        left_pyramid = self.build_pyramid(left, self.n)
        right_pyramid = self.build_pyramid(right, self.n)

        #Estimated disparity pyramid
        disp_left = [i[:, 0, :, :].unsqueeze(1) for i in disp]
        disp_right = [j[:, 1, :, :].unsqueeze(1) for j in disp]

        #Reconstructed images
        left_est = self.get_images(right_pyramid, disp_left, 'left')
        right_est = self.get_images(left_pyramid, disp_right, 'right')

        #x_to_y Projected disparities
        right_to_left_disp = self.get_images(disp_right, disp_left, 'left')
        left_to_right_disp = self.get_images(disp_left, disp_right, 'right')

        #AP, LR, DS Loss
        AP_loss = self.get_AP(left_pyramid, left_est, right_pyramid, right_est)
        LR_loss = self.get_LR(disp_left, right_to_left_disp, disp_right, left_to_right_disp)
        DS_loss = self.get_DS(disp_left, left_pyramid, disp_right, right_pyramid)

        #Total Loss
        total_loss = AP_loss + LR_loss + DS_loss

        return total_loss, AP_loss, LR_loss, DS_loss











