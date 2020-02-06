import os
import argparse
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np

from utils.KITTI import KITTI

from tqdm import tqdm
from tensorboardX import SummaryWriter

from Trainer import Trainer

#Add PP-disparity later
def log_images(writer, img, disp, it):

    images_array = vutils.make_grid(img).to('cpu')
    disp_array = vutils.make_grid(disp * 255).to('cpu').detach()

    writer.add_image('input', images_array, it)
    writer.add_image('disparities', disp_array, it)


def main(args):
    batch_size = args.batch_size
    num_epoch = args.epoch

    raw_text_dir = args.raw_text_dir
    raw_images_dir = args.raw_images_dir
    weight_dir = args.weight_dir

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainset = KITTI(raw_text_dir, raw_images_dir)
    dataloader = torch.utils.data.DataLoader(
                    trainset, batch_size=batch_size,
                    shuffle=True, num_workers=4)
    trainer = Trainer(device, args.decay, batchnorm=True, pretrained=False, lr=args.lr, momentum=args.momentum)
    writer = SummaryWriter()

    if args.resume:
        assert os.path.isfile(args.resume), "{} is not a file.".format(args.resume)
        state = torch.load(args.resume)
        trainer.load(state)
        it = state["iterations"]
        print("Checkpoint is loaded at {} | Iterations: {}".format(args.resume, it))

    else:
        it = 0

    for e in range(1, num_epoch):
        sum_loss = 0
        prog_bar = tqdm(dataloader, desc="Epoch {}".format(e))
        for i, data in enumerate(prog_bar):
            images_l = data[0].to(device)
            images_r = data[1].to(device)

            loss, ap, lr, ds = trainer(images_l, images_r)
            loss = loss.item()/batch_size
            ap = ap.item()/batch_size
            lr = lr.item()/batch_size
            ds = ds.item()/batch_size

            prog_bar.set_postfix(Loss=loss)
            writer.add_scalar('Total Loss', loss, it)
            writer.add_scalar('AP Loss', ap, it)
            writer.add_scalar('LR Loss', lr, it)
            writer.add_scalar('DS Loss', ds, it)

            it += 1
            
            if it % 2000 == 0:
                print('Saving checkpoint...')
                trainer.save(weight_dir, it)
                print("Checkpoint is saved at {} | Iterations: {}".format(weight_dir, it))

                disp_images = trainer.get_disp_images()
                log_images(writer, images_l, disp_images[0], it)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", help="Load weight from file")
    parser.add_argument("--lr", type=float, default=1e-4, required=False, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.8, required=False, help="SGD Momentum")
    parser.add_argument("--decay", type=float, default=0.7, required=False, help="Learning rate decay")
    parser.add_argument("--batch_size", type=int, default=8, required=False, help="Batch size for SGD")
    parser.add_argument("--epoch", type=int, default=50, required=False, help="No. of epoch to train")
    parser.add_argument("--raw_text_dir", type=str, default="./utils/kitti_train_files.txt", required=False, help="Directory: Text file for KITTI training images' directory")
    parser.add_argument("--raw_images_dir", type=str, default="./KITTI/train", required=False, help="Directory: Raw training images")
    parser.add_argument("--weight_dir", type=str, default="./weights", required=False, help="Directory: Weights")
    args = parser.parse_args()

    main(args)
