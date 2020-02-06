import os
import argparse
import numpy as np
import torch
import torchvision.utils as vutils

from tqdm import tqdm

from utils.KITTI import KITTI
from model import Monodepth

def load(model, weight_fn):

    assert os.path.isfile(weight_fn), "{} is not a file.".format(weight_fn)

    state = torch.load(weight_fn)
    weight = state['weight']
    it = state['iterations']
    model.load_state_dict(weight)
    print("Checkpoint is loaded at {} | Iterations: {}".format(weight_fn, it))

def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    raw_text_dir = args.raw_text_dir
    raw_images_dir = args.raw_images_dir
    disparities_dir = args.disparities_dir
    weight_fn = args.weight_fn

    # initialize model in evaluation mode
    model = Monodepth(batchnorm=True).to(device)
    model.eval()

    # load pretrained weights
    load(model, weight_fn)

    # load test dataset
    dataset = KITTI(raw_text_dir, raw_images_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)

    # run test
    disps = np.zeros((200, 256, 512), dtype=np.float32)
    prog_bar = tqdm(dataloader)
    for i, data in enumerate(prog_bar):
        images_l = data[0].to(device)
        disp = model(images_l)

        #FORMAT DISPARITY FILENAME
        # [N, C, H, W] select the first disparity then right disparity
        disps[i] = disp[0][0][0].to('cpu').detach().squeeze()

    np.save(args.disparities_dir + '/disparities.npy', disps)
    print("Testing complete. {} image disparities saved at {}".format(i + 1, disparities_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #FORMAT DIRECTORIES
    parser.add_argument("--raw_text_dir", type=str, default="./utils/kitti_stereo_2015_test_files.txt", required=False, help="Directory: Text file for KITTI testing images' directory")
    parser.add_argument("--raw_images_dir", type=str, default="./KITTI/eval/", required=False, help="Directory: Raw testing images")
    parser.add_argument("--disparities_dir", type=str, default="./disparities", required=False, help="Directory: Disparity results")
    parser.add_argument("--weight_fn", type=str, required=True, help="Directory: Weights")
    args = parser.parse_args()

    main(args)
