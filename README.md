# Monodepth-PyTorch
PyTorch implementation of Unsupervised Monocular Depth Estimation with Left-Right Consistency

Original paper: https://arxiv.org/pdf/1609.03677.pdf Godard, Clément, Oisin Mac Aodha, and Gabriel J. Brostow. "Unsupervised monocular depth estimation with left-right consistency." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.

## A Brief Breakdown
In order to circumvent the numerous obstacles involved in the collection of ground truth data for depth estimation solutions (e.g. LIDAR imaging, manual labelling), the authors of the paper presents a novel architecture that performs monocular (i.e. single-image) depth estimation without requiring ground truth.

Although the proposed approach does not require ground truth labels — which most deep learning approaches do — it is not so much as a hack, but is rather made possible by building on solid geometric principles. Given rectified stereo images (i.e. left and right image pairs transformed to share the same plane), one can calculate the pixel-wise disparity of corresponding points in the image pair, which is inversely proportional to depth as viewed from the observer's perspective.

<p align='center'>
  <img width='700' alt="stereo vision disparity" src="https://user-images.githubusercontent.com/19466657/120427450-ed81e100-c33f-11eb-9b19-3512de4b67af.jpeg"/>
  <br/>
  Image from https://johnwlambert.github.io/stereo/
</p>

### Same principle, but with deep learning
The novelty of the approach lies in the fact that the authors solved a major challenge in monocular depth estimation (i.e. acquiring ground truth labels) by taking advantage of stereo vision disparity, developing an unsupervised method in the process. 

<p align='center'>
  <img width='700' alt="stereo vision disparity" src="https://user-images.githubusercontent.com/19466657/120428646-07242800-c342-11eb-9822-f522b0b55f71.png"/>
  <br/>
  Image from https://arxiv.org/pdf/1609.03677.pdf. The architecture of Monodepth featuring disparity prediction using monocular image, evaluated with its binocular counterpart during training.
</p>

Assuming rectified stereo image-pairs as input, the model feeds the left image (arbitrarily, right image also possible) into the neural network and produce two outputs: a left disparity map & a right disparity map. Using those, in principle, one would be, for example, be able to reproduce the right stereo image using the left stereo image plus the right disparity map, and vice versa. 

The model enforces 3 losses for evaluation, and aggregates them during training:

#### Appearance Matching
<p align='center'>
  <img width="450" alt="appearance matching loss formula" src="https://user-images.githubusercontent.com/19466657/120429801-055b6400-c344-11eb-8371-09ca97b1c2f3.png">
  <br/>
  Image from https://arxiv.org/pdf/1609.03677.pdf. The appearance matching loss.
</p>

The appearance matching loss evaluates the reconstruction error using SSIM (Structural Similarity Index). Reconstruction here means producing an approximation of the right image by applying the right disparity map to the left input image. 

#### Disparity Smoothness
<p align='center'>
  <img width="450" alt="disparity smoothness loss formula" src="https://user-images.githubusercontent.com/19466657/120429958-4489b500-c344-11eb-8ff9-37a19bd8143b.png">
  <br/>
  Image from https://arxiv.org/pdf/1609.03677.pdf. The disparity smoothness loss.
</p>

The disparity smoothness loss ensures that the predicted disparities maintain piecewise smoothness and eliminate discontinuities wherever possible. This is done by weighting disparity gradients using the original image gradients, where the weights are high at edges/boundaries, and low over smooth surfaces.

#### Left-Right Consistency
<p align='center'>
  <img width="450" alt="left-right consistency loss formula" src="https://user-images.githubusercontent.com/19466657/120430928-ba425080-c345-11eb-8850-19bf2e1cf8fd.png">
  <br/>
  Image from https://arxiv.org/pdf/1609.03677.pdf. The left-right consistency loss.
</p>

The left-right consistency loss forms the gist of the novel concept introduced in the paper. In essence, it evaluates the difference between the left disparity map and the *projected* right disparity map, and vice versa. This prompts the model to produce left (with the *right* frame as reference) and right (with the *left* frame as reference) disparity maps that are identical. This is because, ideally, the left and right disparity maps are identical, barring any occlusions. 

## Our Implementation

Our implementation of the paper is wholly based on PyTorch, using the ImageNet pretrained ResNet-101 encoder provided natively for transfer learning. However, due to performance issues and resource constraints, we later regressed to the ResNet-50 as our encoder architecture of choice, using in-house implementation.

<p align='center'>
  <img width="600" alt="our implementation of monodepth" src="https://user-images.githubusercontent.com/19466657/120543208-1050db80-c3ba-11eb-9ff8-93a56129e733.png">
  <br/>
  Our initial model using ResNet-101 as the backbone encoder. Later replaced with our implementation of ResNet-50.
</p>

### Pre-processing and Training
Intending to produce as close an implementation to the original paper as possible, we use the KITTI Stereo Evaluation 2015 dataset for both training and testing purposes. Image pre-processing is only limited to random gamma and brightness adjustments as well as random horizontal-axis flips to improve the model's ability to generalize well. 

### Results
Using TensorBoardX, we monitored the training of the model.

Here, only the final attempt is shown, after several modifications to our implementation due to numerous bugs and theoretical inconsistencies with the original paper. 

<p align='center'>
  <img width="600" alt="training loss vs time" src="https://user-images.githubusercontent.com/19466657/120545016-1b0c7000-c3bc-11eb-91c9-a08c24b8fe85.png">
  <br/>
  The training progress of our final model.
</p>

The model seems to converge to a decent extent. However, the results proved to be problematic. 

#### Inference
<p align='center'>
  <img width='500' alt="KITTI images" src="https://user-images.githubusercontent.com/19466657/120545194-59099400-c3bc-11eb-9e6a-819bc464494b.png"/>
  <img width='500' alt="our model predictions" src="https://user-images.githubusercontent.com/19466657/120545200-5a3ac100-c3bc-11eb-9688-4c17ea02a28c.png"/>
  <br/>
  Left-side images from the KITTI Stereo Evaluation 2015 dataset vs. our model's depth prediction @ 23,200 iterations. Artifacts are apparent in certain parts of the predictions, showing disparity 0 at inappropriate areas.
</p>

#### Testing and Evaluation

<p align='center'>
  <img width='700' alt="KITTI images" src="https://user-images.githubusercontent.com/19466657/120545794-15635a00-c3bd-11eb-8de2-b9fb98518585.png"/>
  <img width='700' alt="our model predictions" src="https://user-images.githubusercontent.com/19466657/120545807-185e4a80-c3bd-11eb-85a9-a6ddebb65164.png"/>
  <br/>
  Benchmarking results from the paper vs. our benchmarking results.
</p>

It is quite apparent that our results are going in the wrong direction. Such results can be due to a multitude of factors, even more so when it concerns deep learning. Possible factors could be: gradient explosion, vanishing gradients, theoretical inconsistencies with the original paper, implementation bug etc. Further post-mortem will be required to pinpoint the source of the issue(s).

### Setup
The KITTI Stereo Evaluation 2015 dataset can be downloaded here: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo

The stereo image pairs are pre-calibrated and rectified, so all that is needed is to re-structure the directory to the format that our dataloader requires, which is simply placing the downloaded KITTI dataset in the home directory of the repository. Basically, the training images will be placed at /KITTI/training, where /image_2 is the rectified and calibrated folder of the left images, while /image_3 is the corresponding folder for right images, where each image pair will share the same name in respective folders. (e.g. the left image, */KITTI/training/image_2/000000_10.jpg* corresponds to the right image, */KITTI/training/image_3/000000_10.jpg*).

#### Training
Once the dataset is downloaded and placed correctly, you can run the training script by doing ```python Train.py```

#### Inference
For inference on the KITTI testing dataset, you can do ```python Test.py```, and the corresponding disparity maps will be stored in */disparities/disparities.npy* as a numpy array file.

#### Benchmarking
To evaluate the results (i.e. predicted disparity vs. ground truth disparity), do ```python evaluate_kitti.py ./disparities/disparities.npy [ground truth disparity directory]```. Results will be printed in terms of the benchmarking scores presented by the original paper.

