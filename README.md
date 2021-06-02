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

Assuming rectified stereo image-pairs as input, the model feeds the left image (arbitrarily, right image also possible) into the neural network and produce two outputs: a left disparity map & a right disparity map. Using those, in principle, one would be, for example, be able to reproduce the right stereo image using the left stereo image plus the left disparity map, and vice versa. 

The model enforces 3 losses for evaluation, and aggregates them during training:

#### Appearance Matching
<p align='center'>
  <img width="450" alt="appearance matching loss formula" src="https://user-images.githubusercontent.com/19466657/120429801-055b6400-c344-11eb-8371-09ca97b1c2f3.png">
  <br/>
  Image from https://arxiv.org/pdf/1609.03677.pdf. The appearance matching loss.
</p>

The appearance matching loss evaluates the reconstruction error using SSIM (Structural Similarity Index).

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

The left-right consistency loss forms the gist of the novel concept introduced in the paper. In essence, it evaluates the difference between the left disparity map and the *projected* right disparity map, and vice versa. This prompts the model to produce left and right disparity maps that are identical (as they should be), since in reality, there are no "left" or "right" disparity maps, but only one disparity map between any stereo image pairs. 

