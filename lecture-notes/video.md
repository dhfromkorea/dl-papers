##videos
mostly adapted from the Stanford's cs231n lecture on video classifaction.

### how is video different from images?
* acrobatics: must watch in full to recognize which technique was performed.
* long-term context: big zoomed-in image of a mouth. signing vs. eating vs. being examined by a dentist


### Dense trajectories and motion boundary descriptors for action recognition Wang et al., 2013 
* https://hal.inria.fr/hal-00725627/document
* motivation: would like to take video that contains a motion as input and classify what type of motion it was (e.g. kiss)
* proposed a better tracking method than KLT trajectories, SIFT trajectories
* the scheme used
	1. divide a frame to a N x N grid and sample a point for each cell for each scale on a pyramid.
	2. track motions of the sampled points for L frames. If motion < threshold, discard them. more on how motion is defined: http://stackoverflow.com/questions/24344930/what-is-the-significance-of-the-eigenvalues-of-an-autocorrelation-matrix-in-imag
	3. make a spatio-temporal dense trajectory polygon (on a pyramid). The shape of polygon will be bascially a M x M cropped region around a tracked point and accumulate them over L frames, and blur horizontallly to form a pyramid.
	4. for each trajectory polygon, we slice them sptially or/and temporally to get 6 different variations of the polygon. Each sliced parts will form a grid where HOG, HOF, MBH will be calculated separately.
	5. concatenate HOG, HOF, MBH results to form a visual descriptor.
	6. map the visual descriptor to one of the K "Words" found using K-means algo on existing training data, following Bag of Word approach. (input feature)
	7. run SVM on the input feature to classify a gesture video

* image gradients: http://stackoverflow.com/questions/19815732/what-is-gradient-orientation-and-gradient-magnitude

* HOG: produces a histogram of image gradients to extract shape/appearance.
* HOF: produces a histogram of optical flows in absolute movements (including background motions by camera)
* MBH: produces a histogram of optical flows but with relative motions (less affected by global motion e.g. camera). basically keeps a second order derivative of x, y = u', v'. assuming camera won't accelerate, but moving objects can.
* Pyramid: downsample images in many levels in various resolutions to be robust for multi-resolution cases and compute optical gradients for large motions. In coarse-to-fine approach, Optical flow gets refined while propagating down from the coarest level using warping and billinear interpolation. http://eric-yuan.me/coarse-to-fine-optical-flow/
* Optical flow: a vector/motion field that describes how a pixel moves from time t to time t+1. Usuallly denoted by U, V vectors where U = dx/dt, V = dy/dt. U,V are derived from Brightness Constancy constraint + other constraints. In Lucase-Kanade method, small motion constraint is added to solve Aperture problem (two unknowns, on equation).
* color encoding: optical flow are visualized in color(direction) and saturation(magnitude). https://people.csail.mit.edu/celiu/SIFTflow/pictures/FlowVisualization.jpg

### Action Recognition with Improved Trajectories
* https://hal.inria.fr/hal-00873267v2/document 
* improve the gesture recognition result by estimating directly camera motion.

### Good features to track
* http://www.ai.mit.edu/courses/6.891/handouts/shi94good.pdf
* edges, corners, textures ... 
* a nice property good features have is scale-invariance or translatin-invariance. Something that can be tracked against camera zoom in out or rotation.

### Two-Frame Motion Estimation Based on Polynomial Expansion
* http://lmi.bwh.harvard.edu/papers/pdfs/gunnar/farnebackSCIA03.pdf
* an optical flow estimation method suggested by Karpathy

### T. Brox and J. Malik, “Large displacement optical flow: Descriptor matching in variational motion estimation,” 2011
* https://lmb.informatik.uni-freiburg.de/Publications/2011/Bro11a/brox_tpami10_ldof.pdf
* proposed a method to estimate optical flow for large motion. 
* improves upon a coarse-to-fine warping strategy that works reasonably well for objects where different parts in each moving object move roughly uniformly. (e.g. a walking human -> different body parts have different velocities)

### 3D Convolutional Neural Networks for Human Action recognition, Ji et al., 2010

* https://ai2-s2-pdfs.s3.amazonaws.com/3c86/dfdbdf37060d5adcff6c4d7d453ea5a8b08f.pdf
* nice diagram to illustrate how 3d conv works: https://arxiv.org/pdf/1412.0767.pdf
* manual feature extraction with slow fusion
* intuition: an image with RGB channels is actually 3d. filter is therefore 3d (x,y,rgb). In this case, the output from the usual 2d convolution is "2d" image. Now suppose you repeat this process many times you will get a set of 2d images that is a 3d volume.
* hard wired layer as a preprocessing function that feeds in greyscale, optical flow x/y, oriented gradient x/y. directly feed in some domain knowledge which the author says works better than random initialization. (but at the end, could be a limiting factor because the network may want to extract features different from the five)
* no parameter sharing? 	

### Large-scale Video Classification with Convolutional Neural Networks, Karpathy et al., 2014
* https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf
* why video is hard (why no ground breaking results yet)
* CNN-> need huge training data, there will be a huge parameter set. will therefore take a long time to train. this slow training time gets to be a bigger problem with video... as it's harder to collect labeled data and more parameters to compute as there's an additional axis (=time)
* two-stream approach: context (full frame but downsampled to half), fovea (center frame).
* generalization performance 
* dataset: http://cs.stanford.edu/people/karpathy/deepvideo/classes.html

### Learning Spatiotemporal Features with 3D Convolutional Networks, Tran et al. 2015
* http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf
* VGGnet but with 3D. performed better than Karpathy's but more poorly than "Beyond Short Snippet...".
* 3D ConvNets are more suitable for spatiotemporal feature
learning compared to 2D ConvNets
* shows a homogeneous architecture (3x3x3 conv, 2x2x2 max pooling stride 2 x 2 x2) works well like VGGnet.
* the first pooling layer has 1 x 2 x 2: the intention of preserving the temporal information in
the early phase

### Two-Stream Convolutional Networks for Action Recognition in Videos, Simonyan and Zisserman 2014
* https://arxiv.org/pdf/1406.2199.pdf
* convnet for videos but with two streams that handle temporal / spatial parts separately
* achieved good results by feeding in optical flows (given enough data, convnet should be able to simulate something simliar to optical flows but since there isnt much data availble at the momeent, this handholding approach seems fruitful)
* spatial stream: pretrained on Imagenet
* temporal stream: encode motion over L frames using one of the two methods
	* optical flows: w x h x 2(dx,dy) x L
	* trajectory
* visualiizated of the 1st conv layers is quite informative. (one of 
the filters simulate MBH)
* multi task learning used (did not understand this part...)

### Sequential Deep Learning for Human Action Recognition, Baccouche et al., 2011
* https://ai2-s2-pdfs.s3.amazonaws.com/12b6/551a0f9f5aa62f7d37f03ebc66631e529c4b.pdf
* suggested Convnet + LSTM. tested on KTH

### Long-term Recurrent Convolutional Networks for Visual Recognition and Description, Donahue et al., 2015
* https://arxiv.org/pdf/1411.4389.pdf
* https://www.youtube.com/watch?v=2BXoj778YU8 (did not quite understand this part)
* tackle three visual problems(activity recognition, image captioning, video description)
* added hand-calculated optical flow to the input (cheating on the sprit of end-to-end learning?) RGB, flowx(normalized), flowy(normalized), magnitude.
* tested on https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/tacos-multi-level-corpus/


### Beyond Short Snippets: Deep Networks for Video Classification
* https://arxiv.org/pdf/1503.08909.pdf
* motivation: want a classifer to remember long-term context (beyond a short clip of a few frames)
* idea:
	* let's keep temporal and spatial pooling separate and be intelligent about where to put temporal pooling and how. i.e. let's use CNN to extract useful spatial info for each frame and use Temporal Pooling layer(s) to consider temporal(motion) info appropriately
	* let's use LSTM with CNN.
* optical flow is important. fed in optical flow images (dx, dy, magnitude) as if it's just another input image by translating dx and dy to the same bound as the input image. OF was fed into both pooling + LSTM.
* conclusion: conv temporal pooling seems to be the winner (one time max pooling over the temporal dimension right after a group of frames passes through CNN). 
* but the go-to architecture has not been decided: a paper that suggests other winners: https://arxiv.org/pdf/1506.01911.pdf

### Delving Deeper into Convolutional Networks for Learning Video Representations, Ballas et al., 2016
* https://arxiv.org/pdf/1511.06432.pdf 
* motivation: so far we've used 3d Convnet (finite context) to consider local motions (happens over a few contiguous frames) vs. RNN (inifite context) to consider global motions (happens over many frames and over). They kinda don't play together well (?)
* idea: let's make CNN recurrent! Take GRU and replace matrix multiplication with convolution. This way, we don't have to do 3d conv.
* achieved empirical results comparable to state-of-the-art approaches previosuly done.

### etc
* Youtube Video Classification Kaggle Competition: https://www.kaggle.com/c/youtube8m
* https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/tacos-	multi-level-corpus/
* KTH: http://www.nada.kth.se/cvap/actions/
* UCF: http://crcv.ucf.edu/data/UCF101.php 
