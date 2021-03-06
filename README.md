## Train coupling localization task

### Running via docker:
`docker run --runtime=nvidia -v path/to/folder/with/images:/data wosel/ese:0.0.3 /data/image1.jpg /data/image2.jpg`

### Running via cmdline:

 - Ensure you are within an environment with `Python 3.8`, `pytorch >= 1.7.0`, `torchvision >= 0.8.2`
 - Download `model.pth` from https://drive.google.com/file/d/13yK1rbHGKcVom6xnOYc5kejbTV6B6qet/view?usp=sharing and place into root directory
 - `pip install -r requirements.txt`
 - `python3 run.py /path/to/image1.jpg /path/to/image2.jpg`

### Training 
See supplied jupyter notebook `train.ipynb` for details. Run either in docker (see above, add `-it --entrypoint /bin/bash` to `docker run` params ) or in same environment as with cmdline. Ensure `jupyter` is available to run.


________________

## Method:

Pretrained (Torchvision supplied - Imagenet) classification Resnet-18 with final layer removed and fully connected layer added with two outputs. Final output coordinate `x` (width) is mean of the two outputs. 
Outputs trained as left-most and right-most point of specified bounding polygon. 



