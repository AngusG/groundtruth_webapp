# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import json
import sys
import array

# For CLI version
import argparse

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class PostProc:

    def __init__(self, im):

        self.im = im
        self.scale = 1
        self.im_height, self.im_width, self.im_channels = self.im.shape

        #self.im = cv2.resize(self.im, (0,0), fx=self.scale, fy=self.scale)
        #cv2.imshow("input", self.im)
        self.im_rgb = np.array(self.im)

    def segment(self, out_path, img_name, n_segments, sigma):

        self.newIm = self.im;

        seg_file = out_path+img_name+'n_seg'+str(n_segments)+'_sigma'+str(sigma)

        # apply SLIC and extract (approximately) the supplied number of segments
        segments = slic(self.im, n_segments=n_segments, sigma=sigma)

        segments_list = segments.tolist() # nested lists with same data, indices
        json_file = seg_file+'.json'

        with open(json_file, 'w') as f:
                json.dump(segments_list, f, indent=2)

        # show the output of SLIC
        fig = plt.figure("Superpixels -- %d segments" % (n_segments))
        ax = fig.add_subplot(1, 1, 1)
        self.newIm = mark_boundaries(self.im, segments, color=(0, 0, 0)) # fn normalises img bw 1 and 0 apparently
        plt.axis("off")

        self.newIm = (self.newIm * 255.0).astype('u1')
        cv2.imshow("Image with superpixel regions", self.newIm)
        cv2.imwrite(seg_file+'.jpg', self.newIm)
        cv2.waitKey(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', help="path to image to segment")
    parser.add_argument('img_name', help="name of image to segment, JPG assumed")
    parser.add_argument('out_path', help="path to save segmented image")
    parser.add_argument('--dsx', type=int, help="image downsampling ratio in x", default=1)
    parser.add_argument('--dsy', type=int, help="image downsampling ratio in y", default=1)
    parser.add_argument('--n_segments', type=int, help="number of segments to use with slic", default=100)
    parser.add_argument('--sigma', type=int, help="width of gaussian smoothing", default=3)
    args = parser.parse_args()

    img_file = args.img_path + args.img_name + ".JPG"

    im = cv2.imread(img_file)[::args.dsy,::args.dsx]
    newIm = cv2.imread(img_file)[::args.dsy,::args.dsx]

    postprocessor = PostProc(im)
    postprocessor.segment(args.out_path,args.img_name,args.n_segments,args.sigma)
