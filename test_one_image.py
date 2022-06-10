# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import warnings
import glob
import cv2
import xlwt
import xlrd
from xlutils.copy import copy
import os
import imageio
from utils import (
    read_data,
    imsave,
    merge,
    gradient,
    lrelu,
    weights_spectral_norm,
    l2_norm
)


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



from test_network import FusionNet

Fusion_net = FusionNet()
warnings.filterwarnings('ignore')



class Fusion:
    def imread(self, path, is_grayscale=True):
        """
        Read image using its path.
        Default value  is gray-scale, and image is read by YCbCr format as the paper said.
        """
        if is_grayscale:
            # flatten=True Read the image as a grayscale map.
            return imageio.imread(path, as_gray=True, pilmode='YCbCr').astype(np.float)
        else:
            return imageio.imread(path, pilmode='YCbCr').astype(np.float)

    def imsave(self, image, path):
        return imageio.imsave(path, image)

    def prepare_data(self, dataset):
        self.data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
        data = glob.glob(os.path.join(self.data_dir, "*.png"))
        data.extend(glob.glob(os.path.join(self.data_dir, "*.bmp")))
        data.sort(key=lambda x: int(x[len(self.data_dir) + 1:-4]))
        return data

    def lrelu(x, leak=0.2):
        return tf.maximum(x, leak * x)

    def input_setup(self, index):
        padding = 0
        sub_ir_sequence = []
        sub_vi_sequence = []
        input_ir = self.imread(self.data_ir[index]) / 127.5 - 1.0  # self.imread(self.data_ir[index]) / 255 #
        input_ir = np.lib.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_ir.shape
        input_ir = input_ir.reshape([w, h, 1])
        input_vi = self.imread(self.data_vi[index]) / 127.5 - 1.0
        input_vi = np.lib.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_vi.shape
        input_vi = input_vi.reshape([w, h, 1])
        sub_ir_sequence.append(input_ir)
        sub_vi_sequence.append(input_vi)
        train_data_ir = np.asarray(sub_ir_sequence)
        train_data_vi = np.asarray(sub_vi_sequence)
        return train_data_ir, train_data_vi

    def Fusion(self):
        num_epochs = 2
        num_epoch = 1
        for idx_num in range(num_epoch, num_epochs):
            print("num_epoch:\t", num_epoch)
            while (num_epoch == idx_num):
                model_path = './checkpoint/Fusion.model-' + str(num_epoch)
                fusion_reader = tf.compat.v1.train.NewCheckpointReader(model_path)
                with tf.name_scope('IR_input'):
                    # infrared image patch
                    tf.compat.v1.disable_eager_execution()
                    ir_images = tf.compat.v1.placeholder(tf.float32, [1, None, None, 1], name='ir_images')
                with tf.name_scope('VI_input'):
                    # visible image patch
                    vi_images = tf.compat.v1.placeholder(tf.float32, [1, None, None, 1], name='vi_images')

                with tf.name_scope('fusion'):
                    self.fusion_image, self.feature = Fusion_net.Fusion_model(vi_images, ir_images, fusion_reader)
                with tf.compat.v1.Session() as sess:
                    init_op = tf.compat.v1.global_variables_initializer()
                    sess.run(init_op)
                    ir_path = r'./Test_ir'
                    vi_path = r'./Test_vi'
                    fused_path = 'FusionNet_Results'
                    self.data_ir = self.prepare_data(ir_path)
                    self.data_vi = self.prepare_data(vi_path)
                    for i in range(len(self.data_ir)):
                        train_data_ir, train_data_vi = self.input_setup(i)
                        start = time.time()
                        result, encoding_feature = sess.run([self.fusion_image, self.feature], feed_dict={
                            ir_images: train_data_ir, vi_images: train_data_vi})
                        result = result.squeeze()
                        result = (result + 1) * 127.5
                        end = time.time()
                        image_path = os.path.join(os.getcwd(), fused_path)
                        if not os.path.exists(image_path):
                            os.makedirs(image_path)
                        num = "%02d" % (i + 1)
                        image_path = os.path.join(image_path, num + ".bmp")
                        self.imsave(result, image_path)
                        print("Testing [%d] successfully,Testing time is [%f]" % (i + 1, end - start))
                num_epoch = num_epoch + 1
            tf.compat.v1.reset_default_graph()

if __name__ == '__main__':
    test_Fusion = Fusion()
    test_Fusion.Fusion()
