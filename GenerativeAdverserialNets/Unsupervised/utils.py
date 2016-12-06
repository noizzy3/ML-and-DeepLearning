"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import os
import json
import random
import pprint
import scipy.misc
import numpy as np
import lmdb
import cv2
import tensorflow as tf
from time import gmtime, strftime

pp = pprint.PrettyPrinter()
F = tf.app.flags.FLAGS

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale=False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


class dataset(object):
    def __init__(self):
        if F.dataset == 'mnist':
            self.data = load_mnist()
        elif F.dataset == 'lsun':
            self.data = lmdb.open(F.data_dir, map_size=500000000000,  # 500 gigabytes
                                  max_readers=100, readonly=True)
        elif F.dataset == 'cifar':
            self.data = load_cifar()
        else:
            raise NotImplementedError("Does not support dataset {}".format(F.dataset))

    def batch(self):
        if F.dataset != 'lsun':
            self.num_batches = len(self.data) // F.batch_size
            for i in range(self.num_batches):
                yield self.data[i * F.batch_size:(i + 1) * F.batch_size]

        else:
            self.num_batches = 3033042 // F.batch_size
            with self.data.begin(write=False) as txn:
                cursor = txn.cursor()
                examples = 0
                batch = []
                for key, val in cursor:
                    img = np.fromstring(val, dtype=np.uint8)
                    img = cv2.imdecode(img, 1)
                    img = transform(img)
                    batch.append(img)
                    examples += 1

                    if examples >= F.batch_size:
                        batch = np.asarray(batch)[:, :, :, [2, 1, 0]]
                        yield batch
                        batch = []
                        examples = 0


def load_mnist():
    data_dir = os.path.join("/home/subham/DeepLearning/semisupervised_GAN", "mnist")

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0)

    np.random.shuffle(X)
    return X / 127.5 - 1


def load_cifar():
    def unpickle(file):
        import cPickle
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    trainX1 = unpickle(F.data_dir + '/data_batch_1')
    trainX2 = unpickle(F.data_dir + '/data_batch_2')
    trainX3 = unpickle(F.data_dir + '/data_batch_3')
    trainX4 = unpickle(F.data_dir + '/data_batch_4')
    trainX5 = unpickle(F.data_dir + '/data_batch_5')

    trainX = np.vstack((trainX1['data'], trainX2['data'], trainX3[
                       'data'], trainX4['data'], trainX5['data']))

    trainX = np.asarray(trainX / 127.5, dtype=np.float32) - 1.0
    trainX = trainX.reshape(-1, 3, 32, 32)
    trainX = trainX.transpose(0, 2, 3, 1)
    np.random.shuffle(trainX)
    return trainX


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h: j * h + h, i * w: i * w + w, :] = image

    return img


def imsave(images, size, path):
    return scipy.misc.toimage(merge(images, size), cmax=1.0, cmin=0.0).save(path)


def square_crop(x, npx):
    h, w = x.shape[:2]
    crop_size = min(h, w)
    i = int((h - crop_size) / 2.)
    j = int((w - crop_size) / 2.)
    return scipy.misc.imresize(x[i:i + crop_size, j:j + crop_size],
                               [npx, npx])


def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = square_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


def vgg16(input, name="vgg16"):
    with tf.variable_scope(name) as scope:
        scope_name = tf.get_variable_scope().name

        shp = input.get_shape().as_list()
        if len(shp) == 3:
            input = tf.expand_dims(input, -1)
        images = tf.image.resize_bicubic(input, [224, 224], align_corners=None, name=None)

        with open("vgg16.tfmodel", mode='rb') as f:
            fileContent = f.read()

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)

        tf.import_graph_def(graph_def, input_map={"images": images})
        print("Graph loaded from disk")

        graph = tf.get_default_graph()
        # for op in self.graph.get_operations():
        #   print(op.values())

        feats = graph.get_tensor_by_name(scope_name + "/import/pool4:0")
        return feats


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn is not None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc",
                        "sy": 1, "sx": 1,
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv",
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx) + 2), 2**(int(layer_idx) + 2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'", "").split()))


def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images) / duration * t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x + 1) / 2 * 255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps=len(images) / duration)


def visualize(sess, dcgan, config, option):
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1. / config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      save_images(samples, [8, 8], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1. / config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(100)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      # z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 3:
    values = np.arange(0, 1, 1. / config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1. / config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
          z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10])
                     for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)

