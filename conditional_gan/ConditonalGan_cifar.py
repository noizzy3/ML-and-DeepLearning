from __future__ import division
import os
import sys
import time
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
#from inception_score import inception_score

F = tf.app.flags.FLAGS


class PreGAN(object):

    def __init__(self, sess):
        self.sess = sess
        if F.dataset != "lsun" and F.inc_score:
            print("Loading inception module")
            self.inception_module = inception_score(self.sess)
            print("Inception module loaded")
        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32,[F.batch_size, F.output_size, F.output_size,F.c_dim],name='real_images')

        self.real_labels = tf.placeholder(tf.float32,[F.batch_size,32])  # was d_labels


        self.z = tf.placeholder(tf.float32, [F.batch_size, F.z_dim], name='z')
        #self.z_sum = tf.histogram_summary("z", self.z)

        self.G_mean, self.G_var, self.G_sample = self.generator(self.z,self.real_labels)
        #self.D, self.D_logits = self.discriminator(self.images, reuse=False)
        self.fake_out, self.classify_out = self.discriminator(self.images, self.real_labels, reuse=False)
        #self.D_, self.D_logits_, = self.discriminator(self.G_mean, reuse=True)
        self.fake_out_, self.classify_out_ = self.discriminator(self.G_mean, self.real_labels, reuse=True) # fake_out and fake_out_ distinguished

        #self.d_sum = tf.histogram_summary("d", self.fake_out)
        #self.d__sum = tf.histogram_summary("d_", self.real_out)
        #self.G_sum = tf.image_summary("G_mean", self.G_mean)
        #self.G_v_sum = tf.histogram_summary("G_var", self.G_var)
        #self.G_v_mean = tf.scalar_summary("G_v_mean", tf.reduce_mean(self.G_var))

        #LOSS FUNCTIONS for DC network......................
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.fake_out,tf.ones_like(self.fake_out))) #tf.nn.l2_loss(self.real_out - self.d_real))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.fake_out_,tf.zeros_like(self.fake_out))) 
                                        #tf.nn.l2_loss(self.fake_out - tf.ones_like(self.fake_out)))
        self.d_loss_D = self.d_loss_real + self.d_loss_fake

        self.d_loss_C = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.classify_out, self.real_labels))
        self.d_loss = self.d_loss_D + self.d_loss_C

    #  LOSS FUCNTION for the Generator part
    
        self.g_loss = tf.constant(0)
        self.g_loss_actual = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.fake_out_,tf.ones_like(self.fake_out)))
                                           #tf.nn.l2_loss(self.fake_out - tf.zeros_like(self.fake_out)))

      # please ignore for time being

        
        #self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        #self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)

        #self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        #self.g_loss_actual_sum = tf.scalar_summary("g_loss_actual", self.g_loss_actual)
        #self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self):
        """Train DCGAN"""
        data = dataset()
        global_step = tf.placeholder(tf.int32, [], name="global_step_epochs")

        # d_optim = tf.train.AdamOptimizer(F.learning_rate_D, beta1=F.beta1D) \
        #                   .minimize(self.d_loss, var_list=self.d_vars)
        # g_optim = tf.train.AdamOptimizer(F.learning_rate_G, beta1=F.beta1G) \
        #                   .minimize(self.g_loss_actual, var_list=self.g_vars)

        learning_rate_D = tf.train.exponential_decay(F.learning_rate_D, global_step,
                                                     decay_steps=F.decay_step,
                                                     decay_rate=F.decay_rate, staircase=True)
        learning_rate_G = tf.train.exponential_decay(F.learning_rate_G, global_step,
                                                     decay_steps=F.decay_step,
                                                     decay_rate=F.decay_rate, staircase=True)
        d_optim = tf.train.AdamOptimizer(learning_rate_D, beta1=F.beta1D).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate_G, beta1=F.beta1G).minimize(self.g_loss_actual, var_list=self.g_vars)

        tf.initialize_all_variables().run()

        #self.g_sum = tf.merge_summary(
        #    [self.z_sum, self.d__sum, self.G_sum, self.G_v_sum, self.G_v_mean, self.d_loss_fake_sum,
        #     self.g_loss_actual_sum])
        #self.d_sum = tf.merge_summary(
        #    [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        #self.writer = tf.train.SummaryWriter(F.log_dir, self.sess.graph)

        counter = 0
        start_time = time.time()

        if F.load_chkpt:
            try:
                self.load(F.checkpoint_dir)
                print(" [*] Load SUCCESS")
            except:
                print(" [!] Load failed...")
        else:
            print(" [*] Not Loaded")

        self.ra, self.rb = -1, 1

        for epoch in xrange(F.epoch):
            idx = 0
            iscore = 0.0, 0.0 # self.get_inception_score()
            batch_iter = data.batch()
            for sample_images in batch_iter:
                #print (".................")
                #print (len(sample_images))
                #print (len(zip(*sample_images)[0]))
                #print (len(zip(*sample_images)[1][0])) #10
                
                
                #sample_images contains the data for the entire batch
                sample_z = np.random.uniform(
                    self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)

                # Update D network
                _,  dlossf = self.sess.run(
                    [d_optim,  self.d_loss_fake],
                    feed_dict={self.images: zip(*sample_images)[0], self.real_labels: zip(*sample_images)[1], self.z: sample_z,
                               global_step: epoch})
                #self.writer.add_summary(summary_str, counter)

                # Update G network
                #iters = max(int(round(-np.log10(dlossf + 1e-10)) / 2), 1)
                iters = 1
                for i in range(iters):
                    _,  gloss, dloss = self.sess.run(
                        [g_optim,  self.g_loss_actual, self.d_loss],
                        feed_dict={self.images: zip(*sample_images)[0], self.real_labels: zip(*sample_images)[1], self.z: sample_z,
                                   global_step: epoch})
                    #self.writer.add_summary(summary_str, counter)
                    sample_z = np.random.uniform(self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)

                errD_fake = self.d_loss_fake.eval({self.images: zip(*sample_images)[0],self.real_labels: zip(*sample_images)[1],self.z: sample_z})

                errD_real = self.d_loss_real.eval({self.images: zip(*sample_images)[0],self.real_labels: zip(*sample_images)[1],self.z: sample_z})
                
                errG = self.g_loss.eval({self.images: zip(*sample_images)[0], 
                    self.real_labels: zip(*sample_images)[1],self.z: sample_z})
                
                errG_actual = self.g_loss_actual.eval({self.images: zip(*sample_images)[0], 
                            self.real_labels: zip(*sample_images)[1], self.z: sample_z})
                lrateD = learning_rate_D.eval({global_step: epoch})
                lrateG = learning_rate_G.eval({global_step: epoch})

                counter += 1
                idx += 1
                print(("Epoch:[%2d] [%4d/%4d] l_D:%.2e l_G:%.2e d_loss_f:%.8f d_loss_r:%.8f " + " g_loss_act:%.2f iters:%d iscore:%f %f") % (epoch, idx, data.num_batches, lrateD, lrateG, errD_fake,errD_real, errG_actual, iters, iscore[0], iscore[1]))

                if np.mod(counter, 100) == 1:
                    samples, d_loss, g_loss = self.sess.run([self.G_mean, self.d_loss, self.g_loss_actual],feed_dict={self.z: sample_z, self.images: zip(*sample_images)[0], self.real_labels: zip(*sample_images)[1]})
                    save_images(samples, [8, 8],F.sample_dir + "/sample.png")
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(F.checkpoint_dir)
                    print("")

            samples, d_loss, g_loss = self.sess.run([self.G_mean, self.d_loss, self.g_loss_actual], feed_dict={self.z: sample_z, self.images: zip(*sample_images)[0], self.real_labels: zip(*sample_images)[1]})
            save_images(samples, [8, 8],F.sample_dir + "/train_{:03d}.png".format(epoch))
            if epoch % 5 == 0:
                iscore = self.get_inception_score()

    def get_inception_score(self):
        if F.dataset == "lsun" or not F.inc_score:
            return 0.0, 0.0

        samples = []
        for k in range(50000 // F.batch_size):
            sample_z = np.random.uniform(
                self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
            images = self.sess.run(self.G_mean, {self.z: sample_z})
            samples.append(images)
        samples = np.vstack(samples)
        return self.inception_module.get_inception_score(samples)

    def discriminator(self, image, l, reuse=False):
        with tf.variable_scope('D'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if F.dataset == "lsun":
                dim = 64
                h0 = lrelu(batch_norm(name='d_bn0')(conv2d(image, dim, name='d_h0_conv')))
                h1 = lrelu(batch_norm(name='d_bn1')(conv2d(h0, dim * 2, name='d_h1_conv')))
                h2 = lrelu(batch_norm(name='d_bn2')(conv2d(h1, dim * 4, name='d_h2_conv')))
                h3 = lrelu(batch_norm(name='d_bn3')(conv2d(h2, dim * 8, name='d_h3_conv')))
                h3 = tf.reshape(h3, [F.batch_size, -1])
                h3 = minibatch_disc(h3, num_kernels=5, scope='d_mb3')
                h4 = linear(h3, 1, 'd_h3_lin')
                return tf.nn.sigmoid(h4), h4

            elif F.dataset == "cifar":
                l1 = tf.expand_dims(l, -1)
                l2 = tf.expand_dims(l1, -1)
                l_ = tf.concat(concat_dim = 2, values = [l2,l2])
                l__ = tf.concat(concat_dim = 2, values = [l2,l_])
                l__ = tf.transpose(l__, perm = [0,3,1,2])
                inputs = tf.concat(concat_dim = 1, values = [image,l__])
                dim = 96
                h0 = lrelu(batch_norm(name='d_bn0')(conv2d(image, dim, 3, 3, 1, 1, name='d_h0_conv')))
                h1 = lrelu(batch_norm(name='d_bn1')(conv2d(h0, dim, 3, 3, 1, 1, name='d_h1_conv')))
                h2 = lrelu(batch_norm(name='d_bn2')(conv2d(h1, dim, 3, 3, 2, 2, name='d_h2_conv')))
                # h2 = tf.nn.dropout(h2, 0.5)
                h3 = lrelu(batch_norm(name='d_bn3')(conv2d(h2, 2 * dim, 3, 3, 1, 1, name='d_h3_conv')))
                h4 = lrelu(batch_norm(name='d_bn4')(conv2d(h3, 2 * dim, 3, 3, 1, 1, name='d_h4_conv')))
                h5 = lrelu(batch_norm(name='d_bn5')(conv2d(h4, 2 * dim, 3, 3, 2, 2, name='d_h5_conv')))
                # h5 = tf.nn.dropout(h5, 0.5)
                h6 = lrelu(batch_norm(name='d_bn6')(conv2d(h5, 2 * dim, 3, 3, 1, 1, name='d_h6_conv')))
                h7 = lrelu(batch_norm(name='d_bn7')(conv2d(h6, 2 * dim, 1, 1, 1, 1, name='d_h7_conv')))
                h8 = tf.reduce_mean(h7, [1, 2])
                h8 = minibatch_disc(h8, num_kernels=50, scope="d_mb_8")
                h_r = linear(h8, 32, 'd_hr_lin') #changed for class labels
                h_f = linear(h8, 1, 'd_hf_lin')
                return h_f, h_r

            else:
                h0_common = tf.nn.tanh(linear(tf.reshape(image, [F.batch_size, -1]), 1000, 'd_h0_common_lin'))
                h1_common = tf.nn.tanh(linear(h0_common, 1000, 'd_h1_common_lin'))
        
                h1_d = minibatch_disc(h1_common, scope='d_h1_d_mbd')
                h2_d = tf.nn.tanh(linear(h1_d, 500, 'd_h2_d_lin'))
                h_fake = linear(h2_d, 1, 'd_fake')

                h2_classifier = tf.nn.tanh(linear(h1_common, 500, 'd_h2_c_lin'))
                h_classifier = linear(h2_classifier, 10, 'd_classifier') #dont make it 32 as the mnist data has 10 class labels
                return h_fake, h_classifier # send unnormalized logits to use in tf.nn.sigmoid_cross_entropy

    def generator(self, z, l):
        with tf.variable_scope("G"):
            if F.dataset == "lsun":
                s = F.output_size
                dim = 64
                s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

                h0 = linear(z, dim * 8 * s16 * s16, scope='g_h0_lin')
                h0 = tf.reshape(h0, [-1, s16, s16, dim * 8])
                h0 = tf.nn.relu(batch_norm(name='g_bn0')(h0))

                h1 = deconv2d(h0, [F.batch_size, s8, s8, dim * 4], name='g_h1')
                h1 = tf.nn.relu(batch_norm(name='g_bn1')(h1))

                h2 = deconv2d(h1, [F.batch_size, s4, s4, dim * 2], name='g_h2')
                h2 = tf.nn.relu(batch_norm(name='g_bn2')(h2))

                h3 = deconv2d(h2, [F.batch_size, s2, s2, dim * 1], name='g_h3')
                h3 = tf.nn.relu(batch_norm(name='g_bn3')(h3))

                h4 = deconv2d(h3, [F.batch_size, s, s, F.c_dim], name='g_h4')
                h4 = tf.nn.tanh(batch_norm(name='g_bn4')(h4))
                mean = h4

                vh0 = linear(z, dim * 4 * s16 * s16, scope='g_vh0_lin')
                vh0 = tf.reshape(vh0, [-1, s16, s16, dim * 4])
                vh0 = tf.nn.relu(batch_norm(name='g_vbn0')(vh0))

                vh1 = deconv2d(vh0, [F.batch_size, s8, s8, dim * 2], name='g_vh1')
                vh1 = tf.nn.relu(batch_norm(name='g_vbn1')(vh1))

                vh2 = deconv2d(vh1, [F.batch_size, s4, s4, dim * 1], name='g_vh2')
                vh2 = tf.nn.relu(batch_norm(name='g_vbn2')(vh2))

                vh3 = deconv2d(vh2, [F.batch_size, s2, s2, dim / 2], name='g_vh3')
                vh3 = tf.nn.relu(batch_norm(name='g_vbn3')(vh3))

                vh4 = deconv2d(vh3, [F.batch_size, s, s, F.c_dim], name='g_vh4')
                vh4 = tf.nn.relu(batch_norm(name='g_vbn4')(vh4))
                var = vh4 + F.eps

                eps = tf.random_normal(mean.get_shape())
                sample = mean + eps * var
                return mean, var, sample

            elif F.dataset == "cifar":
                inputs = tf.concat(concat_dim = 1, values = [z,l])
                s = F.output_size
                dim = 64
                s2, s4, s8 = int(s / 2), int(s / 4), int(s / 8)

                h0 = linear(inputs, dim * 8 * s8 * s8, scope='g_h0_lin')
                h0 = tf.reshape(h0, [-1, s8, s8, dim * 8])
                h0 = tf.nn.relu(batch_norm(name='g_bn0')(h0))

                h1 = deconv2d(h0, [F.batch_size, s4, s4, dim * 4], name='g_h1')
                h1 = tf.nn.relu(batch_norm(name='g_bn1')(h1))

                h2 = deconv2d(h1, [F.batch_size, s2, s2, dim * 2], name='g_h2')
                h2 = tf.nn.relu(batch_norm(name='g_bn2')(h2))

                h3 = deconv2d(h2, [F.batch_size, s, s, F.c_dim], name='g_h3')
                h3 = tf.nn.tanh(batch_norm(name='g_bn3')(h3))
                mean = h3

                vh0 = linear(z, dim * 4 * s8 * s8, scope='g_vh0_lin')
                vh0 = tf.reshape(vh0, [-1, s8, s8, dim * 4])
                vh0 = tf.nn.relu(batch_norm(name='g_vbn0')(vh0))

                vh1 = deconv2d(vh0, [F.batch_size, s4, s4, dim * 2], name='g_vh1')
                vh1 = tf.nn.relu(batch_norm(name='g_vbn1')(vh1))

                vh2 = deconv2d(vh1, [F.batch_size, s2, s2, dim], name='g_vh2')
                vh2 = tf.nn.relu(batch_norm(name='g_vbn2')(vh2))

                vh3 = deconv2d(vh2, [F.batch_size, s, s, F.c_dim], name='g_vh3')
                vh3 = tf.nn.relu(batch_norm(name='g_vbn3')(vh3))
                var = vh3 + F.eps

                eps = tf.random_normal(mean.get_shape())
                sample = mean + eps * var
                return mean, var, sample

            else:
                h0 = tf.nn.tanh(linear(z, 1000, 'g_h0_lin'))
                h1 = tf.nn.tanh(linear(h0, 1000, 'g_h1_lin'))
                h2 = tf.nn.tanh(linear(h1, 784, 'g_h2_lin'))
                mean = h2
                mean = tf.reshape(mean, [F.batch_size, 28, 28, 1])

                vh0 = tf.nn.tanh(linear(z, 1000, 'g_vh0_lin'))
                vh1 = tf.nn.tanh(linear(vh0, 1000, 'g_vh1_lin'))
                vh2 = tf.nn.relu(linear(vh1, 784, 'g_vh2_lin', bias_start=F.var))
                var = vh2 + F.eps
                var = tf.reshape(var, [F.batch_size, 28, 28, 1])
                eps = tf.random_normal(mean.get_shape())

                sample = mean + eps * var
                return mean, var, sample

    def save(self, checkpoint_dir):
        model_name = "model.ckpt"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name))

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

