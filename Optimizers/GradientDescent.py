import numpy as np
import os
import math
from ..ProcessData import seq2oh,oh2seq,saveseq,GetCharMap
import random
import tensorflow as tf
from ..ops.param import *
import time
class GradientDescent():
#    def __init__(self,
#                 generative_model,
#                 predictive_model,
#                 batch_size=32):
#        generative_model.Z_DIM = 64
#        
#        self.z_dim = generative_model.Z_DIM
#        self.batch_size = batch_size
#        self.x = param('z_inputs',tf.random_normal(shape=[self.batch_size,self.z_dim],mean=0,stddev=1))
#        
#        self.generative_model = generative_model
#        """"""
#        self.generative_model.SEQ_LEN = 118
#        self.generative_model.c_dim = 4
#        self.generative_model.kernel_size = 5
#        self.generative_model.DIM = 256
#        self.generative_model.checkpoint_dir = './generative_model'
#        self.generative_model.model_name = 'wgan'
#        """"""
#        self.predictive_model = predictive_model
#        """"""
#        self.predictive_model.DIM = 128
#        self.predictive_model.kernel_size = 5
#        self.predictive_model.checkpoint_dir = './predictive_model'
#        self.predictive_model.model_name = 'cnn'
#        """"""
#        self.seq = self.generative_model.GeneratorNet(self.x,reuse=True)
#        self.score = self.predictive_model.PredictorNet(self.seq,reuse=True)
#        
#        self.loss = -tf.reduce_mean(self.score)
#        
#        gpu_options = tf.GPUOptions(allow_growth=True)
#        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#        self.loss = -tf.reduce_mean(self.score)
#        all_vars =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#        g_vars = [var for var in all_vars if 'Generator' in var.name]
#        
#        self.saver_g = tf.train.Saver(var_list=g_vars)
#        self.load(self.saver_g,self.generative_model.checkpoint_dir,self.generative_model.model_name)
#        
#        p_vars =[var for var in all_vars if 'Predictor' in var.name]
#        
#        self.saver_p = tf.train.Saver(var_list=p_vars)
#        self.load(self.saver_p,self.predictive_model.checkpoint_dir,self.predictive_model.model_name)
    def __init__(self,
                 generative_model,
                 predictive_model,
                 batch_size=32):
        self.z_dim = generative_model.Z_DIM
        self.batch_size = batch_size
        self.x = param('z_inputs',tf.random_normal(shape=[self.batch_size,self.z_dim],mean=0,stddev=1))
        
        self.generative_model = generative_model
        self.predictive_model = predictive_model
        
        self.seq = self.generative_model.GeneratorNet(self.x,reuse=True)
        self.score = self.predictive_model.PredictorNet(self.seq,reuse=True)
        
        self.loss = -tf.reduce_mean(self.score)
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.loss = -tf.reduce_mean(self.score)
        all_vars =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.g_vars = [var for var in all_vars if 'Generator' in var.name]
        self.p_vars =[var for var in all_vars if 'Predictor' in var.name]
        

    def run(self,
            outdir='./GDresult',
            lr=1e-4,
            beta1=0.5,
            beta2=0.9,
            save_freq=100,
            MaxIter=5000):
        self.outdir = outdir
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        if os.path.exists(self.outdir) == False:
            os.makedirs(self.outdir)
        self.MaxIter = MaxIter
        
        # optimizers
        t_vars = params_with_name('z_inputs')
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.opt = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2).minimize(self.loss, var_list=t_vars)


        counter = 1
        start_time = time.time()
        for epoch in range(1, 1+MaxIter):
            # update z
            _, mean_score,scores = self.sess.run([self.opt, self.loss, self.score])
            mean_score = -mean_score
            max_score = np.max(scores)
            # display training status
            counter += 1
            print("Iter: [%4d], time: %4.4f, Mean Score: %.8f, Max Score: %.8f" \
                  % (epoch, time.time() - start_time, mean_score, max_score))

            # save training results for every 100 iterations
            if np.mod(epoch, save_freq) == 0:
                seqs,scores = self.sess.run([self.seq,self.score])
                np.save(outdir+'/ExpIter'+str(epoch),scores)
                seqs = oh2seq(seqs,self.generative_model.invcharmap)
                saveseq(outdir+'/SeqIter'+str(epoch)+'.fa',seqs)
                print('Iter {} was saved!'.format(epoch))

        return
    
    def load(self, saver, checkpoint_dir = None, model_name = None):
        print(" [*] Reading checkpoints...")
        
        if checkpoint_dir == None:
            checkpoint_dir = self.checkpoint_dir
        if model_name == None:
            model_name = self.model_name
            
        with open(checkpoint_dir+ '/' + model_name + 'charmap.txt','r') as f:
            self.invcharmap = str.split(f.read())
            self.charmap = {}
            i=0
            for c in self.invcharmap:
                self.charmap[c] = i
                i+=1
        
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0