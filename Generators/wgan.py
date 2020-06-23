import numpy as np
import tensorflow as tf
import time
from ..ops import Conv1D, Linear, ResBlock
from ..ops.param import params_with_name
from ..ProcessData import load_seq_data,oh2seq,saveseq
from .kmer_statistics import kmer_statistics
import os
import math
import matplotlib as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.use('Agg')

class WGAN:
    def GeneratorNet(self, z, is_training=True, reuse=False):
        with tf.variable_scope("Generator", reuse=reuse):
            output = Linear('Dense', self.Z_DIM, self.SEQ_LEN*self.DIM, z)
            output = tf.reshape(output, [-1, self.SEQ_LEN, self.DIM])
            for i in range(1,6):
                output = ResBlock(output, self.DIM, self.kernel_size, 'ResBlock.{}'.format(i))
            output = Conv1D('Generator.Output', self.DIM, self.c_dim, 1, output)
            output = tf.nn.softmax(output)
            return output
        
    def DiscriminatorNet(self, x, is_training=True, reuse=False):
        with tf.variable_scope("Discriminator", reuse=reuse):
            output = Conv1D('Conv1D.1', self.c_dim, self.DIM, 1, x)
            output = tf.reshape(output, [-1, self.SEQ_LEN, self.DIM])
            for i in range(1,6):
                output = ResBlock(output, self.DIM, self.kernel_size, 'ResBlock.{}'.format(i))
            output = tf.reshape(output, [-1, self.SEQ_LEN*self.DIM])
            output = Linear('Discriminator.Output', self.SEQ_LEN*self.DIM, 1, output)
            return output
        
    def Generator(self,z=None):
        if z is None:
            z = np.random.normal(size=(self.BATCH_SIZE*100,self.Z_DIM))
        num = z.shape[0]
        batches = math.ceil(num/self.BATCH_SIZE)
        generated_seq = []
        for b in range(batches):
            oh = self.sess.run(self.gen_oh,feed_dict={self.z:z[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE,:]})
            generated_seq.extend(oh2seq(oh,self.invcharmap))
        return generated_seq

    def gradient_penalty(self, real, fake):
        alpha = tf.random_uniform(
                shape=[self.BATCH_SIZE,1,1], 
                minval=0.,
                maxval=1.
                )
        differences = fake - real
        interpolates = real + (alpha*differences)
        gradients = tf.gradients(self.DiscriminatorNet(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        return gradient_penalty
    
    def discriminator_loss(self, real, fake):
        return -tf.reduce_mean(real) + tf.reduce_mean(fake)
    
    def generator_loss(self, fake):
        return -tf.reduce_mean(fake)
    
    def BuildModel(self,
                   datafile,
                   kernel_size=5,
                   Z_DIM=64,
                   DIM=256,
                   BATCH_SIZE=32,
                   LAMBDA=10,
                   checkpoint_dir = './generative_model',
                   model_name='wgan'):
        """ Load Parameters """
        print('loading dataset...')
        self.data,self.charmap,self.invcharmap = load_seq_data(datafile)
        self.dataset_num = self.data.shape[0]
        self.SEQ_LEN = self.data.shape[1]
        self.c_dim = self.data.shape[2]
        self.kernel_size = kernel_size
        self.DIM = DIM
        self.Z_DIM = Z_DIM
        self.BATCH_SIZE = BATCH_SIZE
        self.LAMBDA = LAMBDA
        self.checkpoint_dir = checkpoint_dir
        if os.path.exists(self.checkpoint_dir) == False:
            os.makedirs(self.checkpoint_dir)
        self.model_name = model_name
        
        """ Build Model """
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        print('Building model...')
        self.real_input = tf.placeholder(tf.float32, shape=[None, self.SEQ_LEN, self.c_dim],name='input')
        self.z = tf.placeholder(tf.float32, shape=[None, self.Z_DIM],name='z')
        """ Loss Function """
        real_logits = self.DiscriminatorNet(self.real_input)
        fake_input = self.GeneratorNet(self.z)
        fake_logits = self.DiscriminatorNet(fake_input, reuse=True)

        self.gen_oh = self.GeneratorNet(self.z,reuse=True,is_training=False)
        GP = self.gradient_penalty(real=self.real_input, fake=fake_input)

        self.d_loss = self.discriminator_loss(real=real_logits, fake=fake_logits) + self.LAMBDA*GP
        self.g_loss = self.generator_loss(fake=fake_logits)
        
        """ Summary """
        self.d_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_sum = tf.summary.scalar("g_loss", self.g_loss)
        
        self.saver = tf.train.Saver(max_to_keep=1)
        return
    
    def Train(self,
              d_learning_rate=1e-4,
              g_learning_rate=1e-4,
              beta1=0.5,
              beta2=0.9,
              n_critic=3,
              save_freq = 150,
              epoch=1000,
              sample_dir='./samples',
              log_dir = './log'):
        
        self.iteration = self.dataset_num // self.BATCH_SIZE
        self.d_learning_rate = d_learning_rate
        self.g_learning_rate = g_learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.n_critic = n_critic
        self.save_freq = min(save_freq, self.iteration) 
        self.epoch = epoch
        self.sample_dir = sample_dir
        if os.path.exists(self.sample_dir) == False:
            os.makedirs(self.sample_dir)

        self.log_dir = log_dir
        if os.path.exists(self.log_dir) == False:
            os.makedirs(self.log_dir)
            
        #Training parameters
        d_vars = params_with_name('Discriminator')
        g_vars = params_with_name('Generator')

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.d_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.g_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_vars)
        
        # initialize all variables
        self.sess.run(tf.initialize_all_variables())


        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # loop for epoch
        true_kmer = [kmer_statistics(i, oh2seq(self.data[:self.dataset_num//2,:,:],self.invcharmap)) for i in [4,6,8]]
        val_kmer = [kmer_statistics(i, oh2seq(self.data[self.dataset_num//2:,:,:],self.invcharmap)) for i in [4,6,8]]
        val_js = [val_kmer[i].js_with(true_kmer[i]) for i in range(3)]
        print('js_valid: js_4mer: {}, js_6mer: {}, js_8mer: {}'.format(val_js[0],val_js[1],val_js[2]))
        
        true_kmer = [kmer_statistics(i, oh2seq(self.data,self.invcharmap)) for i in [4,6,8]]
        gen = self.inf_train_gen()
        counter = 1
        start_time = time.time()
        train_6js = []
        val_6js = []
        for epoch in range(0, self.epoch):
            # get batch data
            for idx in range(0, self.iteration):
                # update D network
                for i in range(self.n_critic):
                    _data = gen.__next__()
                    _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                           feed_dict={
                                                                   self.z:self.make_noise(shape=[self.BATCH_SIZE, self.Z_DIM]),
                                                                   self.real_input:_data}
                                                           )
                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                       feed_dict={self.z:self.make_noise(shape=[self.BATCH_SIZE, self.Z_DIM])})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                # save training results for every 300 steps
                if np.mod(idx + 1, self.save_freq) == 0:
                    seq = self.Generator()
                    saveseq(self.sample_dir + '/' + self.model_name + 
                            '_train_{:02d}_{:05d}.txt'.format(epoch, idx + 1)
                            ,seq)

                if np.mod(idx + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)
            fake_kmer = [kmer_statistics(i, self.Generator()) for i in [4,6,8]]
            fake_js = [fake_kmer[i].js_with(true_kmer[i]) for i in range(3)]
            print('Epoch [{}]:js_4mer: {}, js_6mer: {}, js_8mer: {}'.format(
                    epoch,
                    fake_js[0],
                    fake_js[1],
                    fake_js[2]))
            train_6js.append(fake_js[1])
            val_6js.append(fake_kmer[1].js_with(val_kmer[1]))
            
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            # self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)
        
        pdf = PdfPages(log_dir+'/6mer_frequence.pdf')
        unique_kmers = val_kmer[1].unique_kmers + fake_kmer[1].unique_kmers
        p_nat = []
        p_fake = []
        for kmers in unique_kmers:
            p_nat.append(val_kmer[1]._kmer_counts[kmers]/val_kmer[1]._total_kmers)
            p_fake.append(fake_kmer[1]._kmer_counts[kmers]/fake_kmer[1]._total_kmers)
        plt.scatter(p_nat,p_fake)
        plt.xlabel('Natural 6-mer frequences')
        plt.ylabel('Generative 6-mer frequences')
        pdf.savefig()
        pdf.close()
        
        pdf = PdfPages(log_dir+'/6mer_JS_Distance.pdf')
        plt.figure()
        plt.plot(np.arange(len(train_6js)),train_6js)
        plt.plot(np.arange(len(train_6js)),val_6js)
        plt.plot([0,len(train_6js)],[val_6js[1],val_6js[1]])
        plt.xlabel('JS Distance')
        plt.ylabel('epoch')
        pdf.savefig()
        pdf.close()
        return
    
    def inf_train_gen(self):
        while True:
            np.random.shuffle(self.data)
            for i in range(0, len(self.data)-self.BATCH_SIZE+1, self.BATCH_SIZE):
                yield self.data[i:i+self.BATCH_SIZE,:,:]
                
    def make_noise(self,shape):
        return np.random.normal(size=shape)

                
    def save(self, checkpoint_dir, step):
        with open(checkpoint_dir+ '/' + self.model_name + 'charmap.txt','w') as f:
            for c in self.charmap:
                f.write(c+'\t')
                
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir = None, model_name = None):
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
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

