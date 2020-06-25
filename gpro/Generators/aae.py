import numpy as np
import tensorflow as tf
import time
from ..ops import Conv1D, Linear, ResBlock
from ..ops.param import params_with_name
from ..ProcessData import load_seq_data,oh2seq,seq2oh,saveseq
from .kmer_statistics import kmer_statistics
import os
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
mpl.use('Agg')

class AAE:
    
    def EncoderNet(self, x, is_training=True, reuse=False):
        with tf.variable_scope("Encoder", reuse=reuse):
            output = Conv1D('Conv1D.1', self.c_dim, self.DIM, 1, x)
            output = tf.reshape(output, [-1, self.SEQ_LEN, self.DIM])
            for i in range(1,1+self.n_layers):
                output = ResBlock(output, self.DIM, self.kernel_size, 'ResBlock.{}'.format(i))
            output = tf.reshape(output, [-1, self.SEQ_LEN*self.DIM])
            z = Linear('z', self.SEQ_LEN*self.DIM, self.Z_DIM, output)
            return z

    def DecoderNet(self, z, apply_softmax=True, is_training=True, reuse=False):
        with tf.variable_scope("Generator", reuse=reuse):
            output = Linear('Dense', self.Z_DIM, self.SEQ_LEN*self.DIM, z)
            output = tf.reshape(output, [-1, self.SEQ_LEN, self.DIM])
            for i in range(1,1+self.n_layers):
                output = ResBlock(output, self.DIM, self.kernel_size, 'ResBlock.{}'.format(i))
            output = Conv1D('Output', self.DIM, self.c_dim, 1, output)
            if apply_softmax == True:
                output = tf.nn.softmax(output)
            return output
    
    def DiscriminatorNet(self, z, is_training=True, reuse=False):
        with tf.variable_scope("Discriminator", reuse=reuse):
            output = Linear('Dense.1',self.Z_DIM, 64*self.Z_DIM,z)
            output = tf.nn.leaky_relu(output)
            output = Linear('output',64*self.Z_DIM, 1,output)
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
    
    def Encoder(self,seq,datatype='str'):
        if datatype == 'str':
            seq = seq2oh(seq,self.charmap)
        num = seq.shape[0]
        batches = math.ceil(num/self.BATCH_SIZE)
        z = []
        for b in range(batches):
            z.append(self.sess.run(self.gen_z,feed_dict={self.real_input:seq[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE,:,:]}))
        z = np.concatenate(z)
        return z
    
    def xent_loss(self, inn, outt):
        return tf.reduce_mean(tf.losses.softmax_cross_entropy(inn,outt))
    
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
    
    def encoder_loss(self,fake):
        return -tf.reduce_mean(fake)
    
    def BuildModel(self,
                   datafile,
                   kernel_size=5,
                   Z_DIM=64,
                   DIM=256,
                   n_layers=5,
                   BATCH_SIZE=64,
                   LAMBDA=10,
                   checkpoint_dir = './generative_model',
                   model_name='aae'
                   ):
        print('loading dataset...')
        self.data,self.charmap,self.invcharmap = load_seq_data(datafile)
        self.data_val = self.data[:self.data.shape[0]//10,:,:]
        self.data = self.data[self.data.shape[0]//10:,:,:]
        self.dataset_num = self.data.shape[0]
        self.SEQ_LEN = self.data.shape[1]
        self.c_dim = self.data.shape[2]
        self.kernel_size = kernel_size
        self.DIM = DIM
        self.Z_DIM = Z_DIM
        self.n_layers = n_layers
        self.BATCH_SIZE = BATCH_SIZE
        self.LAMBDA = LAMBDA
        self.checkpoint_dir = checkpoint_dir
        if os.path.exists(self.checkpoint_dir) == False:
            os.makedirs(self.checkpoint_dir)
        self.model_name = model_name
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),graph=self.graph)
            print('Building model...')
            self.real_input = tf.placeholder(tf.float32, shape=[None, self.SEQ_LEN, self.c_dim],name='input')
            """ Concate Model """
            z = self.EncoderNet(self.real_input)
            z_samples = tf.random_normal(
                        shape=(self.BATCH_SIZE,self.Z_DIM)
                        )
            real_logits = self.DiscriminatorNet(z_samples)
            fake_logits = self.DiscriminatorNet(z,reuse=True)
            out_logits = self.DecoderNet(z,apply_softmax=False)
            """ Loss Function """
            GP = self.gradient_penalty(z_samples,z)
            self.d_loss = self.discriminator_loss(real_logits,fake_logits) + self.LAMBDA*GP
            self.g_loss = self.xent_loss(self.real_input,out_logits)
            self.e_loss = self.encoder_loss(fake_logits)
            """ Generator tensor """
            self.z = tf.placeholder(tf.float32, shape=[None, self.Z_DIM],name='input')
            self.gen_oh = self.DecoderNet(self.z,reuse=True)
            self.gen_z = self.EncoderNet(self.real_input,reuse=True)
            
            self.saver = tf.train.Saver(max_to_keep=1)
        return
    
    def Train(self,
              learning_rate=1e-4,
              beta1=0.5,
              beta2=0.9,
              save_freq = 150,
              epoch=1000,
              sample_dir='./samples',
              log_dir = './log'):
        self.iteration = self.dataset_num // self.BATCH_SIZE
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.save_freq = min(save_freq, self.iteration)
        self.epoch = epoch
        self.sample_dir = sample_dir
        if os.path.exists(self.sample_dir) == False:
            os.makedirs(self.sample_dir)

        self.log_dir = log_dir
        if os.path.exists(self.log_dir) == False:
            os.makedirs(self.log_dir)
            
        with self.graph.as_default():
            #Training parameters
            t_vars = params_with_name('')
            e_vars = [var for var in t_vars if 'Encoder' in var.name]
            d_vars = [var for var in t_vars if 'Discriminator' in var.name]
            g_vars = [var for var in t_vars if 'Generator' in var.name] + e_vars
            
            # optimizers
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.e_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.e_loss, var_list=e_vars)
                self.d_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_vars)
                self.g_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_vars)
            
            # initialize all variables
            self.sess.run(tf.initialize_all_variables())


            # summary writer
            self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

            # loop for epoch
            true_kmer = [kmer_statistics(i, oh2seq(self.data,self.invcharmap)) for i in [4,6,8]]
            val_kmer = [kmer_statistics(i, oh2seq(self.data_val,self.invcharmap)) for i in [4,6,8]]
            val_js = [val_kmer[i].js_with(true_kmer[i]) for i in range(3)]
            print('js_valid: js_4mer: {}, js_6mer: {}, js_8mer: {}'.format(val_js[0],val_js[1],val_js[2]))
            
            true_kmer = [kmer_statistics(i, oh2seq(self.data,self.invcharmap)) for i in [4,6,8]]
            gen = self.inf_train_gen()
            counter = 1
            start_time = time.time()
            conv = 0
            best_js = 1
            train_6js = []
            val_6js = []
            for epoch in range(1, 1+self.epoch):
                # get batch data
                for idx in range(1, 1+self.iteration):

                    # update D network
                    for i in range(3):
                        _data = gen.__next__()
                        _, d_loss = self.sess.run([self.d_opt,self.d_loss,],feed_dict={self.real_input:_data})
                    
                    # updata E network
                    _data = gen.__next__()
                    _, e_loss = self.sess.run([self.e_opt,self.e_loss,],feed_dict={self.real_input:_data})
                    
                    # update G network
                    for i in range(3):
                        _data = gen.__next__()
                        _, g_loss = self.sess.run([self.g_opt,self.g_loss,],feed_dict={self.real_input:_data})
                    
                    # display training status
                    counter += 1
                    
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f, e_loss: %.8f, d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, idx, self.iteration, time.time() - start_time, e_loss, d_loss, g_loss))

                    # save training results
                    if np.mod(idx, self.save_freq) == 0:
                        seq = self.Generator()
                        saveseq(self.sample_dir + '/' + self.model_name + 
                                '_train_{:02d}_{:05d}.txt'.format(epoch, idx + 1)
                                ,seq)

                fake_kmer = [kmer_statistics(i, self.Generator()) for i in [4,6,8]]
                fake_js = [fake_kmer[i].js_with(true_kmer[i]) for i in range(3)]
                val_js = [fake_kmer[i].js_with(val_kmer[i]) for i in range(3)]
                print('Epoch [{}]: js_4mer: {}, js_6mer: {}, js_8mer: {}'.format(
                        epoch,
                        fake_js[0],
                        fake_js[1],
                        fake_js[2]))
                print('Valid Dataset: js_4mer: {}, js_6mer: {}, js_8mer: {}'.format(
                        val_js[0],
                        val_js[1],
                        val_js[2]))
                train_6js.append(fake_js[1])
                val_6js.append(val_js[1])
                if best_js > val_js[1]:
                    best_js = val_js[1]
                    conv = 0
                    self.save(self.checkpoint_dir, counter)
                else:
                    conv += 1
                    if conv > 20:
                        break
                
                # After an epoch, start_batch_id is set to zero
                # non-zero value is only for the first epoch after loading pre-trained model

            pdf = PdfPages(log_dir+'/6mer_frequence.pdf')
            unique_kmers = val_kmer[1].unique_kmers() | fake_kmer[1].unique_kmers()
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
            plt.plot([0,len(train_6js)-1],[val_6js[1],val_6js[1]])
            plt.legend(['JS_train','JS_valid','JS_control'])
            plt.ylabel('JS Distance')
            plt.xlabel('epoch')
            pdf.savefig()
            pdf.close()
        return
    
    def inf_train_gen(self):
        while True:
            np.random.shuffle(self.data)
            for i in range(0, len(self.data)-self.BATCH_SIZE+1, self.BATCH_SIZE):
                yield self.data[i:i+self.BATCH_SIZE,:,:]
                
    def save(self, checkpoint_dir, step):
        with open(checkpoint_dir+ '/' + self.model_name + 'charmap.txt','w') as f:
            for c in self.charmap:
                f.write(c+'\t')
                
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir = './generative_model', model_name='aae'):
        with self.graph.as_default():
            print(" [*] Reading checkpoints...")
            
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
        
#def plot(val,x,name):
#    import matplotlib.pyplot as plt
#    plt.plot(x)
#    plt.plot([0,len(x)],[val,val])
#    plt.savefig(name+'.jpg')
