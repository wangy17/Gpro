import numpy as np
import math
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

class VAE:
    def EncoderNet(self, x, is_training=True, reuse=False):
        with tf.variable_scope("Encoder", reuse=reuse):
            output = Conv1D('Conv1D.1', self.c_dim, self.DIM, 1, x)
            output = tf.reshape(output, [-1, self.SEQ_LEN, self.DIM])
            for i in range(1,1+self.n_layers):
                output = ResBlock(output, self.DIM, self.kernel_size, 'ResBlock.{}'.format(i))
            output = tf.reshape(output, [-1, self.SEQ_LEN*self.DIM])
            z_mean = Linear('z_mean', self.SEQ_LEN*self.DIM, self.Z_DIM, output)
            z_logvar = Linear('z_logvar', self.SEQ_LEN*self.DIM, self.Z_DIM, output)
            return z_mean,z_logvar

    def DecoderNet(self, z, apply_softmax=True, is_training=True, reuse=False):
        with tf.variable_scope("Decoder", reuse=reuse):
            output = Linear('Dense', self.Z_DIM, self.SEQ_LEN*self.DIM, z)
            output = tf.reshape(output, [-1, self.SEQ_LEN, self.DIM])
            for i in range(1,1+self.n_layers):
                output = ResBlock(output, self.DIM, self.kernel_size, 'ResBlock.{}'.format(i))
            output = Conv1D('Output', self.DIM, self.c_dim, 1, output)
            if apply_softmax == True:
                output = tf.nn.softmax(output)
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
        z_mean = []
        z_logvar = []
        for b in range(batches):
            z_mean.append(self.sess.run(self.z_mean,feed_dict={self.real_input:seq[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE,:,:]}))
            z_logvar.append(self.sess.run(self.z_logvar,feed_dict={self.real_input:seq[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE,:,:]}))
        z_mean = np.concatenate(z_mean)
        z_mean = np.reshape(z_mean,(z_mean.shape[0],z_mean.shape[1],1))
        z_logvar = np.concatenate(z_logvar)
        z_logvar = np.reshape(z_logvar,(z_logvar.shape[0],z_logvar.shape[1],1))
        z = np.concatenate([z_mean,z_logvar],axis=2)
        return z
    
    def log_normal_pdf(self,sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
                -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
                axis=raxis)
    
    def compute_loss(self):
        mean, logvar = self.EncoderNet(self.real_input)
        epsilon = tf.random_normal(
                    shape=(32*self.BATCH_SIZE,self.Z_DIM)
                    )
        z = tf.tile(mean,[32,1]) + tf.exp(tf.tile(logvar,[32,1]) / 2) * epsilon
        '''
        x_logit = self.DecoderNet(z,apply_softmax=False)
        cross_ent = tf.nn.softmax_cross_entropy_with_logits(labels=self.real_input,logits=x_logit)
        logpx_z = -tf.reduce_sum(cross_ent, axis=1)
        '''
        x = self.DecoderNet(z)
        logpx_z = tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(x,tf.tile(self.real_input,[32,1,1])),axis=2)),axis=1)
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, tf.tile(mean,[32,1]), tf.tile(logvar,[32,1]))
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
        
    def BuildModel(self,
                   datafile,
                   kernel_size=5,
                   Z_DIM=64,
                   DIM=256,
                   n_layers=5,
                   BATCH_SIZE=64,
                   checkpoint_dir = './generative_model',
                   model_name='vae',):
        """ Load Parameters """
        print('loading dataset...')
        self.data,self.charmap,self.invcharmap = load_seq_data(datafile)
        self.dataset_num = self.data.shape[0]
        self.SEQ_LEN = self.data.shape[1]
        self.c_dim = self.data.shape[2]
        self.kernel_size = kernel_size
        self.DIM = DIM
        self.Z_DIM = Z_DIM
        self.n_layers = n_layers
        self.BATCH_SIZE = BATCH_SIZE
        self.checkpoint_dir = checkpoint_dir
        if os.path.exists(self.checkpoint_dir) == False:
            os.makedirs(self.checkpoint_dir)
        self.model_name = model_name
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),graph=self.graph)
            print('Building model...')
            """ Generator tensor """
            self.real_input = tf.placeholder(tf.float32, shape=[None, self.SEQ_LEN, self.c_dim],name='input')
            self.z = tf.placeholder(tf.float32, shape=[None, self.Z_DIM],name='input')
            self.gen_oh = self.DecoderNet(self.z,reuse=True)
            self.z_mean,self.z_logvar = self.EncoderNet(self.real_input,reuse=True)
            """ Loss Function """
            self.loss = self.compute_loss()
            
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
            # optimizers
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.loss)
            
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
            best_js = 1
            train_6js = []
            val_6js = []
            for epoch in range(1, 1+self.epoch):
                # get batch data
                for idx in range(1, 1+self.iteration):

                    # update network
                    _data = gen.__next__()
                    _, loss = self.sess.run([self.opt,
                                                                   self.loss],
                                                                  feed_dict={self.real_input:_data})

                    # display training status
                    counter += 1
                    
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f, loss: %.8f" \
                          % (epoch, idx, self.iteration, time.time() - start_time, loss))

                    # save training results
                    '''
                    if np.mod(idx, self.save_freq) == 0:
                        seq = self.Generator()
                        saveseq(self.sample_dir + '/' + self.model_name + 
                                '_train_{:02d}_{:05d}.txt'.format(epoch, idx + 1)
                                ,seq)

                    if np.mod(idx, self.save_freq) == 0:
                        self.save(self.checkpoint_dir, counter)
                    '''
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

    def load(self, checkpoint_dir = './generative_model', model_name='vae'):
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