## Generative Module
In this module, we integrated WGAN, VAE and AAE three deep generative model. User can directly call our model to achieve an acceptable result, and also can build your own model.

### Data Set
A txt file, each line repressent a promoter sequence.

For example:
```
seq_yeast.txt:
TGGTGTGTGTTCCGAAACTGCTTCACCTGCCATTGTTTCGACGGCCACGGCTACTGTGAATGATGTTGTTACGGTCTATCCAACATGGAGGCCACAGACTACGAATGAAGAGTCTGTC
CACATTAGTTAATGCTTTTGATAAAATGTATATAAAGGCTGTCGTAATGTGCAGTAGTAAGGACCTGACTGTGTTTGTGGTTCTCTTCATTCTTGAACCTTGTCATTGGTAAAAGACC
AGAAAAAAAAATCGACATATAAAAGTGGGGCAGATACTTCGTGTGACAATGGCCAATTCAAGCCCTTTGGGCAGATGTTGCCCTTCTTCTTTCTTAAAAAGTCTTAGTACGATTGACC
TTTTTCCTTTATTCAACTTGGCGATGAGCTGAAAATTTTTTTGGTTAAGGACCCTTTAGAAGTATTGAATGTGGGAACAAAGACGACAAAAGGTAGTTTTTTCCTTGACTATACTGGT
AAAAACTTCATTAATAGTTACAACTTTTTCAATATCAAGTTGATTAAGAAAAAGAAAATTATTATGGGTTAGCTGAAAACCGTGTGATGCATGTCGTTTAAGGATTGTGTAAAAAAGT
TAATACTTACCCGATGTTAAATCTTATTTTTTCATGCAGTAAGTAATCATGCAACAAGAAAAACCCGTAATTAAGCGAACATAGAACAACTAGCATCCCCGATAAGACGGAATAGAAT
...
```
### Model Performance
We tested our model performance using 100,000 promoter sequences which were experimentally verified expressed in yeast. We gradually down-sampled 50,000, 10000, 5000, 1000, and 500 sequences randomly from the database to test the performance of the generated models under different data size. Experiment would be repeated three times to verify the stability of models.

![Performance of Generators](https://github.com/WangLabTHU/Gpro/blob/master/generator.jpg)
### Interface to Optimizers
If you want define a new generative model, and use our Optimize Module, it would be contain following functions:
```
def Generator(self, z):
''' 
Input:
    z: an numpy array with shape=(N, z_dim), N is the number of sequences
       you want to generate, z_dim is the dimension of latent space
Output:
    seq: a list of str, each str repressent a promoter sequence
    return seq
```
```
def GradientDescent(self, z, gradient):#(Optional for Gradient Desscent Algorithm)
''' 
Inputs:
    z: an numpy array with shape=(N, z_dim),
       N is the number of sequences you want to generate,
       z_dim is the dimension of latent space
    gradient: an numpy array with shape=(N,seqlen,c_dim),
              N is the number of sequences you want to generate,
              seqlen is the length of promoter sequences,
              c_dim is dimension of channel (generally set as 4)
Output:
    z_new: same as z, the one step optimized result of gradient descent
    return z_new
```
