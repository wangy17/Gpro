## Predictive Module
In this module, we integrated CNN and RNN. User can directly call our model to achieve an acceptable result, and also can build your own model.

### Data Set
A txt file, each line contain a promoter sequence with functional score, split by ' ' or '\t'.

For example:
```
exp_yeast.txt:
CAATTGCGTTTGATCTTATTTAATCTCTCTCTAATATTATTATTGAAAATAATATTTCTTTCCTATCAGCAATCGCTTACATATTCATAGTCATTTACTAGGGTATATACCAATAATA	10.753381
AATTTTCCTCGCAGTTTTTCGCTTGTTTAATGCGTATTTTCCCTGGATATATAAAATCCATCTGATTATTCGATAATCTCAAACAAACAACTCAAAACAAAAAAAACTAAATACAACA	12.363241
CCACTTCCTCAATTGCGTTTGATCTTATTTAATCTCTCTCTAATATATAGAAAAAAAAACCATCTGATTATTCGATAATCTCATCTCTAGAACAACACAGTTACTACAACAATCAATC	11.329836
TCTAATATATAAAATACGATTTATTTAGTTGTCTTTGTCAATCCTCATCTTTCTTTACTCATTATTTCATTTCGGTTTTGTCATCTCTAGAACAACACAGTTACTACAACAATCAATC	19.799712
ACCTTTCATCTGTATCCCGTATATTTAAGATGGCGTTTGCTTTCTCCGTTGATTTTTTTCCTTCTTAGTGATTTTTTTTCAAACAAACAACTCAAAACAAAAAAAACTAAATACAACA	13.796239
...
```
### Model Performance
We compared our model with traditional methods such as: SVR, Linear regression, Bayes linear Regression and Gradient boost regression, using 100,000 promoter sequences with expression in yeast. We gradually down-sampled 50,000, 10000, 5000, 1000, and 500 sequences randomly from the database to test the performance of the generated models under different data size. Experiment would be repeated three times to verify the stability of models.

![](https://github.com/WangLabTHU/Gpro/blob/master/predictor.jpg)
### Interface to Optimizers
If you want define a new generative model, and use our Optimize Module, it would be contain following functions:
```
def Predictor(self, seq):
''' 
Input:
    seq: a list of str, each str repressent a promoter sequence
Output:
    exp: a numpy array with shape=(N,), each float represent the functional score of each sequences 
    return exp
```
```
def GradientDescent(self, z, gradient):#(Optional for Gradient Desscent Algorithm)
''' 
Input:
    seq: a list of str, each str repressent a promoter sequence
    gradient: an numpy array with shape=(N,seqlen,c_dim),
              N is the number of sequences you want to generate,
              seqlen is the length of promoter sequences,
              c_dim is dimension of channel (generally set as 4)
Output:
    gradient: a numpy array with shape=(N,seqlen,c_dim),
              N is the number of sequences you want to generate,
              seqlen is the length of promoter sequences,
              c_dim is dimension of channel (generally set as 4)
    return gradient
```
