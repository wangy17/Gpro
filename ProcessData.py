import numpy as np

def GetCharMap(seq):
    invcharmap = []
    for s in seq:
        for c in s:
            if c not in invcharmap:
                invcharmap += c
    invcharmap = sorted(invcharmap)
    charmap = {}
    count = 0
    for c in invcharmap:
        charmap[c] = count
        count += 1
    return charmap,invcharmap

def seq2oh(Seqs,charmap):
    Onehot = []
    Length = len(Seqs[0])
    for i in range(len(Seqs)):
        line = np.zeros([Length,len(charmap)],dtype = 'float')
        for j in range(Length):
            line[j,charmap[Seqs[i][j]]] = 1
        Onehot.append(line)
    Onehot = np.array(Onehot)
    return Onehot

def oh2seq(oh,invcharmap):
    Seqs = []
    for i in range(oh.shape[0]):
        seq = str()
        for j in range(oh.shape[1]):
            seq = seq + invcharmap[np.argmax(oh[i,j,:])]
        Seqs.append(seq)
    return Seqs

def saveseq(filename,seq):
    f = open(filename,'w')
    for i in range(len(seq)):
#        f.write('>'+str(i)+'\n')
        f.write(seq[i]+'\n')
    f.close()
    return

def load_seq_data(filename):
    seq = []
    with open(filename,'r') as f:
        for l in f:
            if l[0] == '>' or l[0] == '#':
                continue
            seq.append(str.strip(l))
    charmap,invcharmap = GetCharMap(seq)
    oh = seq2oh(seq,charmap)
    return oh,charmap,invcharmap

def load_fun_data(filename):
    seq = []
    label = []
    with open(filename,'r') as f:
        for l in f:
            l = str.split(l)
            seq.append(l[-2])
            label.append(float(l[-1]))
    label = np.array(label)
    label = np.reshape(label,[label.shape[0],1])
    return seq,label

#def pdf2CDF(pdf):
#    CDF = pdf.copy()
#    for i in range(1,pdf.shape[0]):
#        CDF[i] = CDF[i] + CDF[i-1]
#    return CDF
#
#def Roulette(CDF):
#    temp = random.random()
#    for i in range(CDF.shape[0]):
#        if temp < CDF[i]:
#            return i

