import collections, sys
from Bio import Seq, SeqIO, SeqRecord
import itertools
from parser import load_csv
import numpy as np

def create_dict_words():
	kmer=3
	nucleotides = ['a', 't', 'g', 'c']
	words=[]
	binary_words={}
	combinations = itertools.product(*itertools.repeat(nucleotides, 3))
	for idx,j in enumerate(combinations):
	    words.append(''.join(j))
	    _ = np.zeros((len(nucleotides)*len(nucleotides)*len(nucleotides)))
	    _[idx] = 1.0
	    binary_words[''.join(j)]=_

	return binary_words


def twin(km):
    return Seq.reverse_complement(km)

def kmers(seq,k):
	kmer=[]
	for i in range(len(seq)-k+1):
		kmer.append(seq[i:i+k])

	return kmer


def binary_representation(fn,k=3,limit=1):
    d = collections.defaultdict(int)
    dict_words = create_dict_words()
    
    seq_l = fn
    kms = kmers(seq_l,k)

    representation=[]
    for idx in range(0,len(kms)-1):
    	f_ =  np.array(dict_words[kms[idx]])
    	s_ = np.array(dict_words[kms[idx+1]])

    	representation.append(np.stack((f_,s_)))
    

    return np.array(representation)

if __name__ == '__main__':
	assert len(sys.argv[1]) > 1

	X,y = load_csv(sys.argv[1])

	

	#a='ACCGATTATGCA'
	X_binarized = [ binary_representation(item) for item in X]
	X_binarized = np.array(X_binarized)
	
	print(X_binarized[0].shape)
	
