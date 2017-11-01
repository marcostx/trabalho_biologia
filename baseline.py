import collections, sys
from Bio import Seq, SeqIO, SeqRecord
import itertools
import numpy as np

def create_dict_words():
	kmer=3
	nucleotides = ['A', 'T', 'G', 'C']
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
    	f_ =  dict_words[kms[idx]]
    	s_ = dict_words[kms[idx+1]]
    	representation.append([f_,s_])
        
    return representation

if __name__ == '__main__':
	a='ACCGATTATGCA'
	binary_representation(a)
