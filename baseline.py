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
    for i in range(len(seq)-k+1):
        yield seq[i:i+k]


def find_kmers(fn,k=3,limit=1):
    d = collections.defaultdict(int)
    
    
    seq_l = fn.split('N')
    for seq in seq_l:
    	print(seq,k)

        for km in kmers(seq,k):
        	#print(km)
        	d[km] +=1
        
        seq = twin(seq)
        for km in kmers(seq,k):
            d[km] += 1

    d1 = [x for x in d if d[x] <= limit]
    for x in d1:
        del d[x]

    return d
