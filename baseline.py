import collections, sys
from Bio import Seq, SeqIO, SeqRecord


def twin(km):
    return Seq.reverse_complement(km)

def kmers(seq,k):
    for i in range(len(seq)-k+1):
        yield seq[i:i+k]


def build(fn,k=3,limit=1):
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

    
#if __name__ == "__main__":
#    if len(sys.argv) < 2: exit("args: <k> ...")
#    k = int(sys.argv[1])
#    fn = 'AAGTTGCGCTAGGGTTAAACTCGGCTAACTCGATTAACATCAGCCGTTTGGTGGCGCAGATTTGCTACTA'
#    d = build(fn,k,1)
    
    