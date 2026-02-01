import numpy as np
import ablang

from utils.binding_site import get_binding_site, map_id


def mutate(seq,start_position,end_position,size = 1):
    alphabet = ["G", "A", "V", "L", "I", "P", "F", "Y", "W", "S", "T", "C", "M", "N", "Q", "D", "E", "K", "R", "H"]
    index = np.random.choice(np.arange(start_position,end_position),size=size)[0]
    alphabet.remove(seq[index])
    replace_alphabet = np.random.choice(alphabet,size=size)[0]
    seq = list(seq)
    seq[index] = replace_alphabet
    seq = "".join(seq)
    return seq

def mutation_chain(ab_info,cdr_type,size = 1):
    chain = cdr_type[0]
    cdr = ab_info[cdr_type]
    cdr1_range,cdr2_range,cdr3_range = map_id(ab_info, chain)
    mutate_cdr = mutate(cdr, 0, len(cdr) - 1, size=size)
    cdr_range = eval(f'{cdr_type}_range')
    start, end = cdr_range[0], cdr_range[1]
    chain_seq = seq_dict[f'{chain}_cdr']
    chain_seq = list(chain_seq)
    chain_seq[start:end] = list(mutate_cdr)
    chain_seq = "".join(chain_seq)
    return chain_seq

if __name__ == '__main__':
    full_seq, full_label, seq_dict, ab_info, label_dict = get_binding_site('6i9i', 'H', 'L', 'D')
