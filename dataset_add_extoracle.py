import os
import json
from tqdm import tqdm
from extoracle.utils import greedy_selection
from nltk.tokenize import sent_tokenize, word_tokenize
from multiprocessing import Process, Manager

def add_extoracle(tmp_list, idx_start, share_list):
    for idx, pair in tqdm(enumerate(tmp_list)):
        tgt_line = pair['summary']
        src_line = pair['source']

        src_sentences = [word_tokenize(t) for t in sent_tokenize(src_line)]
        tgt_sentences = [word_tokenize(t) for t in sent_tokenize(tgt_line)]
        summary_length = len(tgt_sentences)
        ids, sents = greedy_selection(src_sentences, tgt_sentences, summary_length)
        ext_oracle = " ".join([" ".join(sents[i]) for i in ids])
        pair['extoracle'] = ext_oracle

        share_list[idx + idx_start] = pair
    return

if __name__ == '__main__':

    processes_count = 12
    print(f'processes = {processes_count}')

    for json_name in ['DISCHARGE', 'ECHO', 'RADIOLOGY']:

        target_dict = None
        with open('./dataset/' + json_name + '_split.json', 'r') as f_read:
            target_dict = json.load(f_read)

        # for train only
        train_list = target_dict['train']

        manager = Manager()
        new_list = manager.list(train_list)

        idx_chunk_size = len(train_list) // processes_count
        idx_chunk_size = idx_chunk_size + 1

        p_list = []
        for p in range(processes_count):
            
            p_list.append(Process(
                target=add_extoracle, 
                args=(
                    train_list[idx_chunk_size * p: idx_chunk_size * (p+1)], 
                    idx_chunk_size * p, 
                    new_list
                )
            ))

        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

        target_dict['train'] = list(new_list)

        with open(os.path.join('./dataset/' + json_name + '_split.json'), 'w', encoding='utf-8') as write_f:
            write_f.write(json.dumps(target_dict))

    print('====== Done ======')