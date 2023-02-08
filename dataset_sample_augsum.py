import json
# from nltk import sent_tokenize, word_tokenize
import spacy

import random
from tqdm import tqdm
import os
random.seed(42)

if __name__ == '__main__':
    nlp_spacy = spacy.load("en_core_web_sm")

    for json_name in ['DISCHARGE', 'ECHO', 'RADIOLOGY']:
        target_dict = None
        with open('./dataset/' + json_name + '_split.json', 'r') as f_read:
            target_dict = json.load(f_read)

        trainset = target_dict['train']
        evalset = target_dict['eval']
        testset = target_dict['test']

        #####
        train_aug_sum = [_['extoracle'] for _ in trainset]
        train_size = len(trainset)
        eval_size = len(evalset)
        test_size = len(testset)
        print('eval_size test_size', eval_size, test_size)

        random.shuffle(train_aug_sum)
        sampled_eval = train_aug_sum[:eval_size]
        sampled_test = train_aug_sum[eval_size:eval_size+test_size]

        for idx in range(eval_size):
            evalset[idx]['sampleprom2'] = sampled_eval[idx]
        for idx in range(test_size):
            testset[idx]['sampleprom2'] = sampled_test[idx]
        
        #####
        train_aug_sum = [_['summary'] for _ in trainset]
        train_size = len(trainset)
        eval_size = len(evalset)
        test_size = len(testset)
        print('eval_size test_size', eval_size, test_size)

        random.shuffle(train_aug_sum)
        sampled_eval = train_aug_sum[:eval_size]
        sampled_test = train_aug_sum[eval_size:eval_size+test_size]
        
        for idx in range(eval_size):
            evalset[idx]['sampleprom3'] = sampled_eval[idx]
        for idx in range(test_size):
            testset[idx]['sampleprom3'] = sampled_test[idx]

        ###########################################

        target_dict['eval'] = evalset
        target_dict['test'] = testset
        target_dict['train'] = trainset

        with open(os.path.join('./' + json_name + '_split.json'), 'w', encoding='utf-8') as write_f:
            write_f.write(json.dumps(target_dict))

        print(f'====== sampleprom {json_name} Done ======')