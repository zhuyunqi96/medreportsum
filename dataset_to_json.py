import os
import json
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import re
from nltk.tokenize import word_tokenize, sent_tokenize

def get_median(ls):
    # sort the list
    ls_sorted = ls.sort()
    # find the median
    if len(ls) % 2 != 0:
        # total number of values are odd
        # subtract 1 since indexing starts at 0
        m = int((len(ls)+1)/2 - 1)
        return ls[m]
    else:
        m1 = int(len(ls)/2 - 1)
        m2 = int(len(ls)/2)
        return (ls[m1]+ls[m2])/2

if __name__ == '__main__':

    mimic3_path = "E:\dischargeSum\mimic-iii-clinical-database-1.4"

    sumInputSec, sumOutputSec = [], []
    with open("./section_titles/mimic_sec_SUM.txt", "r", encoding="UTF-8") as f_read:
        sumSec = f_read.readlines()
        sumSec = [_.strip().lower() for _ in sumSec if len(_.strip()) > 0]
        special_index = sumSec.index('===============')
        
        sumOutputSec = list(sumSec[:special_index])
        sumInputSec = list(sumSec[special_index+1:])

    print(sumInputSec)
    print(sumOutputSec)
        
    file_path_list = []
    for file in [f for f in os.listdir(mimic3_path) if f.endswith(".csv")]:
        file_path = os.path.join(mimic3_path, file)
        file_path_list.append(file_path)

    file_path_list.sort()
    
    chunksize = 50000
    NOTEEVENTS = file_path_list[18]

    # ======= ECHO =======
    tgt_column = 'Echo'
    tgt_df = None
    with pd.read_csv(NOTEEVENTS, chunksize=chunksize) as reader:
        for chunk in reader:
            check_subject = chunk[chunk['CATEGORY'] == tgt_column]
            check_subject = check_subject[check_subject['ISERROR'].isna()] # find rows that are not errors
            if check_subject.shape[0] > 0:
                if tgt_df is None:
                    tgt_df = check_subject.copy()
                else:
                    tgt_df = pd.concat([tgt_df, check_subject])
    print('tgt_df', tgt_df.shape)
    
    echoSec = []
    echoInputSec, echoOutputSec = [], []
    avg_sent_input, avg_word_input = 0, 0
    avg_sent, avg_word, count_text = 0, 0, 0

    with open("./section_titles/mimic_sec_ECHO.txt", "r", encoding="UTF-8") as f_read:
        echoSec = f_read.readlines()
        echoSec = [_.strip().lower() for _ in echoSec if len(_.strip()) > 0]
        special_index = echoSec.index('===============')
        
        echoInputSec = list(echoSec[:special_index])
        echoOutputSec = list(echoSec[special_index+1:])

    data_list = []
    
    for idx in tqdm(range(tgt_df.shape[0])):
        raw_text = tgt_df.iloc[idx]['TEXT'].lower()
        inputText, outputText = '', ''
        
        search_res_list = []
        for section in echoSec:
            search_res = re.search(section, raw_text)
            if search_res is not None:
                indice_0, indice_1 = search_res.span()
                search_res_list.append([indice_0, indice_1, section])

        section_count = len(search_res_list)
        if section_count > 0:
            search_res_list = sorted(search_res_list, key=lambda x:x[0])
            flag1, flag2 = False, False
            for sec_res in search_res_list:
                if sec_res[-1] in echoInputSec:
                    flag1 = True
                elif sec_res[-1] in echoOutputSec:
                    flag2 = True
            if flag1 and flag2:
                for res_i in range(section_count):
                    sec_res = search_res_list[res_i]
                    if res_i == section_count - 1:
                        if sec_res[-1] not in echoOutputSec:
                            print(search_res_list)
                        assert sec_res[-1] in echoOutputSec
                        outputText += raw_text[sec_res[1]:].strip()
                    else:
                        sec_res_next = search_res_list[res_i+1]
                        inputText += raw_text[sec_res[0]:sec_res_next[0]].strip() + '\n'

                inputText = inputText.strip()
                matches = re.findall(r'this study was compared to the prior study of[\w\W]+\*\*].', inputText)
                for _ in matches:
                    inputText = inputText.replace(_, '')
                    
                inputText = inputText.strip()
                outputText = outputText.strip()

                # write source and summary into dict
                data_list.append(
                    {'source': inputText,
                    'summary': outputText}
                )

                token_sent_input = sent_tokenize(inputText)
                token_word_input = word_tokenize(inputText)
                avg_sent_input += len(token_sent_input)
                avg_word_input += len(token_word_input)

                token_sent = sent_tokenize(outputText)
                token_word = word_tokenize(outputText)

                avg_sent += len(token_sent)
                avg_word += len(token_word)
                count_text += 1

    to_write_json = {'data': data_list}

    json_name = "ECHO.json"
    with open(os.path.join('./dataset', json_name), 'w', encoding='utf-8') as write_f:
        write_f.write(json.dumps(to_write_json))
                
    print('echo', tgt_df.shape)
    print(count_text / tgt_df.shape[0])
    print(count_text, 'avg_sent_input', avg_sent_input / count_text, 'avg_word_input', avg_word_input / count_text)
    print(count_text, 'avg_sent', avg_sent / count_text, 'avg_word', avg_word / count_text)
    # input: avg_sent_input 30.86580486303478 avg_word_input 315.29775315481686
    # output: avg_sent 4.158079409048938 avg_word 49.98664204370576
    # count: 16245 / 45794
    # ======= ECHO =======


    # ======= RADIOLOGY =======
    tgt_df = None
    tgt_column = 'Radiology'
    with pd.read_csv(NOTEEVENTS, chunksize=chunksize) as reader:
        for chunk in reader:                
            check_subject = chunk[chunk['CATEGORY'] == tgt_column]
            check_subject = check_subject[check_subject['ISERROR'].isna()] # find rows that are not errors
            
            if check_subject.shape[0] > 0:
                if tgt_df is None:
                    tgt_df = check_subject.copy()
                else:
                    tgt_df = pd.concat([tgt_df, check_subject])
    print('tgt_df', tgt_df.shape)

    radioSec = []
    radioInputSec, radioOutputSec = [], []
    avg_sent_input, avg_word_input = 0, 0
    avg_sent, avg_word, count_text = 0, 0, 0
    section_titles = []
    with open("./section_titles/mimic_sec_RADIOLOGY.txt", "r", encoding="UTF-8") as f_read:
        radioSec = f_read.readlines()
        radioSec = [_.strip().lower() for _ in radioSec if len(_.strip()) > 0]
        special_index = radioSec.index('===============')
        
        radioInputSec = list(radioSec[:special_index])
        radioOutputSec = list(radioSec[special_index+1:])

    data_list = []

    for idx in tqdm(range(tgt_df.shape[0])):
        raw_text = tgt_df.iloc[idx]['TEXT'].lower()

        inputText, outputText = '', ''

        search_res_list = []
        for section in radioSec:
            search_res = re.search(section, raw_text)
            if search_res is not None:
                indice_0, indice_1 = search_res.span()
                search_res_list.append([indice_0, indice_1, section])

        section_count = len(search_res_list)
        if section_count > 0:
            search_res_list = sorted(search_res_list, key=lambda x:x[0])
            flag1, flag2 = False, False
            for sec_res in search_res_list:
                if sec_res[-1] in radioInputSec:
                    flag1 = True
                elif sec_res[-1] in radioOutputSec:
                    flag2 = True
            if flag1 and flag2:            
                for res_i in range(section_count):
                    sec_res = search_res_list[res_i]
                    if res_i == section_count - 1:
                        if sec_res[-1] in radioInputSec:
                            inputText += raw_text[sec_res[0]:].strip()
                        elif sec_res[-1] in radioOutputSec:
                            outputText += raw_text[sec_res[0]:].strip()
                    else:
                        sec_res_next = search_res_list[res_i+1]
                        
                        if sec_res[-1] in radioInputSec:
                            inputText += raw_text[sec_res[0]:sec_res_next[0]].strip() + '\n'
                        elif sec_res[-1] in radioOutputSec:
                            outputText += raw_text[sec_res[1]:sec_res_next[0]].strip() + '\n'
                    
                inputText = inputText.strip()
                outputText = outputText.strip()

                if '______________________________________________________________________________' in outputText:
                    clean_idx = outputText.index('______________________________________________________________________________')
                    outputText = outputText[:clean_idx]

                # write source and summary into dict
                data_list.append(
                    {'source': inputText,
                    'summary': outputText}
                )

                token_sent_input = sent_tokenize(inputText)
                token_word_input = word_tokenize(inputText)
                avg_sent_input += len(token_sent_input)
                avg_word_input += len(token_word_input)

                token_sent = sent_tokenize(outputText)
                token_word = word_tokenize(outputText)

                avg_sent += len(token_sent)
                avg_word += len(token_word)
                count_text += 1
                
    print('RADIOLOGY', tgt_df.shape)
    print(count_text / tgt_df.shape[0])
    print(count_text, 'avg_sent_input', avg_sent_input / count_text, 'avg_word_input', avg_word_input / count_text)
    print(count_text, 'avg_sent', avg_sent / count_text, 'avg_word', avg_word / count_text)
    # input: avg_sent_input 11.182935748326711 avg_word_input 168.4930309311014
    # output: avg_sent 2.9380031419556696 avg_word 46.07858849621777
    # count: 378745 / 522279

    to_write_json = {'data': data_list}

    json_name = "RADIOLOGY.json"
    with open(os.path.join('./dataset', json_name), 'w', encoding='utf-8') as write_f:
        write_f.write(json.dumps(to_write_json))
    # ======= RADIOLOGY =======


    # ======= DISCHARGE =======
    ADMISSIONS = pd.read_csv(file_path_list[0])
    DIAGNOSES_ICD = pd.read_csv(file_path_list[6])
    D_ICD_DIAGNOSES = pd.read_csv(file_path_list[9])

    D_ICD_DIAGNOSES_dict = dict()
    for idx in tqdm(range(D_ICD_DIAGNOSES.shape[0])):
        D_ICD_DIAGNOSES_dict[D_ICD_DIAGNOSES.iloc[idx]['ICD9_CODE']] = D_ICD_DIAGNOSES.iloc[idx]['LONG_TITLE']

    DIAGNOSES_ICD_dict = dict()
    for idx in tqdm(range(DIAGNOSES_ICD.shape[0])):
        hadm_id = DIAGNOSES_ICD.iloc[idx]['HADM_ID']
        new_list = DIAGNOSES_ICD_dict.get(hadm_id)
        if new_list is None:
            new_list = []
            
        _v = DIAGNOSES_ICD.iloc[idx]['ICD9_CODE']
        new_list.append(_v)
        DIAGNOSES_ICD_dict[hadm_id] = new_list
            
    print('DIAGNOSES_ICD_dict', len(DIAGNOSES_ICD_dict))
            
    hadm_id_diagnose = dict()
    for key, value in DIAGNOSES_ICD_dict.items():
        new_list = []
        for _v in value:
            new_value = D_ICD_DIAGNOSES_dict.get(_v)
            new_list.append(new_value)
        hadm_id_diagnose[key] = new_list

    chunks_count = 0
    addmissions_subject_id = [idx for idx in ADMISSIONS['SUBJECT_ID']]
    addmissions_subject_id = list(set(addmissions_subject_id))
    # category_dict = dict()
    eventnote_subject_id = []
    discharge_dict = dict()

    text_category_list = [
        'Discharge summary', 'ECG', 'Echo', 'Case Management', 'General', 'Nursing', 'Nutrition',
        'Pharmacy', 'Physician', 'Rehab Services', 'Respiratory', 'Social Work', 'Consult', 'Radiology', 'Nursing/other']

    discharge_summary_df = None
    with pd.read_csv(NOTEEVENTS, chunksize=chunksize) as reader:
        for chunk in reader:
            chunks_count += 1
            check = chunk.groupby('CATEGORY')['CATEGORY']
    #         check_size = check.size()
    #         for k in check_size.index:
    #             if k not in category_dict.keys():
    #                 category_dict[k] = check_size[k]
    #             else:
    #                 new_value = category_dict[k]
    #                 category_dict[k] = new_value + check_size[k]
                    
            check_subject = chunk[chunk['CATEGORY'] == 'Discharge summary']
            if check_subject.shape[0] > 0:
                check_subject = check_subject[['SUBJECT_ID', 'CHARTDATE','HADM_ID', 'TEXT']]
                check_subject = check_subject.copy(deep=True)
                
                subject_ADMITTIME = [] # init
                subject_DIAGNOSIS = [] # init

                for (idx, hadm_id) in enumerate(check_subject['HADM_ID']):
                    admissions_hadm_id = ADMISSIONS[ADMISSIONS['HADM_ID'] == hadm_id]
                    for cand_idx in range(admissions_hadm_id.shape[0]):
                        if admissions_hadm_id['DISCHTIME'].iloc[cand_idx].split(' ')[0] == check_subject['CHARTDATE'].iloc[idx]:
                            admissions_hadm_id = admissions_hadm_id.iloc[cand_idx]
                            break

                    assert admissions_hadm_id[['DISCHTIME']].shape[0] == 1
                    subject_ADMITTIME.append(admissions_hadm_id['ADMITTIME'])
                    subject_DIAGNOSIS.append(admissions_hadm_id['DIAGNOSIS'])

                check_subject['ADMITTIME'] = subject_ADMITTIME
                check_subject['DIAGNOSIS'] = subject_DIAGNOSIS
                
                for idx in range(check_subject.shape[0]):
                    hadm_id = check_subject['HADM_ID'].iloc[idx]
                    new_dict = dict()
                    new_dict['ADMITTIME'] = check_subject['ADMITTIME'].iloc[idx]
                    new_dict['CHARTDATE'] = check_subject['CHARTDATE'].iloc[idx]
                    new_dict['related_events'] = []
                    discharge_dict[hadm_id] = new_dict
                    
                if discharge_summary_df is None:
                    discharge_summary_df = check_subject.copy()
                else:
                    discharge_summary_df = pd.concat([discharge_summary_df, check_subject])
            
            for category in list(text_category_list):
                check_text = chunk[chunk['CATEGORY'] == category]
                if check_text.shape[0] > 0:
    #                 print(f'=== {category} ===')
    #                 print(check_text['TEXT'].iloc[0])
                    text_category_list.remove(category)
                    
    # sum_noteevents = 0
    # for k, v in category_dict.items():
    #     sum_noteevents += v
    #     print(k, v)
    # print('sum_noteevents', sum_noteevents)
    print('chunks_count', chunks_count)
    print('discharge_summary_df', discharge_summary_df.shape)

    hadm_id_list = discharge_dict.keys()

    otherevents_row_id = []
    with pd.read_csv(NOTEEVENTS, chunksize=chunksize) as reader:
        for chunk in tqdm(reader):
            check_other = chunk[chunk['CATEGORY'] != 'Discharge summary']
            check_other = check_other[check_other['HADM_ID'].isin(hadm_id_list)]
            
            for idx in range(check_other.shape[0]):
                key = check_other['HADM_ID'].iloc[idx]
                new_dict = discharge_dict[key]
                related_events = new_dict['related_events']
                related_events.append(check_other['ROW_ID'].iloc[idx])
                # update
                new_dict['related_events'] = related_events
                discharge_dict[key] = new_dict
            # add row_id to list
            otherevents_row_id.extend(check_other['ROW_ID'].tolist())
    print('otherevents_row_id', len(otherevents_row_id))

    avg_otherevents = 0
    min_count = 1000000000
    max_count = 0
    count_zero = 0
    zero_otherevent_hadm_id = []
    otherevent_counts = []
    for key, value in discharge_dict.items():
        related_events = value['related_events']
        count = len(related_events)
        if count < min_count:
            min_count = count
        if count > max_count:
            max_count = count
        if count == 0:
            count_zero += 1
            zero_otherevent_hadm_id.append(key)
        avg_otherevents += count
        otherevent_counts.append(count)

    median_count = get_median(otherevent_counts)
    # print('avg_otherevents', sum(otherevent_counts) / len(otherevent_counts))
    # print('min_count', min_count, 'max_count', max_count, 'median_count', median_count)
    # print('count_zero', count_zero)

    data_list = []

    # sumInputSec, sumOutputSec
    sumSections = [_ for _ in sumInputSec]
    sumSections.extend(sumOutputSec)
    avg_sent_input, avg_word_input, count_text = 0, 0, 0
    avg_sent, avg_word = 0, 0
    avg_ICD9, avg_IDC9_word = 0, 0
    indices_no_sum_title = []
    for idx in tqdm(range(discharge_summary_df.shape[0])):
        discharge_report = discharge_summary_df.iloc[idx]['TEXT'].lower()
        if 'final diagnoses:' not in discharge_report and 'discharge diagnosis:' not in discharge_report and 'discharge diagnoses:' not in discharge_report:
            indices_no_sum_title.append(idx)
        else:
            search_res_list = []
            for section in sumSections:
                search_res = re.search(section, discharge_report)
                if search_res is not None:
                    indice_0, indice_1 = search_res.span()
                    search_res_list.append([indice_0, indice_1, section])
                    
            search_res_list = sorted(search_res_list, key=lambda x:x[0])
            
            sumInputText, sumOutputText = '', ''
            section_count = len(search_res_list)
            for sec_i in range(section_count):
                sec_tuple = search_res_list[sec_i]
                section_title = sec_tuple[2]
                
                if sec_i == section_count - 1:
                    if section_title in sumInputSec:
                        sumInputText += discharge_report[sec_tuple[0]:].strip() + '\n'
                    elif section_title in sumOutputSec:
                        sumOutputText += discharge_report[sec_tuple[1]:].strip() + '\n'
                    break
                
                next_tuple = search_res_list[sec_i+1]
                if section_title in sumInputSec:
                    sumInputText += discharge_report[sec_tuple[0]:next_tuple[0]].strip() + '\n'
                elif section_title in sumOutputSec:
                    sumOutputText += discharge_report[sec_tuple[1]:next_tuple[0]].strip() + '\n'
                    
            sumInputText = sumInputText.strip()
            sumOutputText = sumOutputText.strip()
            
            sumOutputTextClean = ''
            for sent in sumOutputText.split('\n'):
                if re.search('last name', sent) is not None and re.search('first name', sent) is not None:
                    break
                else:
                    sumOutputTextClean += sent + '\n'
                    
            sumOutputText = sumOutputTextClean.strip()

            # write source and summary into dict
            data_list.append(
                {'source': sumInputText,
                'summary': sumOutputText}
            )

            sumOutputTextClean = None
            
            token_sent_input = sent_tokenize(sumInputText)
            token_word_input = word_tokenize(sumInputText)
            avg_sent_input += len(token_sent_input)
            avg_word_input += len(token_word_input)
            
            token_sent = sent_tokenize(sumOutputText)
            token_word = word_tokenize(sumOutputText)
            
            avg_sent += len(token_sent)
            avg_word += len(token_word)
            count_text += 1
            
            HADM_ID = discharge_summary_df.iloc[idx]['HADM_ID'].astype(np.int64)
            ICD9_diagnose_list = hadm_id_diagnose.get(HADM_ID)
            ICD9_diagnose_list = [_ for _ in ICD9_diagnose_list if _ is not None]
            ICD9_diagnose_text = '\n'.join(ICD9_diagnose_list)
            for _ in ICD9_diagnose_list:
                avg_ICD9 += 1
            avg_IDC9_word += len(word_tokenize(ICD9_diagnose_text))

    print(count_text, 'avg_sent_input', avg_sent_input / count_text, 'avg_word_input', avg_word_input / count_text)
    print(count_text, 'avg_sent', avg_sent / count_text, 'avg_word', avg_word / count_text)

    print('avg_ICD9', avg_ICD9 / count_text, 'avg_IDC9_word', avg_IDC9_word / count_text)

    print(len(indices_no_sum_title), discharge_summary_df.shape[0])
    print(len(indices_no_sum_title) / discharge_summary_df.shape[0])
    discharge_summary_no_sum_title = discharge_summary_df.iloc[indices_no_sum_title]

    # input: avg_sent_input 100.09717855863742 avg_word_input 2162.197660074018
    # output: avg_sent 2.208902065342831 avg_word 28.84263201878308
    # count: 50258 
    # avg_ICD9 11.756874527438418 avg_IDC9_word 68.62742249990052
    # 9394 59652
    # 0.1574800509622477

    to_write_json = {'data': data_list}

    json_name = "DISCHARGE.json"
    with open(os.path.join('./dataset', json_name), 'w', encoding='utf-8') as write_f:
        write_f.write(json.dumps(to_write_json))
    # ======= DISCHARGE =======

    print('====== Done ======')