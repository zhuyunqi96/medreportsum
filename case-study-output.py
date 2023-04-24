import sys
sys.path.append("../evalSummaC")
sys.path.append("../evalQuestEval")
import json
import os
import numpy as np
from tqdm import tqdm
import nltk
from rouge_score import rouge_scorer
from evaluate import load

from evalSummaC import get_summac_model
from evalQuestEval import get_qe_model

def postprocess_text(preds):
    preds = [pred.strip() for pred in preds]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    return preds

if __name__ == '__main__':

    rouge_types = ["rouge1", "rouge2", "rougeLsum"]

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
    bertscore = load("bertscore")
    summac_model = get_summac_model("../evalSummaC/summac/summac_conv_vitc_sent_perc_e.bin")
    qe_model = get_qe_model()

    target_list = [
        ["bart_discharge",
        "b2b_dis",
        "t5_dis",
        "bertShare_dis",
        "robertaShare_dis",
        "pegasus_dis",
        "prophetnet_dis",
        "gsum_dis",
        ],

        ["bart_echo",
        "b2b_echo",
        "t5_echo",
        "bertShare_echo",
        "robertaShare_echo",
        "pegasus_echo",
        "prophetnet_echo",
        "gsum_echo",
        ],

        ["bart_rad",
        "b2b_rad",
        "t5_rad",
        "bertShare_rad",
        "robertaShare_rad",
        "pegasus_rad",
        "prophetnet_rad",
        "gsum_rad",
        ]
    ]
    for tgt_list in target_list:
        tgt_ref = "../dataset/DISCHARGE_split.json"
        dataset_name = "dis"
        if "_rad" in tgt_list[0]:
            tgt_ref = "../dataset/RADIOLOGY_split.json"
            dataset_name = "rad"
        elif "_echo" in tgt_list[0]:
            tgt_ref = "../dataset/ECHO_split.json"
            dataset_name = "echo"

        generated_dir = os.listdir('../logs/')
        generated_dir = [
            sub_dir for sub_dir in generated_dir if sub_dir in tgt_list
        ]
        generated_dir = sorted(generated_dir)

        with open(tgt_ref, 'r') as f_read:
            test_dict = json.load(f_read)
        test_dict = test_dict['test']
        list_references = postprocess_text([_['summary'] for _ in test_dict])
        list_sources = [_['source'] for _ in test_dict]

        print(generated_dir)
        write_txt = ""

        predict_list = []
        for sub_dir in generated_dir:
            with open('../logs/' + sub_dir + '/gens.json', 'r') as f_read2:
                predict_dict = json.load(f_read2)
                predict_list.append(postprocess_text(predict_dict.values()))

        pred_size = len(predict_list[0])
        comp_size = len(generated_dir)
        for row_i in tqdm(range(pred_size)):

            ref_sum = list_references[row_i]
            source = list_sources[row_i]
            
            r1_scores = []
            r2_scores = []
            for col_i in range(comp_size):
                prediction = predict_list[col_i][row_i]
                score = scorer.score(ref_sum, prediction)
                r1_scores.append(round(score['rouge1'].fmeasure * 100, 4))
                r2_scores.append(round(score['rouge2'].fmeasure * 100, 4))
            
            r1_std = np.std(r1_scores)
            if r1_std > 5 and len(set(r1_scores)) >= 6:
                predictions = [predict_list[col_i][row_i] for col_i in range(comp_size)]
                bertScoreRes = bertscore.compute(
                    predictions=predictions,
                    references=[ref_sum for col_i in range(comp_size)], lang="en"
                )
                summac_score = summac_model.score(
                    [source for col_i in range(comp_size)], predictions
                )["scores"]
                qe_score = qe_model.corpus_questeval(
                    hypothesis=predictions,
                    sources=[source for col_i in range(comp_size)],
                    list_references=[ref_sum for col_i in range(comp_size)],
                    batch_size=4
                )["ex_level_scores"]

                bertScore_f1 = bertScoreRes['f1']

                write_txt += f"### idx {row_i} ### \n\n"
                for col_i in range(comp_size):
                    write_txt += f"=== {generated_dir[col_i]} === \n"
                    write_txt += f"r1 = {r1_scores[col_i]} \n"
                    write_txt += f"r2 = {r2_scores[col_i]} \n"
                    write_txt += f"bertScore_f1 = {bertScore_f1[col_i]} \n"
                    write_txt += f"summac_score = {summac_score[col_i]} \n"
                    write_txt += f"qe_score = {qe_score[col_i]} \n"
                    write_txt += f"\n"
                    write_txt += f"{predictions[col_i]} \n\n\n"

                write_txt += f"=== ref === \n"

                summac_score = summac_model.score(
                    [source], [ref_sum]
                )["scores"]
                qe_score = qe_model.corpus_questeval(
                    hypothesis=[ref_sum],
                    sources=[source],
                    list_references=[ref_sum],
                    batch_size=1
                )["ex_level_scores"]
                write_txt += f"summac_score = {summac_score[0]} \n"
                write_txt += f"qe_score = {qe_score[0]} \n"
                write_txt += f"\n"
                write_txt += f"{ref_sum}\n\n"

                write_txt += f"=== src === \n"
                write_txt += f"{source}\n\n\n"

        with open('./case-study-' + dataset_name + '.txt', 'w', encoding='utf-8') as write_f:
            write_f.write(write_txt)
        print(f"=== done {dataset_name} ===")

