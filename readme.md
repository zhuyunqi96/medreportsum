## **Leveraging Summary Guidance on Medical Report Summarization**

Yunqi Zhu, Xuebing Yang, Yuanyuan Wu, Wensheng Zhang
<<<<<<< HEAD

[pre-print paper](https://arxiv.org/abs/2302.04001)

Download fine-tuned checkpoints through: [OneDrive](https://1drv.ms/f/s!AjwwtI5OAMyOh3RWkzhJ7utPAz2G?e=iTsLmE) or [BaiduPan]()

=======
>>>>>>> d64be36a5724e24f0574c8165d8e3a1f5e606d53

[pre-print paper] (https://arxiv.org/abs/2302.04001)

Download fine-tuned checkpoints through: 

​	[OneDrive] (https://1drv.ms/f/s!AjwwtI5OAMyOh3RWkzhJ7utPAz2G?e=iTsLmE) or [BaiduPan] ()

1. generate "ECHO.json", "DISCHARGE.json", "RADIOLOGY.json" to './dataset'

```bash
# change line 26 in the file, replace it with the path of your mimic-iii dataset (i.e. csv files).
python dataset_to_json.py
```

2. generate "ECHO_indices.json", "ECHO_split.json", etc. to './dataset'

```bash
python dataset_split_TrainEvalTest.py
```

3. add extoracle for train set for "ECHO_split.json" etc. to './dataset'

```bash
python dataset_add_extoracle.py
```

4. sampling oracle, reference from train set, to train, eval and test

```bash
python dataset_sample_augsum.py
```

5. fine-tune BART, T5-large, and BERT2BERT

```bash
# (1) set the "sampleprompt" in the config_*.json file as "sampleprom2", if you want to use oracle guidance.
#     as "sampleprom3", if you want to use reference guidance
# (2) if you want to use original bart to fine-tune, set "use_sampleprompt" in the config_*.json file as false
python run_bart.py config_discharge.json
python run_bart.py config_echo.json
python run_bart.py config_radiology.json

# Note that the followings only implemented original fine-tuning on t5-large and bert2bert 
python run_t5.py config_dis_t5.json
python run_t5.py config_echo_t5.json
python run_t5.py config_rad_t5.json
python run_bert2bert.py config_dis_bert2bert.json
python run_bert2bert.py config_echo_bert2bert.json
python run_bert2bert.py config_rad_bert2bert.json
```