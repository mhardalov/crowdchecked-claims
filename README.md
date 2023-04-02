# CrowdChecked: Detecting Previously Fact-Checked Claims in Social Media

This repo contains the code and datasets for the paper ["*CrowdChecked: Detecting Previously Fact-Checked Claims in Social Media*"](https://aclanthology.org/2022.aacl-main.22/).

## Abstract

While there has been substantial progress in developing systems to automate fact-checking, they still lack credibility in the eyes of the users. 
Thus, an interesting approach has emerged: to perform automatic fact-checking by verifying whether an input claim has been previously fact-checked by professional fact-checkers and 
to return back an article that explains their decision. This is a sensible approach as people trust manual fact-checking, 
and as many claims are repeated multiple times. Yet, a major issue when building such systems is the small number of 
known tweet--verifying article pairs available for training. Here, we aim to bridge this gap by making use of crowd fact-checking, 
i.e., mining claims in social media for which users have responded with a link to a fact-checking article. 
In particular, we mine a large-scale collection of 330,000 tweets paired with a corresponding fact-checking article. We further 
propose an end-to-end framework to learn from this noisy data based on modified self-adaptive training, in a distant supervision scenario. 
Our experiments on the CLEF'21 CheckThat! test set show improvements over the state of the art by two points absolute. 

## Code and Data

### Models

Models: [stsb-bert-base: CrowdChecked Jaccard 30 + CheckThat 2021 Task 2A](https://huggingface.co/mhardalov/crowdchecked-claim-detect-jac30-stsb).

### Datasets

For the CrowdCheck dataset we provide the following files:

* The retrieved Snopes fact-checking articles (data/clef2021-format/vclaims.tar.gz)[data/clef2021-format/vclaims.tar.gz]. 
* The IDs of the claims from Twitter (data/clef2021-format/tweets-all-ids.tsv.tar.gz)[data/clef2021-format/tweets-all-ids.tsv.tar.gz]. We are sharing only the IDs to comply with the Twitter policies.
* The mapping between the Tweets and their corresponding Snopes articles in the CLEF 2021 format (qrels -- (data/clef2021-format/qrels-train-*)[data/clef2021-format/qrels-train-*]). The suffixes of the files show the filtering method (Cosine and Jaccard similarity) and the cutoff threshold, e.g., `qrels-train-30.tsv.tar.gz` -- jaccard similarity with cutoff threshold of 0.30.
* The similarity predictions from SBERT used in the cosine similarity filtering [data/sbert_predictions_ids.csv.tar.gz](data/sbert_predictions_ids.csv.tar.gz). 

The input and output format is the same as the CheckThat-2021 competitions (Task 2A). Please refer to the input/output format described here -- [CheckThat 2021 Task 2A, 
Input Data Format](https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/tree/master/task2#input-data-format).

TBA

### Requirements

The project uses `poetry` to manage its dependencies. 
You need to run the following commands to install the dependencies and run a shell:

```bash
> poetry install
> poetry shell
```

We provide the corresponding `requirements.txt` for convenience.

### Training
To train the model you can use the following script:

```python
    # CROWDCHECKED_PATH, QRELS_PATH, TWEETS_PATH are resolved from the ids shared in the `data` folder.
    # CLEF_PATH is the path to the `https://gitlab.com/checkthat_lab/clef2021-checkthat-lab` repo
    ${PYTHON_DIR}/python ${TRAINER_DIR}/trainer.py \
        --train_data_path ${CROWDCHECKED_PATH}/${QRELS_PATH} \
        --train_tweets_path ${CROWDCHECKED_PATH}/ ${TWEETS_PATH}.tsv \
        --dev_data_path ${CLEF_PATH}/data/subtask-2a--english/train/qrels-dev.tsv \
        --dev_tweets_path ${CLEF_PATH}/data/subtask-2a--english/train/tweets-train-dev.tsv \
        --test_data_path ${CLEF_PATH}/test-gold/subtask-2a--english/qrels-test.tsv \
        --test_tweets_path ${CLEF_PATH}/test-gold/subtask-2a--english/tweets-test.tsv \
        --vclaims_train_path ${CROWDCHECKED_PATH}/vclaims/ \
        --vclaims_dev_path ${CLEF_PATH}/data/subtask-2a--english/train/vclaims/ \
        --vclaims_test_path ${CLEF_PATH}/data/subtask-2a--english/train/vclaims/ \
        --model_name_or_path ${MODEL_NAME} \
        --output_dir ${OUTPUT_PATH} \
        --cache_dir cache \
        --max_seq_length 128 \
        --do_train \
        --do_eval \
        --do_predict \
        --logging_steps 500 \
        --per_gpu_train_batch_size 32 \
        --per_gpu_eval_batch_size 128 \
        --learning_rate 2e-05 \
        --weight_decay 0.01 \
        --adam_epsilon 1e-08 \
        --max_grad_norm 1.0 \
        --num_train_epochs 10 \
        --warmup_proportion 0.1 \
        --seed ${seed} \
        --remove_dates \
        --overwrite_output_dir
```

## References

Please cite as [[1]](https://aclanthology.org/2022.aacl-main.22/).

[1] M. Hardalov, A. Chernyavskiy, I. Koychev, D, Ilvovsky, P. Nakov ["*CrowdChecked: Detecting Previously Fact-Checked Claims in Social Media*"](https://aclanthology.org/2022.aacl-main.22/). In Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 266–285, Online only.

```
@inproceedings{hardalov-etal-2022-crowdchecked,
    title = "{C}rowd{C}hecked: Detecting Previously Fact-Checked Claims in Social Media",
    author = "Hardalov, Momchil  and
      Chernyavskiy, Anton  and
      Koychev, Ivan  and
      Ilvovsky, Dmitry  and
      Nakov, Preslav",
    booktitle = "Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    series = "AACL-IJCNLP~'22",
    year = "2022",
    address = "Online only",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.aacl-main.22",
    pages = "266--285",
}
```

## License
The dataset is licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode), see the [data/LICENSE](data/LICENSE). 
The code in this repository is licenced under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) [license](LICENSE).
