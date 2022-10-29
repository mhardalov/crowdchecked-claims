"""
This scripts demonstrates how to train a sentence embedding model for Information Retrieval.
As dataset, we use Quora Duplicates Questions, where we have pairs of duplicate questions.
As loss function, we use MultipleNegativesRankingLoss. Here, we only need positive pairs, i.e., pairs of sentences/texts that are considered to be relevant. Our dataset looks like this (a_1, b_1), (a_2, b_2), ... with a_i / b_i a text and (a_i, b_i) are relevant (e.g. are duplicates).
MultipleNegativesRankingLoss takes a random subset of these, for example (a_1, b_1), ..., (a_n, b_n). a_i and b_i are considered to be relevant and should be close in vector space. All other b_j (for i != j) are negative examples and the distance between a_i and b_j should be maximized. Note: MultipleNegativesRankingLoss only works if a random b_j is likely not to be relevant for a_i. This is the case for our duplicate questions dataset: If a sample randomly b_j, it is unlikely to be a duplicate of a_i.
The model we get works well for duplicate questions mining and for duplicate questions information retrieval. For question pair classification, other losses (like OnlineConstrativeLoss) work better.
"""
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, evaluation, losses, util
from sentence_transformers.readers import InputExample
from sklearn.metrics.pairwise import cosine_distances
from torch.utils.data import DataLoader

from semi_claim.data import load_vclaims, read_dataset
from semi_claim.utils import CustomJSONEncoder, configure_logging, parse_args, set_seed

#### Just some code to print debug information to stdout

logger = logging.getLogger(__name__)


def trainer(args):
    model = SentenceTransformer(args.model_name_or_path, cache_folder=args.cache_dir)
    model.max_seq_length = args.max_seq_length

    vclaims_dict = defaultdict(dict)

    for subset, vclaims_path in {
        "train": args.vclaims_train_path,
        "dev": args.vclaims_dev_path,
        "test": args.vclaims_test_path,
    }.items():
        if vclaims_path is None or not vclaims_path.exists():
            continue

        vclaims_dict[subset]["vclaims"] = load_vclaims(vclaims_path)
        vclaims_dict[subset]["ir_corpus"] = (
            vclaims_dict[subset]["vclaims"]
            .set_index("vclaim_id")
            .apply(
                lambda x: f" {model.tokenizer.sep_token} ".join(
                    [x["title"], x["subtitle"], x["vclaim"]]
                ),
                axis=1,
            )
            .to_dict()
        )

    if args.do_train:
        logger.info("Training path %s", str(args.train_data_path))
        train_dataset = read_dataset(
            args.train_data_path,
            args.train_tweets_path,
            vclaims_dict["train"]["vclaims"],
            remove_dates=args.remove_dates,
        )

        ######### Read train data  ##########
        train_samples = []
        for _, row in train_dataset.iterrows():
            train_samples.append(
                InputExample(
                    texts=[
                        row["tweet_text"],
                        f" {model.tokenizer.sep_token} ".join(
                            [row["title"], row["subtitle"], row["vclaim"]]
                        ),
                    ],
                    label=1,
                )
            )
        logger.info("Loaded %d training examples", len(train_samples))
        # After reading the train_samples, we create a DataLoader
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        ################### Development  Evaluators ##################
        evaluators = []

        ###### Classification ######
        dev_sentences1 = []
        dev_sentences2 = []
        dev_labels = []

        ir_queries = {}
        ir_relevant_docs = defaultdict(set)

        dev_dataset = read_dataset(
            args.dev_data_path,
            args.dev_tweets_path,
            vclaims_dict["dev"]["vclaims"],
            remove_dates=args.remove_dates,
            sample_negatives=True,
        )

        ######### Read dev data  ##########
        for _, row in dev_dataset.iterrows():
            dev_sentences1 += [row["tweet_text"]] * 2
            dev_sentences2 += [
                f" {model.tokenizer.sep_token} ".join(
                    [row["title"], row["subtitle"], row["vclaim"]]
                ),
                f" {model.tokenizer.sep_token} ".join(
                    [row["negative_title"], row["negative_subtitle"], row["negative_vclaim"]]
                ),
            ]
            dev_labels += [1, 0]

            ir_queries[row["iclaim_id"]] = row["tweet_text"]
            ir_relevant_docs[row["iclaim_id"]].add(row["vclaim_id"])

        binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(
            dev_sentences1, dev_sentences2, dev_labels
        )
        evaluators.append(binary_acc_evaluator)

        # Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR
        # metrices. For our use case MRR@k and Accuracy@k are relevant.
        ir_evaluator = evaluation.InformationRetrievalEvaluator(
            ir_queries,
            vclaims_dict["dev"]["ir_corpus"],
            ir_relevant_docs,
            batch_size=args.eval_batch_size,
            corpus_chunk_size=args.eval_batch_size,
            mrr_at_k=[len(ir_relevant_docs)],
            accuracy_at_k=[1, 3, 5, 10, 20],
            precision_recall_at_k=[1, 3, 5, 10, 20],
            map_at_k=[1, 3, 5],
            score_functions={"cos_sim": util.cos_sim},
            main_score_function="cos_sim",
            show_progress_bar=True,
        )

        evaluators.append(ir_evaluator)

        # Create a SequentialEvaluator. This SequentialEvaluator runs all three evaluators in a sequential order.
        # We optimize the model with respect to the score from the last evaluator (scores[-1])
        seq_evaluator = evaluation.SequentialEvaluator(
            evaluators, main_score_function=lambda scores: scores[-1]
        )

        # logger.info("Evaluate model without training")
        # seq_evaluator(model, epoch=0, steps=0, output_path=str(model_save_path))

        warmup_steps = int(len(train_dataloader) * args.warmup_proportion * args.num_train_epochs)
        # Train the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=seq_evaluator,
            epochs=args.num_train_epochs,
            warmup_steps=warmup_steps,
            output_path=str(args.model_save_path),
            use_amp=args.fp16,
            evaluation_steps=args.logging_steps,
            optimizer_params={"lr": args.learning_rate, "eps": args.adam_epsilon},
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
        )

        logger.info("Loading the best model from checkpoint.")
        model = SentenceTransformer(str(args.model_save_path), cache_folder=args.cache_dir)

    if args.do_eval:
        logger.info("Loading the testing dataset.")
        ir_queries = {}
        ir_relevant_docs = defaultdict(set)

        test_dataset = read_dataset(
            args.test_data_path,
            args.test_tweets_path,
            vclaims_dict["test"]["vclaims"],
            remove_dates=args.remove_dates,
        )

        ######### Read test data  ##########
        for _, row in test_dataset.iterrows():
            ir_queries[row["iclaim_id"]] = row["tweet_text"]
            ir_relevant_docs[row["iclaim_id"]].add(row["vclaim_id"])

        ir_evaluator_test = evaluation.InformationRetrievalEvaluator(
            ir_queries,
            vclaims_dict["test"]["ir_corpus"],
            ir_relevant_docs,
            batch_size=args.eval_batch_size,
            corpus_chunk_size=args.eval_batch_size,
            mrr_at_k=[len(ir_relevant_docs)],
            accuracy_at_k=[1, 3, 5, 10, 20],
            precision_recall_at_k=[1, 3, 5, 10, 20],
            map_at_k=[1, 3, 5, 10, 20],
            score_functions={"cos_sim": util.cos_sim},
            show_progress_bar=True,
        )
        model.evaluate(
            evaluator=ir_evaluator_test, output_path=str(args.model_save_path / "test_eval")
        )


def inference(args):
    model = SentenceTransformer(args.model_name_or_path, cache_folder=args.cache_dir)
    vclaims_df = load_vclaims(args.vclaims_test_path)
    test_dataset = read_dataset(
        args.test_data_path,
        args.test_tweets_path,
        vclaims_df,
        remove_dates=args.remove_dates,
    )
    ######### Read train data  ##########
    sentences1 = {}
    sentences2 = {}
    for _, row in test_dataset.iterrows():
        sentences1[row["iclaim_id"]] = row["tweet_text"]

    for _, row in vclaims_df.iterrows():
        sentences2[row["vclaim_id"]] = f" {model.tokenizer.sep_token} ".join(
            [row["title"], row["subtitle"], row["vclaim"]]
        )
    sentences = list(set(list(sentences1.values()) + list(sentences2.values())))
    embeddings = model.encode(
        sentences,
        batch_size=args.eval_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
    iclaim_keys = sorted(sentences1.keys())
    vclaim_keys = sorted(sentences2.keys())
    embeddings1 = [emb_dict[sentences1[iclaim_id]] for iclaim_id in iclaim_keys]
    embeddings2 = [emb_dict[sentences2[vclaim_id]] for vclaim_id in vclaim_keys]
    cosine_scores = 1 - cosine_distances(embeddings1, embeddings2)
    output_path = args.model_save_path / "predictions" / "predictions.csv"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    results = []
    vclaims_count = len(vclaim_keys)
    ranks = list(range(1, vclaims_count + 1))
    for row_idx, cosine_score in enumerate(cosine_scores):
        sored_keys = (-cosine_score).argsort()
        results += zip(
            [iclaim_keys[row_idx]] * vclaims_count,
            (vclaim_keys[x] for x in sored_keys),
            ranks,
            cosine_score[sored_keys].astype(float).tolist(),
        )
    results = pd.DataFrame(results, columns=["qid", "docno", "rank", "score"])
    results["Q0"] = "Q0"
    results["tag"] = args.model_name_or_path
    logger.info("Outputting predictions  to %s", output_path)
    results[["qid", "Q0", "docno", "rank", "score", "tag"]].to_csv(
        output_path, index=False, header=False, sep="\t", float_format="%.4f"
    )


def main():
    configure_logging()
    args = parse_args()
    set_seed(args.seed)

    logger.info(
        "Running training with args: \n%s", json.dumps(vars(args), cls=CustomJSONEncoder, indent=2)
    )
    if args.do_train or args.do_eval:
        trainer(args)

    if args.do_predict:
        inference(args)


if __name__ == "__main__":
    main()
