import json
from glob import glob

import numpy as np
import pandas as pd


# Check if the dataset exists. If not, download and extract
def load_vclaims(dir):
    vclaims = {}
    for vclaim_fp in sorted(glob(f"{dir}/*.json")):
        with open(vclaim_fp) as f:
            vclaim = json.load(f)
        vclaims[vclaim["vclaim_id"]] = vclaim

    return pd.DataFrame.from_dict(vclaims, orient="index").reset_index(drop=True)


def read_dataset(data_fpath, tweets_fpath, vclaims_df, remove_dates=False, sample_negatives=False):
    gold_df = pd.read_csv(data_fpath, sep="\t", names=["iclaim_id", "0", "vclaim_id", "relevance"])
    gold_df = gold_df.merge(vclaims_df, on="vclaim_id").merge(
        pd.read_csv(tweets_fpath, sep="\t", names=["iclaim_id", "tweet_text"]), on="iclaim_id"
    )

    if remove_dates:
        gold_df["tweet_text"] = gold_df["tweet_text"].str.replace(
            r"\s\w+\s\d{1,2},\s2\d{3}$", "", regex=True
        )

    if sample_negatives:
        vtitles = set(gold_df["title"].unique().tolist())
        vsubtitles = set(gold_df["subtitle"].unique().tolist())
        vclaims = set(gold_df["vclaim"].unique().tolist())
        gold_df["negative_title"] = gold_df["title"].apply(
            lambda x: np.random.choice(list(vtitles - {x}), 1)[0]
        )
        gold_df["negative_subtitle"] = gold_df["subtitle"].apply(
            lambda x: np.random.choice(list(vsubtitles - {x}), 1)[0]
        )
        gold_df["negative_vclaim"] = gold_df["vclaim"].apply(
            lambda x: np.random.choice(list(vclaims - {x}), 1)[0]
        )

    return gold_df
