import re
import pandas as pd

SRC = "ab-ru-parallel.csv"
OUT_DIR = "data"
DEV_SIZE = 1500
SEED = 42

CYRILLIC = re.compile(r"[Ѐ-ӿԀ-ԯⷠ-ⷿꙀ-ꚟ]")


def main():
    df = pd.read_csv(SRC)
    n0 = len(df)
    df = df.dropna()
    df["ru"] = df["ru"].astype(str).str.strip()
    df["ab"] = df["ab"].astype(str).str.strip()

    df = df[(df["ru"].str.len() >= 2) & (df["ab"].str.len() >= 2)]
    df = df[df["ru"] != df["ab"]]

    lr = df["ru"].str.len()
    la = df["ab"].str.len()
    ratio = la / lr.clip(lower=1)
    df = df[(ratio >= 0.4) & (ratio <= 2.6)]

    df = df[df["ab"].apply(lambda s: bool(CYRILLIC.search(s)))]

    df = df.drop_duplicates(subset=["ru", "ab"])
    df = df.drop_duplicates(subset=["ru"])
    df = df.drop_duplicates(subset=["ab"])

    df = df[["ru", "ab"]].reset_index(drop=True)
    n1 = len(df)

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    dev = df.iloc[:DEV_SIZE]
    train = df.iloc[DEV_SIZE:]

    df.to_csv(f"{OUT_DIR}/clean_all.tsv", sep="\t", index=False)
    train.to_csv(f"{OUT_DIR}/train.tsv", sep="\t", index=False)
    dev.to_csv(f"{OUT_DIR}/dev.tsv", sep="\t", index=False)

    print(f"raw rows:     {n0}")
    print(f"after clean:  {n1}  (dropped {n0 - n1})")
    print(f"train:        {len(train)}")
    print(f"dev:          {len(dev)}")


if __name__ == "__main__":
    main()
