"""
Is the language model beating anything that deserves to be called a baseline?

An earlier version compared against a regex keyword table. That table fails on
phrasing rather than on meaning, so beating it shows only that the model handles
English, which is a weak claim.

This adds a retrieval baseline that is actually robust to wording: TF-IDF cosine
nearest neighbour over labelled requests, predicting the neighbour's properties.
It is trained on the literal phrasings and tested on the paraphrased ones, which
is the honest split -- if the language model's advantage really is linguistic
robustness, this is the baseline that should close the gap, and if it does not,
that is the result.

Implemented in numpy rather than pulling in scikit-learn: the whole method is
twenty lines and adding a dependency for it would be worse than writing it.

    python baselines.py --benchmark benchmark.json
"""

import argparse
import json
import math
import pathlib
import re
import statistics as st
import sys
from collections import Counter

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

from evaluate import concepts_of, props_of, rule_based_parse  # noqa: E402
from llm import parse  # noqa: E402

TOKEN = re.compile(r"[a-z]+")


def tokenise(t):
    return TOKEN.findall(t.lower())


class TfidfNN:
    """Nearest neighbour over TF-IDF vectors. Predicts the neighbour's labels."""

    def __init__(self, texts, labels):
        self.labels = labels
        docs = [tokenise(t) for t in texts]
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.vocab = {w: i for i, w in enumerate(sorted(df))}
        n = len(docs)
        self.idf = np.zeros(len(self.vocab))
        for w, i in self.vocab.items():
            self.idf[i] = math.log((1 + n) / (1 + df[w])) + 1.0
        self.M = np.stack([self._vec(d) for d in docs])

    def _vec(self, doc):
        v = np.zeros(len(self.vocab))
        for w in doc:
            i = self.vocab.get(w)
            if i is not None:
                v[i] += 1.0
        v *= self.idf
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    def predict(self, text):
        v = self._vec(tokenise(text))
        if not v.any():
            return set()
        sims = self.M @ v
        return set(self.labels[int(np.argmax(sims))])


def _f1(got, want):
    if not want:
        return 1.0 if not got else 0.0
    if not got:
        return 0.0
    tp = len(got & want)
    if not tp:
        return 0.0
    p, r = tp / len(got), tp / len(want)
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="benchmark.json")
    ap.add_argument("--model", default=None, help="override the LLM")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()

    items = json.loads(pathlib.Path(a.benchmark).read_text(encoding="utf-8"))
    train = [i for i in items if i["style"] == "literal"]
    test = [i for i in items if i["style"] == "paraphrased"]
    if a.limit:
        test = test[:a.limit]

    nn = TfidfNN([i["text"] for i in train], [i["props"] for i in train])

    scores = {"keyword table": [], "tf-idf 1-NN": [], "LLM": []}
    cats = {}
    for it in test:
        want = concepts_of(it["props"])

        scores["keyword table"].append(
            _f1(concepts_of(props_of(rule_based_parse(it["text"]))), want))
        scores["tf-idf 1-NN"].append(
            _f1(concepts_of(nn.predict(it["text"])), want))
        if not a.no_llm:
            kw = {"model": a.model} if a.model else {}
            scores["LLM"].append(
                _f1(concepts_of(props_of(parse(it["text"], **kw))), want))

        cats.setdefault(it["cat"], []).append(len(scores["keyword table"]) - 1)

    print(f"trained on {len(train)} literal requests, tested on {len(test)} "
          f"paraphrased ones\n")
    print(f"{'system':16s}{'concept f1':>12s}")
    print("-" * 28)
    for k, v in scores.items():
        if v:
            print(f"{k:16s}{st.mean(v):12.2f}")

    print(f"\n{'category':16s}" + "".join(f"{k:>16s}" for k in scores if scores[k]))
    print("-" * (16 + 16 * sum(1 for k in scores if scores[k])))
    for c, idx in sorted(cats.items()):
        line = f"{c:16s}"
        for k, v in scores.items():
            if v:
                line += f"{st.mean(v[i] for i in idx):16.2f}"
        print(line)

    out = {k: v for k, v in scores.items() if v}
    pathlib.Path("baseline_results.json").write_text(json.dumps(out, indent=1),
                                                     encoding="utf-8")
    print("\nwrote baseline_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
