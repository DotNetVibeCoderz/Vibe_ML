"""Writes data pickles of the kind Geneformer publishes: dicts of strings to NumPy scalars.

    python make_pickles.py

numpy-scalars-p2.pkl, -p4.pkl, -p5.pkl   one dict, NumPy scalars of every supported kind, in three protocols
                                          (2 stores the raw bytes through _codecs.encode, 3+ as bytes)
token-dictionary.pkl                      Geneformer's token dictionary shape: str -> plain int
not-data.pkl                              a datetime.date, which needs a global outside the allow-list

Every value is exact in its dtype, so the C# tests assert equality, and the expected values are written
out by hand in PickleTests rather than read back from here.
"""
import datetime, os, pickle

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

scalars = {
    "ENSG00000000003": np.float32(1.5),
    "ENSG00000000005": np.float64(-2.25),
    "half": np.float16(0.5),
    "count": np.int64(7),
    "small": np.int8(-3),
    "u16": np.uint16(65535),
    "big": np.uint64(2**63 + 5),
    "flag": np.bool_(True),
    "big_endian": np.array([3.5], dtype=">f8")[0],
    "name": np.str_("abc"),
    "plain": 0.125,
}

for protocol in (2, 4, 5):
    with open(os.path.join(here, f"numpy-scalars-p{protocol}.pkl"), "wb") as f:
        pickle.dump(scalars, f, protocol=protocol)

with open(os.path.join(here, "token-dictionary.pkl"), "wb") as f:
    pickle.dump({"<pad>": 0, "<mask>": 1, "ENSG00000000003": 2, "ENSG00000000005": 3}, f)

with open(os.path.join(here, "not-data.pkl"), "wb") as f:
    pickle.dump({"when": datetime.date(2026, 10, 15)}, f)

print("numpy", np.__version__, "->", sorted(os.listdir(here)))
