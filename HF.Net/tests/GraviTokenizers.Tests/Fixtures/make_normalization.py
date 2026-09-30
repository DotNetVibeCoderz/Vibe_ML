"""Writes normalization.json: strings and their four Unicode normalization forms, from Python's unicodedata."""
import json, random, unicodedata

random.seed(3)
cases = [
    "é", "café", "café", "Ångström", "Å", "Å",
    "q̣̇", "q̣̇", "ḍ̇", "ḍ̇", "̈́", "क़",
    "한국어", "각", "ﬁle", "①②", "ＡＢＣ", "½", "x²",
    "㌀", "Ω", "ά", "΅", " ", "\U0001d400", "à́̂", "̀a", "plain ascii",
]
pool = [c for c in range(0x80, 0x3000) if unicodedata.category(chr(c))[0] in "LMN"] + list(range(0xAC00, 0xAC00 + 300))
for _ in range(120):
    cases.append("".join(chr(random.choice(pool)) for _ in range(random.randint(1, 6))))
rows = [{"text": t, "NFC": unicodedata.normalize("NFC", t), "NFD": unicodedata.normalize("NFD", t),
         "NFKC": unicodedata.normalize("NFKC", t), "NFKD": unicodedata.normalize("NFKD", t)} for t in cases]
json.dump({"unicode": unicodedata.unidata_version, "cases": rows}, open("normalization.json", "w", encoding="utf-8"), ensure_ascii=True)
print(len(rows))
