"""Prints C# literals for ResamplingMethodTests: scipy 1.17 p-values with PermutationMethod in cases
small enough that the permutation distribution is enumerated exactly (so the p-value is deterministic).

Run:  python gen_resampling_methods.py   and paste the output into ResamplingMethodTests.cs.
"""
import numpy as np
from scipy import stats

x = np.array([1.2, -0.4, 2.2, 0.9, 3.1, -1.0])
y = np.array([0.7, -0.1, 1.9, 1.5, 2.0, 0.2])
a = np.array([4.1, 5.3, 2.2, 6.0, 3.9])
b = np.array([6.6, 7.1, 5.2, 8.0])
pm = stats.PermutationMethod()


def show(name, res):
    print(f"// {name}: statistic {float(res.statistic)!r}, pvalue {float(res.pvalue)!r}")


for alt in ("two-sided", "less", "greater"):
    show(f"pearsonr {alt}", stats.pearsonr(x, y, method=pm, alternative=alt))
show("spearmanrho", stats.spearmanrho(x, y, method=pm))
for alt in ("two-sided", "greater"):
    show(f"ttest_ind equal_var {alt}", stats.ttest_ind(a, b, method=pm, alternative=alt))
show("ttest_ind welch", stats.ttest_ind(a, b, equal_var=False, method=pm))
show("mannwhitneyu", stats.mannwhitneyu(a, b, method=pm))
show("mannwhitneyu less", stats.mannwhitneyu(a, b, method=pm, alternative="less"))
d = np.array([0.5, -1.25, 2.0, 0.75, -0.25, 1.5, 3.0, 0.0])
show("wilcoxon pratt", stats.wilcoxon(d, method=pm, zero_method="pratt"))
show("wilcoxon wilcox greater", stats.wilcoxon(d[d != 0], method=pm, alternative="greater"))
show("chatterjeexi", stats.chatterjeexi(x, y, method=pm))
r = stats.anderson_ksamp([a, b, np.array([3.0, 4.5])], method=pm)
print(f"// anderson_ksamp: statistic {float(r.statistic)!r}, pvalue {float(r.pvalue)!r}")
obs = np.array([[3, 1], [1, 2]])
r = stats.chi2_contingency(obs, correction=False, method=pm)
print(f"// chi2_contingency 2x2: statistic {float(r.statistic)!r}, pvalue {float(r.pvalue)!r}")
obs = np.array([[2, 0, 1], [0, 2, 1]])
r = stats.chi2_contingency(obs, correction=False, method=pm)
print(f"// chi2_contingency 2x3: statistic {float(r.statistic)!r}, pvalue {float(r.pvalue)!r}")
r = stats.bws_test(a, b, method=pm)
print(f"// bws_test: statistic {float(r.statistic)!r}, pvalue {float(r.pvalue)!r}")
