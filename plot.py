from sys import argv

import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt


def one_tailed_paired_tests(data1, data2, alternative="greater"):
    """
    Performs one-tailed paired t-test and Wilcoxon signed-rank test.

    Args:
        data1: List-like containing data for the first variable.
        data2: List-like containing data for the second variable.
        alternative: "greater" or "less" depending on the direction of interest.

    Returns:
        A dictionary containing p-values and test statistics for both tests.
    """
    # Check if data lengths are equal
    if len(data1) != len(data2):
        raise ValueError("Data lists must have the same length")

    # Calculate differences
    differences = data1 - data2

    # Paired t-test (one-tailed)
    t_statistic, t_pval = stats.ttest_rel(data1, data2, alternative=alternative)

    # Wilcoxon signed-rank test (one-tailed)
    z_statistic, w_pval = stats.wilcoxon(differences, alternative=alternative)

    # Return results
    return {
        "t_statistic": t_statistic,
        "t_pval": t_pval,
        "z_statistic": z_statistic,
        "w_pval": w_pval,
    }


mult_dict = {
    "crispr": "outputs/crispr_allmodalities_largebatch_updatedpipeline_newbatch/metrics/profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony_PCA_corrected",
    "compound": "outputs/compound_allmodalities_largebatch/metrics/profiles_var_mad_int_featselect_harmony",
    "orf": "outputs/orf_allmodalities_largebatch/metrics/profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony",
}
uniq_dict = {
    "orf": "outputs/orf/metrics/profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony",
    "crispr": "outputs/crispr/metrics/profiles_wellpos_var_mad_int_featselect",
    "compound": "outputs/compound/metrics/profiles_var_mad_int_featselect_harmony",
}
mult_name, uniq_name = argv[1:3]
mult_prefix = mult_dict[mult_name]
uniq_prefix = uniq_dict[uniq_name]

mult_ap = pd.read_parquet(f"{mult_prefix}_ap_negcon.parquet")
mult_map = pd.read_parquet(f"{mult_prefix}_map_negcon.parquet")

uniq_ap = pd.read_parquet(f"{uniq_prefix }_ap_negcon.parquet")
uniq_map = pd.read_parquet(f"{uniq_prefix }_map_negcon.parquet")

mult_map["Metadata_JCP2022"] = mult_map["Metadata_JCP2022"].astype(str)
uniq_map["Metadata_JCP2022"] = uniq_map["Metadata_JCP2022"].astype(str)

mult_map = mult_map.query("Metadata_JCP2022 in @uniq_map['Metadata_JCP2022']")
uniq = uniq_map.merge(mult_map, on="Metadata_JCP2022", suffixes=("_uniq", "_mult"))


g = sns.jointplot(
    uniq.sample(frac=1.0),
    x="mean_average_precision_mult",
    y="mean_average_precision_uniq",
    kind="hex",
    xlim=(-0.01, 1.01),
    ylim=(-0.01, 1.01),
    bins="log",
    # hue="below_corrected_p_uniq",
    # style="below_corrected_p_mult",
)
g.fig.suptitle(f"uniq={uniq_name}, mult={mult_name}, n={len(uniq)}")
# ax.set_xlim(-0.01, 1.01)
# ax.set_ylim(-0.01, 1.01)
# ax.set_title(f"uniq={uniq_name}, mult={mult_name}, n={len(uniq)}")
# ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.savefig(f"{mult_name}_all_{uniq_name}_map.pdf", bbox_inches="tight")
plt.close("all")

ax = sns.histplot(
    uniq[["mean_average_precision_mult", "mean_average_precision_uniq"]],
)
ax.set_title(f"uniq={uniq_name}, mult={mult_name}, n={len(uniq)}")
plt.savefig(f"{mult_name}_all_{uniq_name}_hist.pdf", bbox_inches="tight")
plt.close("all")

ax = sns.violinplot(
    uniq[["mean_average_precision_mult", "mean_average_precision_uniq"]],
)
ax.set_title(f"uniq={uniq_name}, mult={mult_name}, n={len(uniq)}")
plt.savefig(f"{mult_name}_all_{uniq_name}_violin.pdf", bbox_inches="tight")
plt.close("all")


results = one_tailed_paired_tests(
    uniq["mean_average_precision_uniq"], uniq["mean_average_precision_mult"]
)
print(f"T-test statistic: {results['t_statistic']}")
print(f"T-test p-value (one-tailed): {results['t_pval']}")
print(f"Wilcoxon signed-rank statistic: {results['z_statistic']}")
print(f"Wilcoxon signed-rank p-value (one-tailed): {results['w_pval']}")
