import numpy as np


def make_counts_long(counts, clinical, plate_group):
    long_counts = counts.merge(clinical[plate_group], left_index=True, right_index=True)

    long_counts.index.name = "index"

    long_counts = (
        long_counts.reset_index()
        .melt(id_vars=["index", plate_group], value_name="ct", var_name="gene")
        .rename(columns={"index": "sample"})
    )
    return long_counts


def rle_norm(
    df_long,
    response="ct",
    un_norm=None,  # microplate to skip normalization,
    gene_col="mut",
    sample_col="microplate",
    plate_group="shortplate",
    gene_filter_stringency="loose",  # loose or strict
    log_func=np.log,
):

    if un_norm is None:
        un_norm = []

    # Calculate scaling factors
    _df = df_long.copy()
    _df["ct_log"] = _df[response].map(log_func)
    _df = _df.dropna(subset=["ct_log"])
    _df["ct_log"] = _df["ct_log"].replace(-np.inf, np.nan)
    # filter out bad (-inf) genes.
    if gene_filter_stringency == "strict":
        bad_genes = _df[_df["ct_log"].isnull()][gene_col].unique()
        print("Genes removed:", len(bad_genes))
        _df = _df[~_df[gene_col].isin(bad_genes)]
    else:
        print(f"Entries removed: {_df['ct_log'].isnull().sum()}")
        _df = _df.dropna(subset="ct_log")

    _df["log_mean"] = _df.groupby([gene_col])["ct_log"].transform("mean")
    _df["pseudoref"] = _df["ct_log"] - _df["log_mean"]
    _df["median_ratio"] = _df.groupby([sample_col])["pseudoref"].transform("median")
    _df["scaling_factor"] = np.exp(_df["median_ratio"])

    # get per microplate scaling factor
    scaling_factors = _df.groupby([sample_col])[["scaling_factor", plate_group]].mean()
    scaling_factors[plate_group] = scaling_factors[plate_group].astype(int)
    for i in un_norm:
        scaling_factors.loc[i, "scaling_factor"] = 1
    scaling_factors = scaling_factors[["scaling_factor"]]

    # Apply changes
    out_df_long = df_long.copy()
    _get_ct_rle = df_long[[response, sample_col]].merge(scaling_factors.reset_index())
    out_df_long[f"{response}_rle"] = (
        _get_ct_rle[response] / _get_ct_rle["scaling_factor"]
    )

    return out_df_long, scaling_factors


def deseq2_norm(counts):
    """
    Return normalized counts and size_factors.

    Uses the median of ratios method.

    Parameters
    ----------
    counts : pandas.DataFrame
            Raw counts. One column per gene, rows are indexed by sample barcodes.

    Returns
    -------
    deseq2_counts : pandas.DataFrame
        DESeq2 normalized counts.
        One column per gene, rows are indexed by sample barcodes.

    size_factors : pandas.DataFrame
        DESeq2 normalization factors.
    """
    # Compute gene-wise mean log counts
    log_counts = counts.apply(np.log)
    logmeans = log_counts.mean(0)
    # Filter out genes with -∞ log means
    filtered_genes = ~np.isinf(logmeans).values
    # Subtract filtered log means from log counts
    log_ratios = log_counts.iloc[:, filtered_genes] - logmeans[filtered_genes]
    # Compute sample-wise median of log ratios
    log_medians = log_ratios.median(1)
    # Return raw counts divided by size factors (exponential of log ratios)
    # and size factors
    size_factors = np.exp(log_medians)
    deseq2_counts = counts.div(size_factors, 0)
    return deseq2_counts, size_factors
