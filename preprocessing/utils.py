"""Utility functions used across multiple modules."""


def validate_columns(meta, required_columns):
    """Validate that required columns are present in metadata DataFrame.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata DataFrame to validate
    required_columns : list[str]
        List of column names that must be present

    Raises
    ------
    ValueError
        If any required columns are missing from the DataFrame
    """
    missing_cols = set(required_columns) - set(meta.columns)
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")
