"""
Post-processing and plotting utilities, as well as string mappings,
for the 01_experiments.py and succeeding notebooks.
"""

import re
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.pyplot as plt
from typing import Optional

# OCR names for display
ocr_strs = {
    "tesseract": "Tesseract",
    "google_ocr": "Google Cloud Vision",
    "azure": "Azure AI Vision",
    "textract": "Amazon Textract",
    "None": "No OCR",
}
ocr_strs_short = {
    "tesseract": "Tesseract",
    "google_ocr": "Google",
    "azure": "Azure",
    "textract": "Textract",
    "None": "None/All",
}
ocr_engines = list(ocr_strs.keys())

# Strategies for display
method_strs = {
    "None": "\\textsc{ocr only}",
    "all_pages": "\\textsc{+all pages}",
    "chosen_page": "\\textsc{+chosen page}",
    "first_page": "\\textsc{+first page}",
    "pbp": "\\textsc{ocr only pbp}",
    "vision*": "\\textsc{vision*}",
    "vision*_pbp": "\\textsc{vision* pbp}",
    "google_ocr_azure_textract_pbp": "\\textsc{all ocr pbp}",
    "all_pages_pbp": "\\textsc{+all pages pbp}",
}

gpt_model_strs = lambda s: f"\\textsc{{{s}}}"


def gpt_output_postprocessing(s: str) -> str:
    """
    General purpose GPT output string postprocessing.
    - Convert escaped newlines to their actual values
    - Replace multiple newlines with single newline, and multiple spaces with single space
    - Remove whitespace before newlines, e.g. "\n[NEW_PAGE]  \n" -> "\n[NEW_PAGE]\n"
    - Optionally remove all whitespace, as whitespace variations are not that
        important and GT labels sometimes needlessly include them
    """
    s = s.replace("\\n", "\n")
    while s != (s := s.replace("\n\n", "\n")):
        pass
    while s != (s := s.replace("  ", " ")):
        pass
    s = re.sub(r"\s+\n", "\n", s)
    return s


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate columns from a DataFrame based on both column names and contents.
    Only the first occurrence of each duplicate column is kept.

    Parameters:
    df (pd.DataFrame): The input DataFrame with potential duplicate columns.

    Returns:
    pd.DataFrame: A DataFrame with duplicate columns removed.
    """
    seen = set()
    cols_to_keep = []

    for idx, col in enumerate(df.columns):
        # Function to recursively convert lists (and nested lists) to tuples
        def make_hashable(x):
            if isinstance(x, list):
                return tuple(make_hashable(item) for item in x)
            elif isinstance(x, dict):
                # Convert dictionaries to sorted tuples of key-value pairs
                return tuple(sorted((k, make_hashable(v)) for k, v in x.items()))
            else:
                return x

        # Apply make_hashable to each element in the column
        try:
            contents = tuple(make_hashable(x) for x in df[col])
        except TypeError as e:
            raise ValueError(f"Unhashable type encountered in column '{col}': {e}")

        # Create a unique signature combining column name and contents
        signature = (col, contents)

        if signature not in seen:
            seen.add(signature)
            cols_to_keep.append(idx)
        else:
            # Duplicate found; skip adding this column
            pass

    # Select columns by their positional indices to handle duplicate names correctly
    df_unique = df.iloc[:, cols_to_keep].copy()
    return df_unique


def get_mode_elements(mode: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Get OCR engine, method, and GPT model from mode string."""
    if mode in ocr_engines:
        ocr_engine, method, gpt_model = mode, "None", "None"
    elif "->" in mode:
        method, gpt_model = mode.split("->")
        if "+" in method:
            ocr_engine, method = method.split("+")
            assert ocr_engine in ocr_engines
        elif method in ocr_engines:
            ocr_engine, method = method, "None"
        else:
            ocr_engine, method = "None", method
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return ocr_engine, method, gpt_model


def get_mode_str(ocr_engine: str, method: str, gpt_model: str) -> str:
    """Generate a string representation of the mode."""
    out = f"{ocr_engine}+{method}->{gpt_model}"
    out = out.replace("None+", "").replace("+None", "").replace("->None", "")
    return out


def generate_colored_latex_table(
    df: pd.DataFrame,
    metric_col: str,
    improvement_cols: dict,
    label: str,
    caption: str,
    cmap_name_positive="Greens",
    cmap_name_negative="Reds",
):
    """
    Generate a LaTeX table with color-coded cells for each improvement column and
    format the metric column to bold the lowest value and italicize the second lowest.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        metric_col (str): The column name for the metric.
        improvement_cols (dict): A dictionary mapping improvement column names to LaTeX labels.
        label (str): The label for the table.
        caption (str): The caption for the table.
        cmap_name_positive (str): The colormap for positive values (default: "Greens").
        cmap_name_negative (str): The colormap for negative values (default: "Reds").

    Returns:
        str: The LaTeX table as a string.
    """

    # Helper function to create a custom colormap with white at zero
    def create_custom_colormap(cmap_name_positive, cmap_name_negative):
        colors = {
            "red": plt.cm.get_cmap(cmap_name_negative)(0.5),
            "white": (1, 1, 1, 1),  # Pure white
            "green": plt.cm.get_cmap(cmap_name_positive)(0.5),
        }
        return LinearSegmentedColormap.from_list(
            "custom_colormap", [colors["red"], colors["white"], colors["green"]]
        )

    # Custom colormap
    custom_cmap = create_custom_colormap(cmap_name_positive, cmap_name_negative)

    # Helper function to apply color coding for a single column
    def colorize_column(df, col):
        max_abs_value = max(abs(df[col].min()), abs(df[col].max()))
        norm = TwoSlopeNorm(vmin=-max_abs_value, vcenter=0, vmax=max_abs_value)

        def colorize(value):
            rgba = custom_cmap(norm(value))
            rgb = tuple(int(x * 255) for x in rgba[:3])  # Convert to 0-255 RGB scale
            return f"\\cellcolor[RGB]{{{rgb[0]},{rgb[1]},{rgb[2]}}}{value:.2f}"

        return df[col].apply(colorize)

    # Apply colorization to each improvement column
    for improvement_col in improvement_cols.keys():
        df[f"{improvement_col}_colored"] = colorize_column(df, improvement_col)

    # **Step 1: Identify Unique Sorted Metric Values**
    # Round the metric column numerically without converting to string
    df["_rounded_metric"] = df[metric_col].round(3)

    # Get sorted unique metric values
    sorted_unique_metrics = sorted(df["_rounded_metric"].unique())

    # Assign bold and italic values
    bold_value = sorted_unique_metrics[0] if len(sorted_unique_metrics) >= 1 else None
    italic_value = sorted_unique_metrics[1] if len(sorted_unique_metrics) >= 2 else None

    # **Step 2: Format the Metric Column Based on Rank**
    def format_metric(row):
        value = row["_rounded_metric"]
        if value == bold_value:
            return f"\\textbf{{{value:.3f}}}"
        elif value == italic_value:
            return f"\\textit{{{value:.3f}}}"
        else:
            return f"{value:.3f}"

    df[metric_col] = df.apply(format_metric, axis=1)

    # Drop the temporary rounded metric column
    df.drop(columns=["_rounded_metric"], inplace=True)

    # Define the column alignment for LaTeX (adjust as needed for the number of columns)
    col_alignment = "rr" + "c" * (
        1 + len(improvement_cols)
    )  # First two columns right-aligned, rest centered

    # If you have a mapping for method strings, ensure it's defined
    # Example:
    # method_strs = {"method1": "Method One", "method2": "Method Two"}
    # Make sure to define or pass `method_strs` accordingly
    if "method_strs" in globals():
        df["method"] = df["method"].apply(
            lambda x: method_strs.get(x, x)
        )  # Replace with display names
    else:
        # If method_strs is not defined, skip or handle accordingly
        pass

    df["gpt_model"] = df["gpt_model"].apply(gpt_model_strs)

    # Generate the LaTeX table without the header
    latex_table = df[
        ["method", "gpt_model", metric_col]
        + [f"{col}_colored" for col in improvement_cols.keys()]
    ].to_latex(
        index=False,
        escape=False,  # Allow LaTeX commands in cells
        column_format=col_alignment,  # Adjust alignment dynamically
        longtable=False,  # Use regular table
        header=False,
    )

    # Create column headers dynamically
    headers = (
        f"Method & $\\rightarrow$ MLLM & {metric_col} & "
        + " & ".join(improvement_cols.values())
        + " \\\\\n"
    )

    # Add booktabs formatting
    latex_table = (
        (
            "\\begin{table}[H]\n"
            "\\centering\n"
            f"\\begin{{tabular}}{{@{{}}{col_alignment}@{{}}}}\n"  # Booktabs format with alignment
            "\\toprule\n"
            + headers
            + "\\midrule\n"
            + latex_table.split("\\toprule")[1]
            .split("\\bottomrule")[0]
            .strip()  # Remove extra newline
            + "\\bottomrule\n"
            "\\end{tabular}\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{tab:{label}}}\n"
            "\\end{table}"
        )
        # .replace("_", "\_")
        .replace("None", "-")
    )

    return latex_table
