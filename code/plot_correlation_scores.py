import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple


try:
    from pyensembl import EnsemblRelease
except ImportError:
    pass

plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42, "font.family": "sans-serif"})


def get_regions_to_annotate(
    coherence_df: pd.DataFrame, annotation_coherence_threshold: float, spacing: int
) -> List[Tuple[float, float]]:
    """
    Identify regions to annotate based on coherence scores and spacing.

    Args:
        coherence_df (pd.DataFrame): DataFrame containing methylation coherence scores and window information.
        annotation_coherence_threshold (float): Minimum methylation coherence score for annotation.
        spacing (int): Minimum spacing between regions to be considered separate.

    Returns:
        List[Tuple[float, float]]: List of start and end positions of regions to annotate.
    """
    regions_to_annotate = coherence_df[
        coherence_df["Coherence_Score"] > annotation_coherence_threshold
    ]
    window_positions = np.sort(
        np.concatenate(
            [
                regions_to_annotate["Window_Start"],
                regions_to_annotate["Window_End"] + regions_to_annotate["Window_Size"],
            ]
        )
    )
    window_diffs = np.diff(window_positions)

    position_breaks = np.where(window_diffs > spacing)[0]
    position_breaks = np.insert(position_breaks, 0, -1)
    position_breaks = np.append(position_breaks, len(window_positions) - 1)

    regions_to_annotate = []
    for i in range(0, len(position_breaks) - 1):
        start = window_positions[position_breaks[i] + 1]
        end = window_positions[position_breaks[i + 1]]
        regions_to_annotate.append((start, end))
    return regions_to_annotate


def get_gene_db() -> EnsemblRelease:
    """
    Get and prepare the Ensembl gene database.

    Returns:
        EnsemblRelease:  Ensembl gene database.
    """
    data = EnsemblRelease(110)
    data.download()
    data.index()
    return data


def get_annotations(
    coherence_df: pd.DataFrame, annotation_coherence_threshold: float, spacing: int
) -> List[Tuple[Tuple[float, float], List[str]]]:
    """
    Get gene annotations for regions of interest.

    Args:
        coherence_df (pd.DataFrame): DataFrame containing coherence scores and window information.
        annotation_coherence_threshold (float): Minimum coherence score for annotation.
        spacing (int): Minimum spacing between regions to be considered separate.

    Returns:
        List[Tuple[Tuple[float, float], List[str]]]: List of tuples containing region coordinates and associated gene names.
    """
    regions_to_annotate = get_regions_to_annotate(
        coherence_df, annotation_coherence_threshold, spacing
    )
    gene_db = get_gene_db()
    annotations = []
    for region in regions_to_annotate:
        genes = gene_db.gene_names_at_locus(
            contig="7", position=int(region[0]), end=int(region[1])
        )
        if genes:
            annotations.append((region, genes))
    return annotations


def plot_coherence_scores(
    coherence_df: pd.DataFrame,
    annotations: List[Tuple[Tuple[float, float], List[str]]] = None,
    plot_output: str = None,
):
    """
    Plot coherence scores and optionally add gene annotations.

    Args:
        coherence_df (pd.DataFrame): DataFrame containing coherence scores and window information.
        annotations (List[Tuple[Tuple[float, float], List[str]]], optional): List of annotations to add to the plot.
        plot_output (str, optional): Output file for the plot.
    """
    chromosome = coherence_df["Chromosome"].iloc[0].replace("chr", "")
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.scatter(
        (coherence_df["Window_Start"] + coherence_df["Window_End"]) / (2 * 1e6),
        coherence_df["Coherence_Score"],
        s=1,
        color="blue",
    )
    ax.set_xlabel("Genome position (Mb)")
    ax.set_ylabel("Coherence score")
    ax.set_title(f"Methylation coherence scores in variable regions\n Chromosome {chromosome}")

    if annotations:
        for annotation in annotations:
            region = annotation[0]
            genes = annotation[1]
            intersects_region = (coherence_df["Window_Start"] < region[1]) & (
                coherence_df["Window_End"] > region[0]
            )
            region_height = coherence_df[intersects_region]["Coherence_Score"].max()
            ax.text(
                (region[0] + region[1]) / (2 * 1e6),
                region_height + 0.12,
                "\n".join(genes),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=10,
            )

    plt.tight_layout()

    if plot_output:
        plt.savefig(plot_output, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the coherence scores for variable methylation regions"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input tsv file containing the methylation coherence scores for a single chromosome.",
        required=True,
    )
    parser.add_argument(
        "--add_annotations",
        action=argparse.BooleanOptionalAction,
        help="Add gene annotations to the plot",
        required=False,
    )
    parser.add_argument(
        "--annotation_coherence_threshold",
        type=float,
        help="Minimum methylation coherence score for gene annotation",
        required=False,
        default=4.0,
    )
    parser.add_argument(
        "--region_spacing",
        type=int,
        help="Minimum spacing between individual regions to be considered separate",
        required=False,
        default=20000,
    )
    parser.add_argument(
        "--plot_output",
        type=str,
        help="Output filepath for the plot",
        required=False,
        default=None,
    )
    args = parser.parse_args()

    if args.add_annotations and "pyensembl" not in sys.modules:
        raise ImportError(
            "The pyensembl module is required for gene annotations. Please install it using 'pip install pyensembl'.")
    coherence_df = pd.read_csv(args.input_file, sep="\t")

    if args.add_annotations:
        annotations = get_annotations(
            coherence_df, args.annotation_coherence_threshold, args.region_spacing
        )
    else:
        annotations = None

    plot_coherence_scores(coherence_df, annotations, args.plot_output)
