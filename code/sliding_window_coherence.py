import argparse
import multiprocessing as mp

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, replace

import pandas as pd
import numpy as np

import tqdm

@dataclass
class Window:
    """
    Represents a fixed window of CpG positions.

    Attributes:
        left_edge (int): Left edge of the window.
        left_cpg_index (int): Index of the leftmost CpG in the window.
        right_cpg_index (int): Index of the rightmost CpG (exclusive).
        window_size (int): Size of the window.
    """

    left_edge: int
    left_cpg_index: int
    right_cpg_index: int
    window_size: int


@dataclass
class SlidingWindow:
    """
    Represents a sliding window of CpG positions.
    The window is defined by its size and the minimum and maximum left edges for the CpGs it contains.

    Attributes:
        start_left_edge (int): Left edge of the window at the start.
        end_left_edge (int): Left edge of the window at the end.
        left_cpg_index (int): Index of the leftmost CpG in the window.
        right_cpg_index (int): Index of the rightmost CpG (exclusive).
        window_size (int): Size of the window.
    """

    start_left_edge: int
    end_left_edge: int
    left_cpg_index: int
    right_cpg_index: int
    window_size: int


def get_variable_cpg_positions(
    read_methylation_data: pd.DataFrame, min_coverage: int = 10, min_alt_reads: int = 3
) -> pd.DataFrame:
    """
    Identify variable CpG positions based on total coverage and number of methylated and non-methylated reads.

    Args:
        read_methylation_data (pd.DataFrame): Input DataFrame containing read data.
        min_coverage (int): Minimum coverage required for a variable CpG position.
        min_alt_reads (int): Minimum number of methylated and non-methylated reads required for a variable CpG position.

    Returns:
        pd.DataFrame: DataFrame containing variable CpG positions.
    """
    cpg_position_df = (
        read_methylation_data.groupby(["Chromosome", "Position"])["Is_Methylated"]
        .sum()
        .reset_index()
    )
    cpg_position_df = cpg_position_df.rename(columns={"Is_Methylated": "Methylated_Counts"})

    cpg_position_df_size = (
        read_methylation_data.groupby(["Chromosome", "Position"])
        .size()
        .reset_index()
        .rename(columns={0: "Coverage"})
    )
    cpg_position_df = cpg_position_df.merge(cpg_position_df_size, how="inner")

    coverage_filter = cpg_position_df["Coverage"] >= min_coverage
    methylated_filter = cpg_position_df["Methylated_Counts"] >= min_alt_reads
    non_methylated_filter = cpg_position_df["Methylated_Counts"] < (
        cpg_position_df["Coverage"] - min_alt_reads
    )

    position_filter = coverage_filter & methylated_filter & non_methylated_filter
    variable_cpg_df = cpg_position_df[position_filter]

    return variable_cpg_df[["Chromosome", "Position"]]


def get_distances_to_next_cpg(cpg_positions: np.ndarray, window: Window) -> Tuple[int, int]:
    """
    Calculate distances to the next CpG position from the current window.

    Args:
        cpg_positions (np.ndarray): Sorted array of CpG positions.
        window (Window): Current CpG window.

    Returns:
        Tuple[int, int]: Distances to the next left and right CpG positions.
    """
    distance_to_next_left_cpg = cpg_positions[window.left_cpg_index]+1 - window.left_edge

    if window.right_cpg_index == cpg_positions.size:
        distance_to_next_right_cpg = np.inf
    else:
        distance_to_next_right_cpg = cpg_positions[window.right_cpg_index]+1 - window.left_edge-window.window_size

    return distance_to_next_left_cpg, distance_to_next_right_cpg


def get_sliding_window_store(
    cpg_positions: np.ndarray, window_size: int, min_cpgs_per_window: int = 0
) -> List[SlidingWindow]:
    """
    Generate a list of sliding windows based on CpG positions.

    Args:
        cpg_positions (np.ndarray): Array of sorted CpG positions.
        window_size (int): Size of each window.
        min_cpgs_per_window (int): Minimum number of CpGs required for each window.

    Returns:
        List[SlidingWindow]: List of sliding window objects.
    """
    cpg_positions = np.sort(np.unique(cpg_positions))
    # Get position of first window
    window_left_edge = cpg_positions[0] - window_size + 1
    left_cpg_index = 0
    # Find the first index where the cpg position is greater than the end of the window
    right_cpg_index = np.argmax(
        np.where(
            cpg_positions - window_left_edge - window_size > 0,
            -np.inf,
            cpg_positions - window_left_edge - window_size,
        )
    )
    current_window = Window(window_left_edge, left_cpg_index, right_cpg_index, window_size)

    sliding_windows = []
    while current_window.left_cpg_index < cpg_positions.size:

        distance_to_next_left_cpg, distance_to_next_right_cpg = get_distances_to_next_cpg(
            cpg_positions, current_window
        )

        # increment both left and right cpg index if they are equidistant
        if distance_to_next_left_cpg == distance_to_next_right_cpg:
            new_window = replace(
                current_window,
                left_edge=cpg_positions[current_window.left_cpg_index] + 1,
                left_cpg_index=current_window.left_cpg_index + 1,
                right_cpg_index=current_window.right_cpg_index + 1,
            )
        # if left cpg is closer, increment left cpg index for new window
        elif distance_to_next_left_cpg < distance_to_next_right_cpg:
            new_window = replace(
                current_window,
                left_edge=cpg_positions[current_window.left_cpg_index] + 1,
                left_cpg_index=current_window.left_cpg_index + 1,
            )
        # if right cpg is closer, increment right cpg index for new window
        else:
            new_window_right_edge = cpg_positions[current_window.right_cpg_index] + 1
            new_window = replace(
                current_window,
                left_edge=new_window_right_edge - window_size,
                right_cpg_index=current_window.right_cpg_index + 1,
            )

        # If sufficient CpGs in window, add to list
        if current_window.right_cpg_index - current_window.left_cpg_index >= min_cpgs_per_window:
            sliding_windows.append(
                SlidingWindow(
                    current_window.left_edge,
                    new_window.left_edge,
                    current_window.left_cpg_index,
                    current_window.right_cpg_index,
                    current_window.window_size,
                )
            )

        current_window = new_window

    return sliding_windows


def get_window_std(window_methylation_array: np.ndarray) -> float:
    """
    Calculate the standard deviation of average read methylation in a window.

    Args:
        window_methylation_array (np.ndarray): Array representing the methylation states of CpGs in a window.

    Returns:
        float: Standard deviation of the average methylation per read.
    """
    read_means = np.nanmean(window_methylation_array, axis=1)
    window_std = np.nanstd(read_means)
    return window_std


def process_window_methylation_array(
    window_data: Tuple[np.ndarray, SlidingWindow, np.random.Generator],
    n_permutations_per_window: int = 50,
    ci_threshold: int = 10,
) -> Dict[str, Any]:
    """
    Process a window methylation array to calculate various statistics.

    Args:
        window_data (Tuple[np.ndarray, SlidingWindow, np.random.Generator]): Tuple containing window methylation data, the window positions, and a random number generator.
        n_permutations_per_window (int): Number of permutations to perform.
        ci_threshold (int): Confidence interval threshold.

    Returns:
        Dict[str, Any]: Dictionary containing calculated statistics for the window.
    """
    window_methylation_array, sliding_window, RNG = window_data
    window_std = get_window_std(window_methylation_array)
    shuffled_std_store = []

    for _ in range(n_permutations_per_window):
        shuffled_array = RNG.permuted(window_methylation_array, axis=0)
        shuffled_std_store.append(get_window_std(shuffled_array))

    window_data = {}
    window_data["Window_Start"] = sliding_window.start_left_edge
    window_data["Window_End"] = sliding_window.end_left_edge
    window_data["Window_Size"] = sliding_window.window_size
    window_data["N_CpGs"] = window_methylation_array.shape[1]
    window_data["N_Reads"] = window_methylation_array.shape[0]
    window_data["True_STD"] = window_std
    window_data["Mean_Permuted_STD"] = np.mean(shuffled_std_store)

    window_data["Permuted_STD_Low_CI"] = np.percentile(shuffled_std_store, ci_threshold / 2)
    window_data["Permuted_STD_High_CI"] = np.percentile(shuffled_std_store, 100 - ci_threshold / 2)
    window_data["Coherence_Score"] = window_std / window_data["Mean_Permuted_STD"]
    return window_data


def get_window_methylation_array(
    read_methylation_data_window: pd.DataFrame, min_cpgs_per_read: int
) -> np.ndarray:
    """
    Convert a dataframe of CpG methylation from a window into a 2D numpy array of methylation states.

    Args:
        read_methylation_data_window (pd.DataFrame): DataFrame containing read data for a specific window.
        min_cpgs_per_read (int): Minimum number of CpGs required per read.

    Returns:
        np.ndarray: 2D array where each row represents a read and each column a CpG position.
                    Values are methylation states (1 for methylated, 0 for unmethylated, NaN for missing).
    """
    read_methylation_data_window = read_methylation_data_window[
        ["Chromosome", "Position", "Read_Index", "Is_Methylated"]
    ]
    pivoted_df = read_methylation_data_window.pivot(
        index=["Chromosome", "Read_Index"], columns="Position", values="Is_Methylated"
    )
    pivoted_df_array = pivoted_df.values

    # filter out reads with fewer valid CpGs than the minimum
    valid_cpgs = np.count_nonzero(~np.isnan(pivoted_df_array), axis=1)
    pivoted_df_array = pivoted_df_array[valid_cpgs >= min_cpgs_per_read]
    return pivoted_df_array


def first_occurrence_indices(arr: np.ndarray) -> np.ndarray:
    """
    Find the indices of the first occurrence of each unique value in a sorted array.

    Args:
        arr (np.ndarray): Sorted input array.

    Returns:
        np.ndarray: Array of indices where each unique value first occurs.
    """
    _, idx = np.unique(arr, return_index=True)
    return idx[np.argsort(arr[idx])]


def get_all_window_methylation_arrays(
    read_methylation_data: pd.DataFrame,
    min_cpgs_per_read: int,
    sliding_windows: List[SlidingWindow],
) -> List[np.ndarray]:
    """
    Generate window arrays for all specified window.

    Args:
        read_methylation_data (pd.DataFrame): DataFrame containing methylation data at the read level.
        min_cpgs_per_read (int): Minimum number of CpGs required per read.
        sliding_windows (List[SlidingWindow]): List of sliding windows to process.

    Returns:
        List[np.ndarray]: List of 2D numpy arrays, each representing the methylation states for a window.
    """
    read_methylation_data = read_methylation_data.sort_values(by=["Chromosome", "Position"])
    position_starts = first_occurrence_indices(read_methylation_data["Position"].values)

    window_methylation_arrays = []
    for window in sliding_windows:
        valid_indices = (
            position_starts[window.left_cpg_index],
            position_starts[window.right_cpg_index],
        )
        read_methylation_data_window = read_methylation_data.iloc[
            valid_indices[0] : valid_indices[1]
        ]
        window_methylation_array = get_window_methylation_array(
            read_methylation_data_window, min_cpgs_per_read
        )
        window_methylation_arrays.append(window_methylation_array)
    return window_methylation_arrays


def get_window_coherence_data(
    read_methylation_data: pd.DataFrame,
    sliding_windows: List[SlidingWindow],
    min_cpgs_per_read: int,
    chunk_size: int = 2000,
) -> None:
    """
    Process window data and save results to a file.

    Args:
        read_methylation_data (pd.DataFrame): DataFrame containing read methylation data.
        sliding_windows (List[SlidingWindow]): List of sliding windows to process.
        min_cpgs_per_read (int): Minimum number of CpGs for a read to be included.
        chunk_size (int): Size of chunks for processing.

    Returns:
        window_df (pd.DataFrame): DataFrame containing window CpG variance data.
    """
    if len(read_methylation_data['Chromosome'].unique()) > 1:
        raise ValueError("Only one chromosome per read_methylation_data file is supported.")
    
    read_chromosome = read_methylation_data["Chromosome"].values[0]
    
    start_rng = np.random.default_rng()
    seeds = start_rng.choice(100000, size=chunk_size, replace=False)
    RNGs = [np.random.default_rng(seeds[i]) for i in range(chunk_size)]
    window_store = []
    window_chunks = np.arange(0, len(sliding_windows), chunk_size)

    pool = mp.Pool(mp.cpu_count())

    for chunk_start in tqdm.tqdm(window_chunks):
        sliding_windows_chunk = sliding_windows[chunk_start : (chunk_start + chunk_size)]
        window_methylation_arrays_chunk = get_all_window_methylation_arrays(
            read_methylation_data, min_cpgs_per_read, sliding_windows_chunk
        )
        data = [
            (window_methylation_arrays_chunk[i], sliding_windows_chunk[i], RNGs[i])
            for i in range(len(window_methylation_arrays_chunk))
        ]

        processed_chunk = pool.map(process_window_methylation_array, data)
        window_store.extend(processed_chunk)

    window_df = pd.DataFrame(window_store)
    window_df["Chromosome"] = read_chromosome
    return window_df


def run_window_cpg_coherence(
    methylation_data_path: str,
    window_size: int,
    min_cpgs_per_window: int,
    out_path: str,
    methylation_threshold: float = 0.1,
    min_cpgs_per_read: int = 5,
) -> None:
    """
    Run the entire CpG coherence pipeline.
    We are looking for regions of the genome where the methylation states of variable CpGs are correlated across reads.

    Args:
        methylation_data_path (str): Path to the methylation data file.
        window_size (int): Size of each sliding window in basepairs.
        min_cpgs_per_window (int): Minimum number of CpGs for a window to be included.
        out_path (str): Path to save the output file.
        methylation_threshold (float): Pacbio probability threshold for calling confident methylation.
        min_cpgs_per_read (int): Minimum number of CpGs per read.

    Returns:
        None
    """
    # read in methylation data with gz if necessary
    
    read_methylation_data = pd.read_csv(methylation_data_path,sep='\t')

    # filter only to CpGs with confident methylation calls
    confidently_methylated = read_methylation_data["Methylation"] >= methylation_threshold
    confidently_unmethylated = read_methylation_data["Methylation"] <= 1 - methylation_threshold
    confidently_methylated_filter = confidently_methylated | confidently_unmethylated

    read_methylation_data = read_methylation_data[confidently_methylated_filter]
    read_methylation_data["Is_Methylated"] = (read_methylation_data["Methylation"] < 0.5).astype(float)

    # filter only to CpGs with sufficient methylated and non-methylated reads
    variable_cpg_positions = get_variable_cpg_positions(read_methylation_data)
    read_methylation_data = read_methylation_data.merge(variable_cpg_positions, how="inner")


    cpg_positions = variable_cpg_positions["Position"].values
    
    sliding_windows = get_sliding_window_store(cpg_positions, window_size, min_cpgs_per_window)
    
    window_coherence = get_window_coherence_data(
        read_methylation_data, sliding_windows, min_cpgs_per_read
    )

    window_coherence.to_csv(out_path, sep='\t', index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get CpG coherence data across sliding windows of a specified size"
    )
    parser.add_argument(
        "--methylation_data_path",
        type=str,
        help="Path to the read-level CpG methylation data",
        required=True,
    )
    parser.add_argument("--out_path", type=str, help="Path to store output", required=True)
    parser.add_argument("--window_size", type=int, help="Window size in basepairs", required=False,default=250)
    parser.add_argument(
        "--min_cpgs_per_window", type=int, help="Minimum CpGs per window", required=False,default=20
    )
    
    parser.add_argument('--methylation_threshold', type=float, default=0.1, help='Probability threshold for confident methylation calls. CpG sites with methylation probabilities below this threshold and above 1-threshold are considered unmethylated and methylated, respectively.')

    args = parser.parse_args()

    run_window_cpg_coherence(
        args.methylation_data_path,
        args.window_size,
        args.min_cpgs_per_window,
        args.out_path,
    )

