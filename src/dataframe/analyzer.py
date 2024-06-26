import itertools
import pandas as pd


class Analyzer:
    @staticmethod
    def find_single_id_candidate_columns(df: pd.DataFrame):
        total_records = df.shape[0]
        column_list = df.columns
        id_candidates = []
        for column in column_list:
            column_series = df[column]
            column_unique = column_series.unique()
            column_unique_count = column_unique.size
            if column_unique_count == total_records:
                id_candidates.append(column)
        return id_candidates

    @staticmethod
    def find_id_pair_candidates(df: pd.DataFrame):
        total_records = df.shape[0]
        column_list = df.columns
        id_candidate_pairs = []
        combo_df = pd.DataFrame()
        for column_set in itertools.combinations(column_list, 2):
            if column_set is None:
                continue
            first_column = column_set[0]
            second_column = column_set[1]
            combo_df["combo"] = df[first_column].astype(str) + df[second_column].astype(str)
            combined_unique = combo_df["combo"].unique()
            combined_unique_count = combined_unique.size
            if combined_unique_count == total_records:
                id_candidate_pairs.append(column_set)
        return id_candidate_pairs
