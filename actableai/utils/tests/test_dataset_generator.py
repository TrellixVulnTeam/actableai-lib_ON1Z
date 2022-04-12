import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from actableai.utils.dataset_generator import DatasetGenerator


def call_dataset_generator(
    tmp_path: Path,
    columns_parameters: List[dict],
    rows: int,
    save_output: bool,
    save_parameters: bool,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Call dataset generator

    Parameters
    ----------
    tmp_path: The temporary path from pytest, this is where the output will be saved if needed
    columns_parameters: The dataset columns parameters
    rows: The number of rows to generate
    save_output: If true will save the output as a csv to then read it and return the content
    save_parameters: If true will save the parameters as a json to then parse them and return the generated dataset
    random_state: The random state

    Returns
    -------
    Either the generated dataframe directly, or the read dataframe from the generated csv file
    """

    output_path = None
    if save_output:
        output_path = tmp_path / "dataset.csv"

    save_parameters_path = None
    if save_parameters:
        save_parameters_path = tmp_path / "parameters.json"

    df = DatasetGenerator.generate(
        columns_parameters=columns_parameters,
        rows=rows,
        output_path=output_path if save_parameters_path is None else None,
        save_parameters_path=save_parameters_path,
        random_state=random_state,
    )

    if save_parameters_path is not None:
        df = None
        df = DatasetGenerator.generate_from_file(
            parameters_path=save_parameters_path, output_path=output_path
        )

    if output_path is not None:
        df = pd.read_csv(output_path)
    return df


class TestGenerateDataset:
    def test_generate_simple_dataset(self):
        columns_parameters = [
            {"type": "text"},
            {
                "type": "number",
            },
            {"type": "date"},
        ]
        df = DatasetGenerator.generate(columns_parameters=columns_parameters, rows=10)

        assert df is not None

    def test_generate_and_save_simple_dataset(self, tmp_path):
        columns_parameters = [
            {"type": "text"},
            {
                "type": "number",
            },
            {"type": "date"},
        ]
        output_path = tmp_path / "dataset.csv"
        res = DatasetGenerator.generate(
            columns_parameters=columns_parameters, rows=10, output_path=output_path
        )

        assert res is None
        assert os.path.isfile(output_path)

        df = pd.read_csv(output_path)

        assert df is not None


@pytest.mark.parametrize("save_output", [True, False])
@pytest.mark.parametrize("save_parameters", [True, False])
class TestGenerateDatasetParameters:
    @pytest.mark.parametrize("rows", [10, 100, 500])
    def test_rows(self, tmp_path, save_output, save_parameters, rows):
        columns_parameters = [
            {"type": "text"},
            {
                "type": "number",
            },
            {"type": "date"},
        ]
        df = call_dataset_generator(
            tmp_path, columns_parameters, rows, save_output, save_parameters
        )

        assert len(df) == rows

    @pytest.mark.parametrize("columns", [2, 4, 8])
    @pytest.mark.parametrize("column_type", ["text", "number", "date"])
    def test_columns(
        self, tmp_path, save_output, save_parameters, columns, column_type
    ):
        columns_parameters = [{"type": column_type} for _ in range(columns)]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )

        assert len(df.columns) == columns

    @pytest.mark.parametrize("columns", [2, 4, 8])
    @pytest.mark.parametrize("random_state", [0, 1, 2])
    def test_columns_mixed_type(
        self, tmp_path, save_output, save_parameters, columns, random_state
    ):
        random_generator = np.random.default_rng(random_state)
        columns_parameters = [
            {"type": random_generator.choice(["text", "number", "date"])}
            for _ in range(columns)
        ]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )

        assert len(df.columns) == columns

    def test_no_type(self, tmp_path, save_output, save_parameters):
        columns_parameters = [{"name": "test"}]

        with pytest.raises(Exception):
            call_dataset_generator(
                tmp_path, columns_parameters, 10, save_output, save_parameters
            )

    @pytest.mark.parametrize("random_state", [0, 1, 2])
    @pytest.mark.parametrize("column_type", ["text", "number", "date"])
    def test_random_state_one_column(
        self, tmp_path, save_output, save_parameters, random_state, column_type
    ):
        columns_parameters = [{"type": column_type}]
        df_1 = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters, random_state
        )
        df_2 = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters, random_state
        )
        df_3 = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )

        assert df_1.equals(df_2)
        assert not df_1.equals(df_3)

    @pytest.mark.parametrize("random_state", [0, 1, 2])
    def test_random_state_three_columns(
        self, tmp_path, save_output, save_parameters, random_state
    ):
        columns_parameters = [
            {"type": "text"},
            {
                "type": "number",
            },
            {"type": "date"},
        ]
        df_1 = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters, random_state
        )
        df_2 = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters, random_state
        )
        df_3 = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )

        assert df_1.equals(df_2)
        assert not df_1.equals(df_3)

    @pytest.mark.parametrize("column_type", ["text", "number", "date"])
    def test_name_one_column(self, tmp_path, save_output, save_parameters, column_type):
        columns_parameters = [{"name": "column_name", "type": column_type}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )

        assert df.columns[0] == "column_name"

    def test_name_three_columns(self, tmp_path, save_output, save_parameters):
        columns_parameters = [
            {"name": "column_text", "type": "text"},
            {
                "name": "column_number",
                "type": "number",
            },
            {"name": "column_date", "type": "date"},
        ]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )

        assert df.columns[0] == "column_text"
        assert df.columns[1] == "column_number"
        assert df.columns[2] == "column_date"

    @pytest.mark.parametrize("column_type", ["text", "number", "date"])
    def test_no_name_one_column(
        self, tmp_path, save_output, save_parameters, column_type
    ):
        columns_parameters = [{"type": column_type}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )

        assert df.columns[0] == "col_0"

    def test_no_name_three_columns(self, tmp_path, save_output, save_parameters):
        columns_parameters = [
            {"type": "text"},
            {
                "type": "number",
            },
            {"type": "date"},
        ]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )

        assert df.columns[0] == "col_0"
        assert df.columns[1] == "col_1"
        assert df.columns[2] == "col_2"

    def test_one_value(self, tmp_path, save_output, save_parameters):
        columns_parameters = [{"values": [0]}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 500, save_output, save_parameters
        )

        assert len(df[df.columns[0]].unique()) <= 1
        assert df[df.columns[0]].unique()[0] == 0

    def test_three_values(self, tmp_path, save_output, save_parameters):
        columns_parameters = [{"values": [0, 10, 20]}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 500, save_output, save_parameters
        )

        unique_values = df[df.columns[0]].unique()

        assert len(unique_values) <= 3
        for value in unique_values:
            assert value in [0, 10, 20]

    @pytest.mark.parametrize("n_categories", [2, 4, 8])
    def test_text_column_categories(
        self, tmp_path, save_output, save_parameters, n_categories
    ):
        columns_parameters = [{"type": "text", "n_categories": n_categories}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 500, save_output, save_parameters
        )

        assert len(df[df.columns[0]].unique()) <= n_categories

    @pytest.mark.parametrize("range_min", [2, 4, 8])
    @pytest.mark.parametrize("range_max", [12, 14, 18])
    def test_text_column_range(
        self, tmp_path, save_output, save_parameters, range_min, range_max
    ):
        columns_parameters = [{"type": "text", "range": (range_min, range_max)}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 500, save_output, save_parameters
        )

        col_lengths = df[df.columns[0]].str.len()
        assert col_lengths.min() >= range_min
        assert col_lengths.max() < range_max

    @pytest.mark.parametrize("range_min", [2, 4, 8])
    @pytest.mark.parametrize("range_max", [12, 14, 18])
    def test_text_column_word_range(
        self, tmp_path, save_output, save_parameters, range_min, range_max
    ):
        columns_parameters = [{"type": "text", "word_range": (range_min, range_max)}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 500, save_output, save_parameters
        )

        col_word_count = df[df.columns[0]].str.count(" ") + 1
        assert col_word_count.min() >= range_min
        assert col_word_count.max() < range_max

    def test_text_column_default(self, tmp_path, save_output, save_parameters):
        columns_parameters = [{"type": "text"}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 500, save_output, save_parameters
        )

        col_word_count = df[df.columns[0]].str.count(" ") + 1
        col_lengths = df[df.columns[0]].str.len() - col_word_count + 1
        col_min_lengths = col_word_count * 5
        col_max_lengths = col_word_count * 10
        assert len(df[df.columns[0]].unique()) <= 500
        assert (col_lengths >= col_min_lengths).all()
        assert (col_lengths < col_max_lengths).all()
        assert col_word_count.min() >= 1
        assert col_word_count.min() < 2

    def test_text_column_default_categories(
        self, tmp_path, save_output, save_parameters
    ):
        columns_parameters = [{"type": "text", "range": (1, 2), "word_range": (1, 5)}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 500, save_output, save_parameters
        )

        col_word_count = df[df.columns[0]].str.count(" ") + 1
        col_lengths = df[df.columns[0]].str.len() - col_word_count + 1
        col_min_lengths = col_word_count * 1
        col_max_lengths = col_word_count * 2
        assert len(df[df.columns[0]].unique()) <= 500
        assert (col_lengths >= col_min_lengths).all()
        assert (col_lengths < col_max_lengths).all()
        assert col_word_count.min() >= 1
        assert col_word_count.min() < 5

    def test_text_column_default_range(self, tmp_path, save_output, save_parameters):
        columns_parameters = [{"type": "text", "n_categories": 2, "word_range": (1, 5)}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 500, save_output, save_parameters
        )

        col_word_count = df[df.columns[0]].str.count(" ") + 1
        col_lengths = df[df.columns[0]].str.len() - col_word_count + 1
        col_min_lengths = col_word_count * 5
        col_max_lengths = col_word_count * 10
        assert len(df[df.columns[0]].unique()) <= 2
        assert (col_lengths >= col_min_lengths).all()
        assert (col_lengths < col_max_lengths).all()
        assert col_word_count.min() >= 1
        assert col_word_count.min() < 5

    def test_text_column_default_word_range(
        self, tmp_path, save_output, save_parameters
    ):
        columns_parameters = [
            {
                "type": "text",
                "n_categories": 2,
                "range": (1, 2),
            }
        ]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 500, save_output, save_parameters
        )

        col_word_count = df[df.columns[0]].str.count(" ") + 1
        col_lengths = df[df.columns[0]].str.len() - col_word_count + 1
        col_min_lengths = col_word_count * 1
        col_max_lengths = col_word_count * 2
        assert len(df[df.columns[0]].unique()) <= 2
        assert (col_lengths >= col_min_lengths).all()
        assert (col_lengths < col_max_lengths).all()
        assert col_word_count.min() >= 1
        assert col_word_count.min() < 2

    def test_number_column_float(self, tmp_path, save_output, save_parameters):
        columns_parameters = [{"type": "number", "float": True}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )

        assert df[df.columns[0]].dtype == float

    def test_number_column_int(self, tmp_path, save_output, save_parameters):
        columns_parameters = [{"type": "number", "float": False}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )

        assert df[df.columns[0]].dtype == int

    @pytest.mark.parametrize("range_min", [2, 4, 8])
    @pytest.mark.parametrize("range_max", [12, 14, 18])
    def test_number_column_float_range(
        self, tmp_path, save_output, save_parameters, range_min, range_max
    ):
        columns_parameters = [
            {"type": "number", "float": True, "range": (range_min, range_max)}
        ]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 500, save_output, save_parameters
        )

        assert df[df.columns[0]].min() >= range_min
        assert df[df.columns[0]].max() < range_max

    @pytest.mark.parametrize("range_min", [2, 4, 8])
    @pytest.mark.parametrize("range_max", [12, 14, 18])
    def test_number_column_int_range(
        self, tmp_path, save_output, save_parameters, range_min, range_max
    ):
        columns_parameters = [
            {"type": "number", "float": False, "range": (range_min, range_max)}
        ]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 500, save_output, save_parameters
        )

        assert df[df.columns[0]].min() >= range_min
        assert df[df.columns[0]].max() < range_max

    def test_number_column_default(self, tmp_path, save_output, save_parameters):
        columns_parameters = [{"type": "number"}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 500, save_output, save_parameters
        )

        assert df[df.columns[0]].dtype == float
        assert df[df.columns[0]].min() >= 0
        assert df[df.columns[0]].max() < 500

    def test_number_column_default_float(self, tmp_path, save_output, save_parameters):
        columns_parameters = [{"type": "number", "range": (1, 2)}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 500, save_output, save_parameters
        )

        assert df[df.columns[0]].dtype == float
        assert df[df.columns[0]].min() >= 1
        assert df[df.columns[0]].max() < 2

    def test_number_column_default_range(self, tmp_path, save_output, save_parameters):
        columns_parameters = [{"type": "number", "float": False}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 500, save_output, save_parameters
        )

        assert df[df.columns[0]].dtype == int
        assert df[df.columns[0]].min() >= 0
        assert df[df.columns[0]].max() < 500

    @pytest.mark.parametrize("freq", ["D", "M", "2D", "2M"])
    def test_date_column_freq(self, tmp_path, save_output, save_parameters, freq):
        columns_parameters = [{"type": "date", "freq": freq}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )

        assert pd.infer_freq(df[df.columns[0]]) == freq

    def test_date_column_start(self, tmp_path, save_output, save_parameters):
        columns_parameters = [{"type": "date", "start": "2000-01-01"}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])

        assert df[df.columns[0]][0] == pd.to_datetime("2000-01-01")

    def test_date_column_end(self, tmp_path, save_output, save_parameters):
        columns_parameters = [{"type": "date", "end": "2000-01-01"}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])

        assert df[df.columns[0]][9] == pd.to_datetime("2000-01-01")

    def test_date_column_start_freq(self, tmp_path, save_output, save_parameters):
        columns_parameters = [{"type": "date", "start": "2000-01-01", "freq": "D"}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])

        assert df[df.columns[0]][9] == pd.to_datetime("2000-01-10")

    def test_date_column_end_freq(self, tmp_path, save_output, save_parameters):
        columns_parameters = [{"type": "date", "end": "2000-01-10", "freq": "D"}]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])

        assert df[df.columns[0]][0] == pd.to_datetime("2000-01-01")

    def test_date_column_start_end(self, tmp_path, save_output, save_parameters):
        columns_parameters = [
            {"type": "date", "start": "2000-01-01", "end": "2000-01-10"}
        ]
        df = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])

        assert pd.infer_freq(df[df.columns[0]]) == "D"

    def test_date_column_start_end_freq(self, tmp_path, save_output, save_parameters):
        columns_parameters = [
            {"type": "date", "start": "2000-01-01", "end": "2000-01-10", "freq": "M"}
        ]

        with pytest.raises(Exception):
            call_dataset_generator(
                tmp_path, columns_parameters, 10, save_output, save_parameters
            )

    def test_date_column_default(self, tmp_path, save_output, save_parameters):
        columns_parameters = [{"type": "date"}]

        df = call_dataset_generator(
            tmp_path, columns_parameters, 10, save_output, save_parameters
        )
        assert pd.infer_freq(df[df.columns[0]]) == "D"
