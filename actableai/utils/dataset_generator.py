import string
import json
from copy import copy, deepcopy
from pathlib import Path
from time import time
from typing import Optional, List, Union, Tuple
from datetime import date, timedelta

import numpy as np
import pandas as pd


class DatasetGenerator:
    @classmethod
    def __generate_text_column(
        cls,
        random_generator: np.random.Generator,
        rows: int,
        n_categories: Optional[int] = None,
        word_len_range: Optional[Tuple[int, int]] = None,
        word_count_range: Optional[Tuple[int, int]] = None,
    ) -> List[str]:
        """Handle/pre-process the text column parameters (in place)

        Parameters
        ----------
        random_generator: Random Generator
        rows: Number of rows
        n_categories: Number of categories (number of unique strings), default: rows
        word_len_range: Range for the lengths of the words, min included, max excluded, default: (5, 10)
        word_count_range: Range for the number of words per row, min included, max excluded, default (1, 2)

        Returns
        -------
        List of values to use for the column
        """

        values = []

        if n_categories is None:
            n_categories = rows
        if word_len_range is None or len(word_len_range) != 2:
            word_len_range = (5, 10)
        if word_count_range is None or len(word_count_range) != 2:
            word_count_range = (1, 2)

        for _ in range(n_categories):
            random_string = ""
            word_counts = random_generator.integers(
                word_count_range[0], word_count_range[1]
            )

            for _ in range(word_counts):
                word_len = random_generator.integers(
                    word_len_range[0], word_len_range[1]
                )
                random_string += "".join(
                    random_generator.choice(list(string.ascii_letters))
                    for _ in range(word_len)
                )

                random_string += " "

            values.append(random_string[:-1])

        return values

    @classmethod
    def __generate_number_column(
        cls,
        random_generator: np.random.Generator,
        rows: int,
        is_float: Optional[bool] = None,
        number_range: Optional[Tuple[int, int]] = None,
    ) -> List[Union[int, float]]:
        """Handle/pre-process the number column parameters (in place)

        Parameters
        ----------
        random_generator: Random Generator
        rows: Number of rows
        is_float: True if the numbers generated need to be floating point number, default: True
        number_range: Min included, max excluded, default: (0, 10)

        Returns
        -------
        List of values to use for the column
        """

        values = []

        if is_float is None:
            is_float = True
        if number_range is None or len(number_range) != 2:
            number_range = (0, rows)

        for _ in range(rows):
            if is_float:
                number = random_generator.uniform(number_range[0], number_range[1])
            else:
                number = random_generator.integers(number_range[0], number_range[1])

            values.append(number)

        return values

    @classmethod
    def __generate_date_column(
        cls,
        random_generator: np.random.Generator,
        rows: int,
        freq: str = None,
        start: Union[str, pd.Timestamp] = None,
        end: Union[str, pd.Timestamp] = None,
    ) -> pd.DatetimeIndex:
        """Handle/pre-process the date column parameters (in place)

        Parameters
        ----------
        random_generator: Random Generator
        rows: Number of rows
        freq: Frequency, see pandas date_range freq parameter, default: "D"
        start: Start date, default: None
        end: End date, default: Today - <random_number_of_days>

        Returns
        -------
        List of values to use for the column
        """

        if freq is not None and start is not None and end is not None:
            raise Exception("Error while parsing columns parameters")
        if freq is None and (start is None or end is None):
            freq = "D"
        if end is None and start is None:
            end = date.today() - timedelta(
                days=random_generator.integers(0, 60000, dtype=int)
            )

        return pd.date_range(start=start, end=end, freq=freq, periods=rows)

    @classmethod
    def generate(
        cls,
        columns_parameters: List[dict],
        rows: int = 1000,
        output_path: Optional[Union[str, Path]] = None,
        save_parameters_path: Optional[Union[str, Path]] = None,
        random_state: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """Generate a dataset, this function generates random data and no sense should be expected from it

        Parameters
        ----------
        columns_parameters: List of parameters for columns, see below
        rows: Number of rows to generate
        output_path: Path where the dataset will be saved, if None the function returns a DataFrame
        save_parameters_path: If not None will save the parameters used to generate this dataset to this path
        random_state: Seed for the random generator, used to fix the result, default: None


        Columns Parameters Specification
        --------------------------------
        - Common parameters:
        {
            "name": <column_name>,                      # Name of the column, default: col_<index>
            "type": <type>,                             # Type of the column, choices: ["text", "number", "date"]
            "values": [<value_list>]                    # List of values, if len(values) == rows then those values will
                                                        be used in the given order. If not each row will pick a random
                                                        value in values. It will always override other parameters

                                                        # If value is set type is omitted
        }
        - Text column parameters
        {
            "type": "text",
            "n_categories": <n_categories>,             # The number of categories (number of unique strings), default:
                                                        rows
            "range": (<min_range>, <max_range>),        # Range for generated word lengths, min included, max excluded,
                                                        default: (5, 10)
            "word_range": (<min_range, <max_range>),    # Range for the number of words to create, min included, max
                                                        excluded, default: (1, 2)
        }
        - Number column parameters:
        {
            "type": "number",
            "float": <True or False>,                   # default: True
            "range": (<min_range>, <max_range>),        # Min included, max excluded, default: (0, <rows>)
        }
        - Date column:
        {
            "type": "date",
            "freq": <frequency>,                        # See pandas date_range freq parameter, default: "D"
            "start": <date>,                            # Start date, default: None
            "end": <date>                               # End date, default: Today - <random_number_of_days>

                                                        # At least one of those three parameters (freq, start, and end)
                                                        must be None
        }


        Examples
        --------
        Column parameters examples:
        Text column containing yes or no:
        {
            "values": ["yes", "no"]
        }
        Text column containing unique random string of len 10
        {
            "type": "text",
            "range": (10, 11)
        }
        Number column with random float between 0 and 1
        {
            "type": "number",
            "float": True,
            "range": (0, 1)
        }
        Number column containing either 10, 100 or 1000
        {
            "values": [10, 100, 1000]
        }

        Returns
        -------
        Pandas DataFrame containing the generated dataset
        """
        if save_parameters_path is not None:
            parameters = {
                "rows": rows,
                "columns_parameters": columns_parameters,
                "random_state": random_state if random_state is not None else "None",
            }
            with open(save_parameters_path, "w") as parameters_file:
                json.dump(parameters, parameters_file)

        random_generator = np.random.default_rng(random_state)
        columns_parameters = deepcopy(columns_parameters)

        # Construct columns
        for column_index, column in enumerate(columns_parameters):
            # Set column name
            if "name" not in column:
                column["name"] = f"col_{column_index}"

            # Generate the values list if needed
            if "values" not in column or len(column["values"]) <= 0:
                if column["type"] == "text":
                    column["values"] = cls.__generate_text_column(
                        random_generator,
                        rows,
                        n_categories=column.get("n_categories"),
                        word_len_range=column.get("range"),
                        word_count_range=column.get("word_range"),
                    )
                elif column["type"] == "number":
                    column["values"] = cls.__generate_number_column(
                        random_generator,
                        rows,
                        is_float=column.get("float"),
                        number_range=column.get("range"),
                    )
                elif column["type"] == "date":
                    column["values"] = cls.__generate_date_column(
                        random_generator,
                        rows,
                        freq=column.get("freq"),
                        start=column.get("start"),
                        end=column.get("end"),
                    )
                else:
                    raise Exception("Error while parsing columns parameters")

            # Create the final values list
            if len(column["values"]) < rows:
                values = copy(column["values"])
                column["values"] = [
                    random_generator.choice(values) for _ in range(rows)
                ]
            elif len(column["values"]) > rows:
                column["values"] = column["values"][:rows]

        df = pd.DataFrame()
        for column in columns_parameters:
            df[column["name"]] = column["values"]

        if output_path is None:
            return df
        df.to_csv(output_path, index=False)

    @classmethod
    def generate_from_file(
        cls,
        parameters_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Optional[pd.DataFrame]:
        """Generate dataset from a file containing the parameters

        Parameters
        ----------
        parameters_path: The path to the parameters json file
        output_path: Path where the dataset will be saved, if None the function returns a DataFrame

        Returns
        -------
        Pandas DataFrame containing the generated dataset
        """
        with open(parameters_path) as parameters_file:
            parameters = json.load(parameters_file)

            rows = parameters["rows"]
            columns_parameters = parameters["columns_parameters"]
            random_state = parameters["random_state"]
            if random_state == "None":
                random_state = None

            return cls.generate(
                columns_parameters=columns_parameters,
                rows=rows,
                output_path=output_path,
                random_state=random_state,
            )
