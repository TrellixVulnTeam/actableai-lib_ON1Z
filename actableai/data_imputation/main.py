import pandas as pd

from actableai.data_imputation import (
    DataFrame,
    NullDetector,
    ValidationDetector,
    MisplacedDetector,
)
from actableai.data_imputation.error_detector.rule_parser import RulesRaw
from actableai.tasks.data_imputation import construct_rules
from error_detector import RulesBuilder

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)

    df = DataFrame("./experiments/data/apartment/apartment.csv")

    rule_not_ok = rules = [
        {
            "title": "test",
            "validations": [
                {
                    "when": [
                        {
                            "column": {
                                "value": "rental_price",
                                "label": "rental_price",
                            },
                            "operator": {"label": ">", "value": ">"},
                            "comparedColumn": {
                                "value": "rental_price",
                                "label": "rental_price",
                            },
                        }
                    ],
                    "then": [
                        {
                            "column": {"value": "sqft", "label": "sqft"},
                            "operator": {"label": ">", "value": ">"},
                            "comparedColumn": {
                                "value": "sqft",
                                "label": "sqft",
                            },
                        },
                        {
                            "column": {
                                "value": "initial_price",
                                "label": "initial_price",
                            },
                            "operator": {"label": ">", "value": ">"},
                            "comparedColumn": {
                                "value": "initial_price",
                                "label": "initial_price",
                            },
                        },
                    ],
                },
            ],
            "misplaced": [
                # {
                #     "column": {
                #         "label": "Sample",
                #         "value": "Sample",
                #     },
                #     "operator": {"label": "<", "value": "<"},
                #     "value": "0",
                #     "isRegex": False,
                # }
            ],
        }
    ]

    rule_str = construct_rules(rule_not_ok)

    custom_rules = RulesBuilder.parse(df.column_types, RulesRaw(*rule_str))

    detectors = [
        NullDetector(),
        ValidationDetector(constraints=custom_rules.constraints),
        MisplacedDetector(customize_rules=custom_rules.match_rules),
    ]
    errors = df.detect_error(*detectors)

    for err in errors:
        print(err)

    fixed_df = df.auto_fix(errors)
    for fix_info in sorted(fixed_df.fix_info, key=lambda x: f"{x.col}_{x.index}"):
        print(fix_info)

    fixed_df.to_csv("./experiments/data/apartment/fixed.csv", index=False)
