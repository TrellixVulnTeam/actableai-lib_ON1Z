import json
import os
import sys
import time
import warnings
from itertools import product

import boto3
import econml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import shap
from actableai.causal.models import AAICausalEstimator
from actableai.causal.params import (
    LinearDMLSingleBinaryTreatmentParams,
    SparseLinearDMLSingleBinaryTreatmentParams,
)
from actableai.data_validation.base import CheckLevels
from actableai.data_validation.params import CausalDataValidator
from actableai.tasks.causal_inference import remote_causal
from dowhy import CausalModel
from econml.dml import DML, CausalForestDML, LinearDML, SparseLinearDML
from jinja2 import Environment, FileSystemLoader
from ray import tune
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    Lasso,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error


warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)
ray.init(log_to_driver=False)

session = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)
s3 = session.resource("s3")
bucket = s3.Bucket("actable-ai-machine-learning")


data_dir = "causal_test_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
for s3_file in bucket.objects.filter(Prefix=data_dir):
    file_object = s3_file.key
    print(f"Dowloading {file_object}")
    bucket.download_file(file_object, file_object)


def preprocess_acic(df):
    # rename common cause columns
    common_causes = [c for c in df.columns if c.split(".")[0] == "x"]
    d = {k: k.split(".")[1] for k in common_causes}
    df = df.rename(columns=d)
    common_causes = ["x_" + str(i) for i in range(1, 59)]

    # convert treatment variable type to boolean
    m = {"ctl": 0, "trt": 1}
    df["z"] = df["z"].map(m)

    # convert common causes to either boolean or dummies categorical
    bool_x = [
        c for c in common_causes if ((df[c].dtypes != float) and (df[c].nunique() == 2))
    ]
    cat_x = [
        c for c in common_causes if ((df[c].dtypes != float) and (df[c].nunique() > 2))
    ]

    for c in bool_x:
        values = sorted(df[c].unique())
        m = {values[0]: 1, values[1]: 0}
        df[c] = df[c].map(m)

    df = pd.get_dummies(df, columns=cat_x, drop_first=True)
    common_causes = [c for c in df.columns if c[0] == "x"]
    treatment = "z"
    outcome = "y"
    return df, treatment, outcome, common_causes


records = []
for parameter_num in range(1, 78):
    for simulation_num in range(1, 2):
        filename = f"{data_dir}/aciccomp2016_{parameter_num}_{simulation_num}.csv"
        print(f"processing {filename}")
        try:
            start = time.time()
            df = pd.read_csv(filename)
            ate = (df["y.1"] - df["y.0"]).mean()
            treated = df["z"] == "trt"
            att = (df.loc[treated, "y.1"] - df.loc[treated, "y.0"]).mean()
            atc = (df.loc[~treated, "y.1"] - df.loc[~treated, "y.0"]).mean()
            pd_table, treatment, outcome, common_causes = preprocess_acic(df)
            model_params = [
                LinearDMLSingleBinaryTreatmentParams(),
                SparseLinearDMLSingleBinaryTreatmentParams(polyfeat_degree=(1, 4)),
            ]
            results = remote_causal(
                pd_table=pd_table,
                treatment=treatment,
                outcome=outcome,
                common_causes=common_causes,
                discrete_treatment=True,
                target_units="att",
                model_params=model_params,
                RAY_CPU_PER_TRIAL=2,
                RAY_GPU_PER_TRIAL=0,
                RAY_MAX_CONCURRENT=3,
                trials=3,
                verbose=0,
            )
            run_time = results["runtime"]
            estimated_att = results["data"]["effect"][0]
            estimated_att_lb = results["data"]["lb"][0]
            estimated_att_ub = results["data"]["ub"][0]
            res = (
                parameter_num,
                simulation_num,
                ate,
                att,
                atc,
                estimated_att,
                estimated_att_lb,
                estimated_att_ub,
                run_time,
            )
            records.append(res)
            print(f"res={res}, Elapsed = {time.time()-start:.2f} s")
        except Exception as e:
            print(e)
            continue

results_df = pd.DataFrame(
    records,
    columns=[
        "parameter_num",
        "simulation_num",
        "ate",
        "att",
        "atc",
        "estimated_att",
        "estimated_att_lb",
        "estimated_att_ub",
        "run_time",
    ],
)

results_df["residual"] = results_df["estimated_att"] - results_df["att"]

rmse = np.round(
    np.sqrt(mean_squared_error(results_df["att"], results_df["estimated_att"])), 2
)
std_err = np.round(np.std(results_df["residual"].values), 2)

prev_score_df = pd.read_csv("score.csv")

new_score_df = pd.DataFrame(
    [("aciccomp2016", rmse, std_err)], columns=["Data set", "RMSE", "STD"]
)
new_score_df.to_csv("score.csv", index=False)

textHML = """
  <!DOCTYPE html>
  <html>
    <head lang="en">
      <meta charset="UTF-8">
      <title>{{ title }}</title>
    </head>
  <body>
      <h2>New score</h2>
     {{ new_score_df }}
      <h2>Previous score</h2>
     {{ prev_score_df }}
  </body>
  </html>
"""

html_file = "index.html"
file = open(html_file, "w")
file.write(textHML)
file.close()

env = Environment(loader=FileSystemLoader("."))
template = env.get_template(html_file)


template_vars = {
    "title": "ReportMetric",
    "prev_score_df": prev_score_df.to_html(),
    "new_score_df": new_score_df.to_html(),
}
html_out = template.render(template_vars)

file = open(html_file, "w")
file.write(html_out)
file.close()
