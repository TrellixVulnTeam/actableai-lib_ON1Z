import os
import boto3
import torch
import mxnet as mx
import numpy as np
import pandas as pd
from scipy.stats import sem
import actableai.timeseries.models
import actableai.timeseries.params as params
from jinja2 import Environment, FileSystemLoader

mx_ctx = mx.cpu()
torch_device = torch.device("cpu")

# Todo: will grant public read access
bucket = None

multivariate_data_folder = 'ts_test_data/'
if not os.path.exists(multivariate_data_folder):
    os.makedirs(multivariate_data_folder)
    for s3_file in bucket.objects.filter(Prefix = multivariate_data_folder):
        file_object = s3_file.key
        print(f'Dowloading {file_object}')
        bucket.download_file(file_object,file_object)


def evaluate(df_train, param, prediction_length):

    target_dim = len(df_train.columns)

    univariate_params = {"feedforward":params.FeedForwardParams(hidden_layer_size=10, epochs=2,
                                context_length=None,learning_rate=0.001,l2=1e-08),
                        "prophet":params.ProphetParams(),
                        "rforecast":params.RForecastParams()
                        }
    if df_train.shape[0] <= 1000:
        univariate_params["deepar"]=params.DeepARParams(num_cells=20, num_layers=2, epochs=1,
                                context_length=None)

    multivariate_params = {"feedforward": params.FeedForwardParams(hidden_layer_size=10, epochs=2,
                                context_length=None,learning_rate=0.001,l2=1e-08),
                        "deepvar": params.DeepVARParams(
                                epochs=2, num_layers=2, num_cells=20, learning_rate=0.0001, dropout_rate=0,
                                context_length=None,l2=1e-4),
                        "transformer": params.TransformerTempFlowParams(
                            context_length=None,
                            epochs=2,learning_rate=0.001, l2=0.0001, num_heads=8, d_model=8),
                        }

    m = actableai.timeseries.models.AAITimeseriesForecaster(
        prediction_length, mx_ctx, torch_device,
        univariate_model_params=[univariate_params[param]] if target_dim==1 else None,
        multivariate_model_params=[multivariate_params[param]] if target_dim>1 else None)

    m.fit(
              df_train,
              trials=1,
              loss="mean_wQuantileLoss",
              tune_params={
                  "resources_per_trial": {
                      "cpu": float(2),
                  },"raise_on_failed_trial": False
              },
              max_concurrent=int(5),
              eval_samples=5,
            )

    df_score=m.score(df_train)
    return df_score


repeat_test = 2
l_data = sorted(os.listdir(multivariate_data_folder))
multivariate_params = ["feedforward", "deepvar", "transformer"]

list_df=[]
for mp in multivariate_params:
    full_dict_score={}

    for df_name in l_data:

        l_tmp_score = []
        file_location = multivariate_data_folder + df_name
        df = pd.read_csv(file_location)

        pd_date = pd.to_datetime(df['date'])
        df = df.set_index(pd_date)
        df.index.name = 'date'
        df.drop(columns=['date'], inplace=True)


        for i in range(repeat_test):
            df_score = evaluate(df, mp, 10)
            need = list(df_score['agg_metrics'].keys())[-24:]
            dict_score = { your_key: df_score['agg_metrics'][your_key] for your_key in need }
            l_tmp_score.append(list(dict_score.values()))

        mean_score = np.round(np.mean(l_tmp_score, axis=0), 2)
        full_dict_score[df_name] = mean_score

        sem_score = np.round(sem(l_tmp_score, axis=0), 2)
        full_dict_score['sem_'+df_name] = sem_score

    df_score = pd.DataFrame(full_dict_score, columns=full_dict_score.keys())
    df_score.insert(0, 'score',  dict_score.keys())
    df_score.insert(1, 'model_name', mp)
    list_df.append(df_score)


prev_score_df = pd.read_csv('score.csv',index_col=[0]).reset_index(drop=True)

new_score_df = pd.concat(list_df).reset_index(drop=True)
new_score_df.to_csv('score.csv')


col_index = 0
def highlight_max(new_score_df, prev_score_df=None, props=''):
    global col_index
    series_col = prev_score_df.iloc[:,col_index]
    col_index += 1

    if pd.api.types.is_numeric_dtype(series_col) and series_col.name[:3]!='sem':
        return np.where(series_col>new_score_df, props, 'color:red')

    elif pd.api.types.is_string_dtype(series_col) :
        return np.where(series_col != None, '', '')

style1 = new_score_df.style.apply(highlight_max, prev_score_df=prev_score_df, props='color:green').render()

textHML='''
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
'''

html_file = "index.html"
file = open(html_file,"w")
file.write(textHML)
file.close()

env = Environment(loader=FileSystemLoader('.'))
template = env.get_template(html_file)


template_vars = {"title" : "ReportMetric",
                 "new_score_df": style1,
                 "prev_score_df":prev_score_df.style.render()
                 }
html_out = template.render(template_vars)

file = open(html_file,"w")
file.write(html_out)
file.close()





