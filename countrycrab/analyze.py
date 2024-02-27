###
# Copyright (2024) Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###

import os
import numpy as np
import pandas as pd
import mlflow
from ray import tune
from mlflow.entities import ViewType
from pathlib import Path
from mlflow.utils.file_utils import local_file_uri_to_path

def vector_its(iteration, probability, p_target = 0.99):
    epsilon = 1e-6
    its = (iteration + 1) * np.log(1 - p_target) / np.log(1 - probability+epsilon)
    if len(np.where(probability == 1)[0])>0:
        its[np.where(probability == 1)[0]] = np.where(probability >=p_target)[0][0]*np.ones(len(np.where(probability == 1)[0]))
    return its

def generate_report_hpo(tracking_uri,experiment_name):
    mlflow.set_tracking_uri(tracking_uri)
    all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
    runs = mlflow.search_runs(experiment_ids=all_experiments, run_view_type=ViewType.ALL)
    runs_id = runs['run_id']
    data_frames: typing.List[pd.DataFrame] = []
    for run_id in runs_id:        
        mlflow_run_id = run_id
        mlflow_run = mlflow.get_run(mlflow_run_id)
        artifact_path: Path = Path(local_file_uri_to_path(mlflow_run.info.artifact_uri))
        experiment_uri = artifact_path / experiment_name
        if os.path.isdir(experiment_uri):
            experiment = tune.ExperimentAnalysis((experiment_uri).as_posix())
            data_frames.append(experiment.dataframe())
    
    result = pd.concat(data_frames, axis=0, ignore_index=True)#[['config/instance','config/noise','tts','max_flips_opt']]
    return result

    