import string

from tempfile import mkdtemp
import dowhy.datasets
from networkx.algorithms.structuralholes import effective_size
import numpy as np
import pandas as pd
import pytest
from actableai.causal.params import (
    LinearDMLCategoricalTreatmentParams,
    LinearDMLCategoricalTreatmentAGParams,
    LinearDMLSingleContTreatmentParams,
    LinearDMLSingleContTreatmentAGParams,
    LinearDMLSingleBinaryTreatmentParams,
    LinearDMLSingleBinaryTreatmentAGParams,
    SparseLinearDMLSingleBinaryTreatmentParams,
    SparseLinearDMLSingleContTreatmentParams,
)

from actableai.data_validation.base import *
from actableai.data_validation.checkers import *
from actableai.tasks.causal_inference import (
    LogCategoricalOutcomeNotAllowed,
    LogCategoricalTreatmentNotAllowed,
    infer_causal,
)
from actableai.causal.tree_utils import make_pretty_tree
from actableai.utils.testing import unittest_hyperparameters
from actableai.data_validation.base import (
    CAUSAL_INFERENCE_CATEGORICAL_MINIMUM_TREATMENT,
)


@pytest.fixture
def simple_linear_dataset():
    np.random.seed(123)
    data = dowhy.datasets.linear_dataset(
        beta=10,
        num_common_causes=5,
        num_instruments=2,
        num_effect_modifiers=1,
        num_samples=40,
        treatment_is_binary=True,
        num_discrete_common_causes=1,
    )
    pd_table = data["df"]
    treatments = data["treatment_name"]
    outcomes = [data["outcome_name"]]
    effect_modifiers = data["effect_modifier_names"]
    common_causes = data["common_causes_names"]
    return pd_table, treatments, outcomes, effect_modifiers, common_causes


@pytest.fixture
def cat_em_dataset():
    np.random.seed(123)
    data = dowhy.datasets.linear_dataset(
        beta=10,
        num_common_causes=5,
        num_instruments=2,
        num_effect_modifiers=1,
        num_samples=100,
        treatment_is_binary=False,
        num_discrete_effect_modifiers=1,
    )
    pd_table = data["df"]
    treatments = data["treatment_name"]
    outcomes = [data["outcome_name"]]
    effect_modifiers = data["effect_modifier_names"]
    common_causes = data["common_causes_names"]

    values = sorted(pd_table[effect_modifiers[0]].unique())
    value_map = {v: string.ascii_letters[v] for v in values}
    pd_table[effect_modifiers[0]] = pd_table[effect_modifiers[0]].map(value_map)
    pd_table = pd_table.astype({effect_modifiers[0]: str})
    return pd_table, treatments, outcomes, effect_modifiers, common_causes


@pytest.fixture
def single_cont_treatment_dataset():
    np.random.seed(123)

    # Example 1.1,
    # source: https://github.com/microsoft/EconML/blob/master/notebooks/Double%20Machine%20Learning%20Examples.ipynb

    # Treatment effect function
    def exp_te(x):
        return np.exp(2 * x[0])

    # DGP constants
    np.random.seed(123)
    n = 200
    n_w = 30
    support_size = 5
    n_x = 1
    # Outcome support
    support_Y = np.random.choice(np.arange(n_w), size=support_size, replace=False)
    coefs_Y = np.random.uniform(0, 1, size=support_size)
    epsilon_sample = lambda n: np.random.uniform(-1, 1, size=n)
    # Treatment support
    support_T = support_Y
    coefs_T = np.random.uniform(0, 1, size=support_size)
    eta_sample = lambda n: np.random.uniform(-1, 1, size=n)

    # Generate controls, covariates, treatments and outcomes
    W = np.random.normal(0, 1, size=(n, n_w))
    X = np.random.uniform(0, 1, size=(n, n_x))
    # Heterogeneous treatment effects
    TE = np.array([exp_te(x_i) for x_i in X])
    T = np.dot(W[:, support_T], coefs_T) + eta_sample(n)
    Y = TE * T + np.dot(W[:, support_Y], coefs_Y) + epsilon_sample(n)

    df = pd.DataFrame(
        {
            "Y": Y,
            "T": T,
        }
    )
    for c in range(X.shape[1]):
        df["X" + str(c)] = X[:, c]
    for c in range(W.shape[1]):
        df["W" + str(c)] = W[:, c]

    pd_table = df
    treatments = ["T"]
    outcomes = ["Y"]
    effective_modifiers = ["X0"]
    common_causes = ["W" + str(c) for c in range(W.shape[1])]
    return pd_table, treatments, outcomes, effective_modifiers, common_causes


@pytest.fixture
def single_binary_treatment_dataset():
    # Example 2.1,
    # source: https://github.com/microsoft/EconML/blob/master/notebooks/Double%20Machine%20Learning%20Examples.ipynb
    # Treatment effect function
    def exp_te(x):
        return np.exp(2 * x[0])  # DGP constants

    np.random.seed(123)
    n = 100
    n_w = 30
    support_size = 5
    n_x = 4
    # Outcome support
    support_Y = np.random.choice(range(n_w), size=support_size, replace=False)
    coefs_Y = np.random.uniform(0, 1, size=support_size)
    epsilon_sample = lambda n: np.random.uniform(-1, 1, size=n)
    # Treatment support
    support_T = support_Y
    coefs_T = np.random.uniform(0, 1, size=support_size)
    eta_sample = lambda n: np.random.uniform(-1, 1, size=n)

    # Generate controls, covariates, treatments and outcomes
    W = np.random.normal(0, 1, size=(n, n_w))
    X = np.random.uniform(0, 1, size=(n, n_x))
    # Heterogeneous treatment effects
    TE = np.array([exp_te(x_i) for x_i in X])
    # Define treatment
    log_odds = np.dot(W[:, support_T], coefs_T) + eta_sample(n)
    T_sigmoid = 1 / (1 + np.exp(-log_odds))
    T = np.array([np.random.binomial(1, p) for p in T_sigmoid])
    # Define the outcome
    Y = TE * T + np.dot(W[:, support_Y], coefs_Y) + epsilon_sample(n)

    df = pd.DataFrame(
        {
            "Y": Y,
            "T": T,
        }
    )
    for c in range(X.shape[1]):
        df["X" + str(c)] = X[:, c]
    for c in range(W.shape[1]):
        df["W" + str(c)] = W[:, c]

    pd_table = df
    treatments = ["T"]
    outcomes = ["Y"]
    effect_modifiers = ["X0"]
    common_causes = ["W" + str(c) for c in range(W.shape[1])]
    return pd_table, treatments, outcomes, effect_modifiers, common_causes


def treatment_values_filler(
    pd_table: pd.DataFrame, treatment: str = "v0"
) -> pd.DataFrame:
    # Ensure there is enough treatment control values
    chosen_idx = np.random.choice(
        pd_table.shape[0],
        size=CAUSAL_INFERENCE_CATEGORICAL_MINIMUM_TREATMENT * 2,
        replace=False,
    )
    true_idx, false_idx = np.array_split(chosen_idx, 2)
    for idx in true_idx:
        pd_table.at[idx, treatment] = True
    for idx in false_idx:
        pd_table.at[idx, treatment] = False
    return pd_table


class TestRemoteCausal:
    def test_linear_dataset(self, init_ray):
        np.random.seed(123)
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=5,
            num_instruments=2,
            num_effect_modifiers=1,
            num_samples=500,
            treatment_is_binary=True,
            num_discrete_common_causes=1,
        )
        pd_table = data["df"]

        model_params = [LinearDMLSingleBinaryTreatmentParams(min_samples_leaf=5)]

        r = infer_causal(
            pd_table,
            treatments=data["treatment_name"],
            outcomes=[data["outcome_name"]],
            effect_modifiers=data["effect_modifier_names"],
            common_causes=data["common_causes_names"],
            model_params=model_params,
            ag_hyperparameters=unittest_hyperparameters(),
            cv=2,
            mc_iters=2,
        )
        assert r["status"] == "SUCCESS"
        assert all(
            [
                k in r["data"]
                for k in [
                    "effect",
                    "controls",
                    "causal_graph_dot",
                    "tree_interpreter_dot",
                    "refutation_results",
                ]
            ]
        )

    def test_single_cont_treatment(self, single_cont_treatment_dataset, init_ray):
        (
            pd_table,
            treatments,
            outcomes,
            effect_modifiers,
            common_causes,
        ) = single_cont_treatment_dataset

        results = infer_causal(
            pd_table=pd_table,
            treatments=treatments,
            outcomes=outcomes,
            effect_modifiers=effect_modifiers,
            common_causes=common_causes,
            ag_hyperparameters=unittest_hyperparameters(),
            feature_importance=True,
            cv=2,
            mc_iters=2,
        )
        assert results["status"] == "SUCCESS"
        assert all(
            [
                k in results["data"]
                for k in [
                    "effect",
                    "controls",
                    "causal_graph_dot",
                    "tree_interpreter_dot",
                    "refutation_results",
                    "Y_res",
                    "T_res",
                    "X",
                    "model_t_scores",
                    "model_y_scores",
                    "model_t_feature_importances",
                    "model_y_feature_importances",
                ]
            ]
        )

    def test_log_treatment(self, single_cont_treatment_dataset, init_ray):
        (
            pd_table,
            treatments,
            outcomes,
            effect_modifiers,
            common_causes,
        ) = single_cont_treatment_dataset

        model_params = [LinearDMLSingleContTreatmentParams()]

        results = infer_causal(
            pd_table=pd_table,
            treatments=treatments,
            outcomes=outcomes,
            effect_modifiers=effect_modifiers,
            common_causes=common_causes,
            log_treatment=True,
            model_params=model_params,
            ag_hyperparameters=unittest_hyperparameters(),
            cv=2,
            mc_iters=2,
        )
        assert results["status"] == "SUCCESS"
        assert all(
            [
                k in results["data"]
                for k in [
                    "effect",
                    "controls",
                    "causal_graph_dot",
                    "tree_interpreter_dot",
                    "refutation_results",
                ]
            ]
        )

    def test_log_cat_outcome(self, single_cont_treatment_dataset, init_ray):
        (
            pd_table,
            treatments,
            outcomes,
            effect_modifiers,
            common_causes,
        ) = single_cont_treatment_dataset
        pd_table[outcomes] = np.random.choice(
            ["A", "B", "C"], size=len(pd_table), replace=True
        )

        model_params = [LinearDMLSingleContTreatmentParams()]
        with pytest.raises(LogCategoricalOutcomeNotAllowed):
            results = infer_causal(
                pd_table=pd_table,
                treatments=treatments,
                outcomes=outcomes,
                effect_modifiers=effect_modifiers,
                common_causes=common_causes,
                log_outcome=True,
                model_params=model_params,
                ag_hyperparameters=unittest_hyperparameters(),
                cv=2,
                mc_iters=2,
            )

    def test_single_binary_treatment(self, single_binary_treatment_dataset, init_ray):
        (
            pd_table,
            treatments,
            outcomes,
            effect_modifiers,
            common_causes,
        ) = single_binary_treatment_dataset

        model_params = [
            LinearDMLSingleBinaryTreatmentAGParams(
                label_t=treatments[0],
                label_y=outcomes[0],
                model_directory=mkdtemp(),
                presets="medium_quality_faster_train",
            )
        ]

        results = infer_causal(
            pd_table=pd_table,
            treatments=treatments,
            outcomes=outcomes,
            effect_modifiers=effect_modifiers,
            common_causes=common_causes,
            model_params=model_params,
            ag_hyperparameters=unittest_hyperparameters(),
            ag_presets="medium_quality_faster_train",
            feature_importance=True,
            cv=2,
            mc_iters=2,
        )
        assert results["status"] == "SUCCESS"
        assert all(
            [
                k in results["data"]
                for k in [
                    "effect",
                    "controls",
                    "causal_graph_dot",
                    "tree_interpreter_dot",
                    "refutation_results",
                    "Y_res",
                    "T_res",
                    "X",
                    "model_t_scores",
                    "model_y_scores",
                    "model_t_feature_importances",
                    "model_y_feature_importances",
                ]
            ]
        )

    def test_boolean_treatment(self, single_binary_treatment_dataset, init_ray):
        (
            pd_table,
            treatments,
            outcomes,
            effect_modifiers,
            common_causes,
        ) = single_binary_treatment_dataset

        pd_table[treatments[0]] = pd_table[treatments[0]].apply(
            lambda x: True if x == 1 else False
        )
        model_params = [
            LinearDMLSingleBinaryTreatmentAGParams(
                label_t=treatments[0],
                label_y=outcomes[0],
                model_directory=mkdtemp(),
                presets="medium_quality_faster_train",
            )
        ]

        results = infer_causal(
            pd_table=pd_table,
            treatments=treatments,
            outcomes=outcomes,
            effect_modifiers=effect_modifiers,
            common_causes=common_causes,
            model_params=model_params,
            ag_hyperparameters=unittest_hyperparameters(),
            ag_presets="medium_quality_faster_train",
            feature_importance=True,
            cv=2,
            mc_iters=2,
        )
        assert results["status"] == "SUCCESS"
        assert all(
            [
                k in results["data"]
                for k in [
                    "effect",
                    "controls",
                    "causal_graph_dot",
                    "tree_interpreter_dot",
                    "refutation_results",
                    "Y_res",
                    "T_res",
                    "X",
                    "model_t_scores",
                    "model_y_scores",
                    "model_t_feature_importances",
                    "model_y_feature_importances",
                ]
            ]
        )

    def test_missing_treatment(self, simple_linear_dataset, init_ray):
        np.random.seed(123)
        (
            pd_table,
            treatments,
            outcomes,
            effect_modifiers,
            common_causes,
        ) = simple_linear_dataset
        pd_table_no_treatment = pd_table.drop(columns=treatments)

        model_params = [LinearDMLSingleBinaryTreatmentParams()]

        results = infer_causal(
            pd_table=pd_table_no_treatment,
            treatments=treatments,
            outcomes=outcomes,
            effect_modifiers=effect_modifiers,
            common_causes=common_causes,
            model_params=model_params,
            ag_hyperparameters=unittest_hyperparameters(),
        )
        assert results["status"] == "FAILURE"
        assert len(results["validations"]) > 0
        assert results["validations"][0]["name"] == "ColumnsExistChecker"
        assert results["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_missing_outcome(self, simple_linear_dataset, init_ray):
        np.random.seed(123)
        (
            pd_table,
            treatments,
            outcomes,
            effect_modifiers,
            common_causes,
        ) = simple_linear_dataset
        pd_table_no_outcome = pd_table.drop(columns=outcomes)

        model_params = [LinearDMLSingleBinaryTreatmentParams()]

        results = infer_causal(
            pd_table=pd_table_no_outcome,
            treatments=treatments,
            outcomes=outcomes,
            effect_modifiers=effect_modifiers,
            common_causes=common_causes,
            model_params=model_params,
            ag_hyperparameters=unittest_hyperparameters(),
            cv=2,
            mc_iters=2,
        )
        assert results["status"] == "FAILURE"
        assert len(results["validations"]) > 0
        assert results["validations"][0]["name"] == "ColumnsExistChecker"
        assert results["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_no_effect_modifiers(self, simple_linear_dataset, init_ray):
        np.random.seed(123)
        (
            pd_table,
            treatments,
            outcomes,
            effect_modifiers,
            common_causes,
        ) = simple_linear_dataset

        # Ensure there is enough treatment control values
        pd_table = treatment_values_filler(pd_table=pd_table)

        model_params = [LinearDMLSingleBinaryTreatmentParams(min_samples_leaf=5)]

        results = infer_causal(
            pd_table=pd_table,
            treatments=treatments,
            outcomes=outcomes,
            effect_modifiers=None,
            common_causes=common_causes,
            model_params=model_params,
            ag_hyperparameters=unittest_hyperparameters(),
        )
        assert results["status"] == "SUCCESS"
        assert all(
            [
                k in results["data"]
                for k in [
                    "effect",
                    "controls",
                    "causal_graph_dot",
                    "tree_interpreter_dot",
                    "refutation_results",
                ]
            ]
        )

    def test_missing_effect_modifiers(self, simple_linear_dataset, init_ray):
        np.random.seed(123)
        (
            pd_table,
            treatments,
            outcomes,
            effect_modifiers,
            common_causes,
        ) = simple_linear_dataset
        pd_table_missing_effect_modifier = pd_table.drop(columns=["X0"])

        model_params = [LinearDMLSingleBinaryTreatmentParams(min_samples_leaf=5)]

        results = infer_causal(
            pd_table=pd_table_missing_effect_modifier,
            treatments=treatments,
            outcomes=outcomes,
            common_causes=common_causes,
            effect_modifiers=effect_modifiers,
            model_params=model_params,
            ag_hyperparameters=unittest_hyperparameters(),
            cv=2,
            mc_iters=2,
        )
        assert results["status"] == "FAILURE"
        assert len(results["validations"]) > 0
        assert results["validations"][0]["name"] == "ColumnsExistChecker"
        assert results["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_no_common_causes(self, simple_linear_dataset, init_ray):
        np.random.seed(123)
        (
            pd_table,
            treatments,
            outcomes,
            effect_modifiers,
            common_causes,
        ) = simple_linear_dataset

        # Ensure there is enough treatment control values
        pd_table = treatment_values_filler(pd_table=pd_table)

        model_params = [LinearDMLSingleBinaryTreatmentParams(min_samples_leaf=5)]

        results = infer_causal(
            pd_table=pd_table,
            treatments=treatments,
            outcomes=outcomes,
            effect_modifiers=effect_modifiers,
            common_causes=None,
            model_params=model_params,
            ag_hyperparameters=unittest_hyperparameters(),
        )
        assert results["status"] == "SUCCESS"
        assert all(
            [
                k in results["data"]
                for k in [
                    "effect",
                    "controls",
                    "causal_graph_dot",
                    "tree_interpreter_dot",
                    "refutation_results",
                ]
            ]
        )

    def test_missing_common_causes(self, simple_linear_dataset, init_ray):
        np.random.seed(123)
        (
            pd_table,
            treatments,
            outcomes,
            effect_modifiers,
            common_causes,
        ) = simple_linear_dataset
        pd_table_missing_common_cause = pd_table.drop(columns=["W0"])

        model_params = [LinearDMLSingleBinaryTreatmentParams(min_samples_leaf=5)]

        results = infer_causal(
            pd_table=pd_table_missing_common_cause,
            treatments=treatments,
            outcomes=outcomes,
            common_causes=common_causes,
            effect_modifiers=effect_modifiers,
            model_params=model_params,
            ag_hyperparameters=unittest_hyperparameters(),
            cv=2,
            mc_iters=2,
        )
        assert results["status"] == "FAILURE"
        assert len(results["validations"]) > 0
        assert results["validations"][0]["name"] == "ColumnsExistChecker"
        assert results["validations"][0]["level"] == CheckLevels.CRITICAL

    def test_no_effect_modifiers_and_common_causes(
        self, simple_linear_dataset, init_ray
    ):
        np.random.seed(123)
        (
            pd_table,
            treatments,
            outcomes,
            effect_modifiers,
            common_causes,
        ) = simple_linear_dataset

        # Ensure there is enough treatment control values
        pd_table = treatment_values_filler(pd_table=pd_table)

        model_params = [LinearDMLSingleBinaryTreatmentParams(min_samples_leaf=5)]

        results = infer_causal(
            pd_table=pd_table,
            treatments=treatments,
            outcomes=outcomes,
            effect_modifiers=None,
            common_causes=None,
            model_params=model_params,
            ag_hyperparameters=unittest_hyperparameters(),
            cv=2,
            mc_iters=2,
        )
        assert results["status"] == "SUCCESS"
        assert all(
            [
                k in results["data"]
                for k in [
                    "effect",
                    "controls",
                    "causal_graph_dot",
                    "tree_interpreter_dot",
                    "refutation_results",
                ]
            ]
        )

    def test_binary_outcome(self, simple_linear_dataset, init_ray):
        np.random.seed(123)
        (
            pd_table,
            treatments,
            outcomes,
            effect_modifiers,
            common_causes,
        ) = simple_linear_dataset

        model_params = [LinearDMLSingleBinaryTreatmentParams(min_samples_leaf=5)]
        pd_table[outcomes] = (pd_table[outcomes] >= pd_table[outcomes].median()).astype(
            str
        )

        # Ensure there is enough treatment control values
        pd_table = treatment_values_filler(pd_table=pd_table)

        results = infer_causal(
            pd_table=pd_table,
            treatments=treatments,
            outcomes=outcomes,
            positive_outcome_value="True",
            effect_modifiers=None,
            common_causes=None,
            model_params=model_params,
            ag_hyperparameters=unittest_hyperparameters(),
            cv=2,
            mc_iters=2,
        )
        assert results["status"] == "SUCCESS"
        assert all(
            [
                k in results["data"]
                for k in [
                    "effect",
                    "controls",
                    "causal_graph_dot",
                    "tree_interpreter_dot",
                    "refutation_results",
                ]
            ]
        )

    def test_categorical_effect_modifier(self, cat_em_dataset, init_ray):
        np.random.seed(123)
        (
            pd_table,
            treatments,
            outcomes,
            effect_modifiers,
            common_causes,
        ) = cat_em_dataset

        r = infer_causal(
            pd_table,
            treatments=treatments,
            outcomes=outcomes,
            effect_modifiers=effect_modifiers,
            common_causes=common_causes,
            ag_hyperparameters=unittest_hyperparameters(),
            cv=2,
            mc_iters=2,
        )
        assert r["status"] == "SUCCESS"
        assert all(
            [
                k in r["data"]
                for k in [
                    "effect",
                    "controls",
                    "causal_graph_dot",
                    "tree_interpreter_dot",
                    "refutation_results",
                ]
            ]
        )

    def test_categorical_outcome(self):
        df = pd.DataFrame(
            {
                "t": [2, 2, 2, 2, 2, None, 3, 3, 4, 4] * 10,
                "y": ["+", "-", "+", "-", "+", "-", "+", "-", "+", "-"] * 10,
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
                "w": ["a", "a", "a", "a", "a", "b", "b", None, "b", "b"] * 10,
            }
        )
        r = infer_causal(
            df,
            treatments=["t"],
            outcomes=["y"],
            effect_modifiers=["x"],
            common_causes=["w"],
            positive_outcome_value="+",
            ag_hyperparameters=unittest_hyperparameters(),
            cv=2,
            mc_iters=2,
        )
        assert r["status"] == "SUCCESS"
        assert all(
            [
                k in r["data"]
                for k in [
                    "effect",
                    "controls",
                    "causal_graph_dot",
                    "tree_interpreter_dot",
                    "refutation_results",
                ]
            ]
        )

    def test_make_pretty_tree(self):
        tree_interpreter_dot = "\n".join(
            [
                "digraph Tree {",
                'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;',
                "graph [ranksep=equally, splines=polyline] ;",
                "edge [fontname=helvetica] ;",
                '0 [label="location <= 0.5\\nsamples = 989\\nCATE mean\\n1.01\\nCATE std\\n0.023", fillcolor="#b2d3b8"] ;',
                '1 [label="samples = 446\\nCATE mean\\n0.986 (0.974, 0.998)\\nCATE std\\n0.0", fillcolor="#ffffff"] ;',
                '0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;',
                '2 [label="location <= 1.5\\nsamples = 543\\nCATE mean\\n1.029\\nCATE std\\n0.013", fillcolor="#72b07e"] ;',
                '0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;',
                '3 [label="samples = 206\\nCATE mean\\n1.012 (1.002, 1.023)\\nCATE std\\n0.0", fillcolor="#a8ceb0"] ;',
                "2 -> 3 ;",
                '4 [label="samples = 337\\nCATE mean\\n1.039 (1.021, 1.057)\\nCATE std\\n0.0", fillcolor="#519d60"] ;',
                "2 -> 4 ;",
                "{rank=same ; 0} ;",
                "{rank=same ; 2} ;",
                "{rank=same ; 1; 3; 4} ;",
                "}",
            ]
        )
        final_dot = "\n".join(
            [
                "digraph Tree {",
                'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;',
                "graph [ranksep=equally, splines=polyline] ;",
                "edge [fontname=helvetica] ;",
                '0 [label="location is great\\nsamples = 989\\nAverage effect\\n1.01", fillcolor="#b2d3b8"] ;',
                '1 [label="samples = 446\\nAverage effect (CI lower, CI upper)\\n0.986 (0.974, 0.998)", fillcolor="#ffffff"] ;',
                '0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;',
                '2 [label="location is good\\nsamples = 543\\nAverage effect\\n1.029", fillcolor="#72b07e"] ;',
                '0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;',
                '3 [label="samples = 206\\nAverage effect (CI lower, CI upper)\\n1.012 (1.002, 1.023)", fillcolor="#a8ceb0"] ;',
                "2 -> 3 ;",
                '4 [label="samples = 337\\nAverage effect (CI lower, CI upper)\\n1.039 (1.021, 1.057)", fillcolor="#519d60"] ;',
                "2 -> 4 ;",
                "{rank=same ; 0} ;",
                "{rank=same ; 2} ;",
                "{rank=same ; 1; 3; 4} ;",
                "}",
            ]
        )
        cat_name = "location"
        cat_vals = {0: "great", 1: "good", 2: "poor"}
        pretty_tree = make_pretty_tree(tree_interpreter_dot, [cat_name], [cat_vals])
        assert pretty_tree == final_dot

    def test_not_enough_values(self, simple_linear_dataset, init_ray):
        np.random.seed(123)
        (
            pd_table,
            treatments,
            outcomes,
            effect_modifiers,
            common_causes,
        ) = simple_linear_dataset
        pd_table = (
            pd_table[pd_table["v0"]]
            .head(20)
            .append(pd_table[pd_table["v0"] == False].head(10))
        )

        model_params = [LinearDMLSingleBinaryTreatmentParams(min_samples_leaf=5)]

        results = infer_causal(
            pd_table=pd_table,
            treatments=treatments,
            outcomes=outcomes,
            effect_modifiers=None,
            common_causes=None,
            model_params=model_params,
            ag_hyperparameters=unittest_hyperparameters(),
            cv=2,
            mc_iters=2,
        )
        assert results["status"] == "FAILURE"
        assert len(results["validations"]) > 0
        assert "InsufficientCategoricalRows" in [
            results["validations"][i]["name"]
            for i in range(len(results["validations"]))
        ]
        assert CheckLevels.CRITICAL in [
            results["validations"][i]["level"]
            for i in range(len(results["validations"]))
        ]
