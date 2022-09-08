import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics,
                        component)
from kfp.v2 import compiler

@component(
    packages_to_install = [
        "pandas",
        "sklearn"
    ],
)
def get_data(
    dataset_train: Output[Dataset],
    dataset_test: Output[Dataset]

):
    from sklearn import datasets
    from sklearn.model_selection import train_test_split as tts
    import pandas as pd
    # import some data to play with

    data_raw = datasets.load_breast_cancer()
    data = pd.DataFrame(data_raw.data, columns=data_raw.feature_names)
    data["target"] = data_raw.target

    train, test = tts(data, test_size=0.3)

    train.to_csv(dataset_train.path)
    test.to_csv(dataset_test.path)


@component(
    packages_to_install = [
        "pandas",
        "sklearn",
        "xgboost"
    ],
)
def train_xgb_model(
    dataset: Input[Dataset],
    model_artifact: Output[Model]
):
    from xgboost import XGBClassifier
    import pandas as pd

    data = pd.read_csv(dataset.path)

    model = XGBClassifier(
        objective="binary:logistic"
    )
    model.fit(
        data.drop(columns=["target"]),
        data.target,
    )

    score = model.score(
        data.drop(columns=["target"]),
        data.target,
    )

    model_artifact.metadata["train_score"] = float(score)
    model_artifact.metadata["framework"] = "XGBoost"

    model.save_model(model_artifact.path)


@component(
    packages_to_install = [
        "pandas",
        "sklearn",
        "xgboost"
    ],
)
def eval_model(
    test_set: Input[Dataset],
    xgb_model: Input[Model],
    metrics: Output[ClassificationMetrics],
    smetrics: Output[Metrics]
):
    from xgboost import XGBClassifier
    import pandas as pd

    data = pd.read_csv(test_set.path)
    model = XGBClassifier()
    model.load_model(xgb_model.path)

    score = model.score(
        data.drop(columns=["target"]),
        data.target,
    )

    from sklearn.metrics import roc_curve
    y_scores =  model.predict_proba(data.drop(columns=["target"]))[:, 1]
    fpr, tpr, thresholds = roc_curve(
         y_true=data.target.to_numpy(), y_score=y_scores, pos_label=True
    )
    metrics.log_roc_curve(fpr.tolist(), tpr.tolist(), thresholds.tolist())

    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(data.drop(columns=["target"]))

    metrics.log_confusion_matrix(
       ["False", "True"],
       confusion_matrix(
           data.target, y_pred
       ).tolist(),  # .tolist() to convert np array to list.
    )

    xgb_model.metadata["test_score"] = float(score)
    smetrics.log_metric("score", float(score))



@dsl.pipeline(
    name="pipeline-v2",
)

def pipeline():
    dataset_op : kfp.dsl.ContainerOp = get_data()
    train_op : kfp.dsl.ContainerOp = train_xgb_model(dataset_op.outputs["dataset_train"])
    eval_op : kfp.dsl.ContainerOp  = eval_model(
        test_set=dataset_op.outputs["dataset_test"],
        xgb_model=train_op.outputs["model_artifact"]
    )
    dataset_op.add_pod_annotation("scheduling.clops.clova.ai/machine-type", "pipe-cpu-half-v1")

if __name__ == '__main__':
    client = kfp.Client()

    pipeline_conf = kfp.dsl.PipelineConf()
    pipeline_conf.image_pull_policy = "Always"
    pipeline_conf.ttl_seconds_after_finished = 10

    kfp.compiler.Compiler().compile(pipeline, __file__ + '.yaml', pipeline_conf=pipeline_conf)

    client.create_run_from_pipeline_func(
        pipeline,
        mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE,
        enable_caching=False,
        arguments={},
        pipeline_conf=pipeline_conf
        )
