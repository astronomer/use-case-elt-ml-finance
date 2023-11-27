"""
### Engineer features and train machine learning models in parallel on Stripe data

This DAG retrieves Stripe API like data from a relational database, creates 
machine learning features and trains sci-kit learn machine learning models in 
parallel, using dynamic task mapping. Finally, plots of the model results are 
created.

You will need to specify a database connection to a database supported by the
Astro Python SDK, for example `postgres_default` and an AWS connection `aws_default`.
"""

from airflow.decorators import dag, task
from pendulum import datetime
from astro import sql as aql
from astro.sql.table import Table, Metadata
from airflow.configuration import conf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso
import numpy as np
import os

AWS_CONN_ID = "aws_default"
DATA_BUCKET_NAME = "finance-elt-ml-data"
ENVIRONMENT = os.getenv("MY_ENVIRONMENT", "local")

if ENVIRONMENT == "local":
    DB_CONN_ID = "postgres_default"
    DB_SCHEMA = "tmp_astro"
if ENVIRONMENT == "prod":
    DB_CONN_ID = "snowflake_default"
    DB_SCHEMA = "TAMARAFINGERLIN"


@aql.dataframe()
def feature_eng(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    y = df["avg_amount_captured"]
    X = df.drop(columns=["avg_amount_captured"])[
        [
            "customer_satisfaction_speed",
            "customer_satisfaction_product",
            "customer_satisfaction_service",
            "product_type",
        ]
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_train_df = pd.DataFrame({"avg_amount_captured": y_train})
    y_train_df.index = X_train.index

    y_test_df = pd.DataFrame({"avg_amount_captured": y_test})
    y_test_df.index = X_test.index

    numeric_columns = [
        "customer_satisfaction_speed",
        "customer_satisfaction_product",
        "customer_satisfaction_service",
    ]

    scaler = StandardScaler()
    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

    onehot_encoder = OneHotEncoder(sparse=False, drop="first")
    onehot_encoder.fit(X_train[["product_type"]])

    product_type_train = onehot_encoder.transform(X_train[["product_type"]])
    product_type_test = onehot_encoder.transform(X_test[["product_type"]])
    product_type_df_train = pd.DataFrame(
        product_type_train,
        columns=onehot_encoder.get_feature_names_out(["product_type"]),
    )
    product_type_df_test = pd.DataFrame(
        product_type_test,
        columns=onehot_encoder.get_feature_names_out(["product_type"]),
    )

    index_X_train = X_train.index
    index_X_test = X_test.index

    X_train = pd.concat(
        [X_train.reset_index(drop=True), product_type_df_train.reset_index(drop=True)],
        axis=1,
    ).drop(columns=["product_type"])
    X_test = pd.concat(
        [X_test.reset_index(drop=True), product_type_df_test.reset_index(drop=True)],
        axis=1,
    ).drop(columns=["product_type"])

    X_train.index = index_X_train
    X_test.index = index_X_test

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_df": y_train_df,
        "y_test_df": y_test_df,
    }


def train_model(feature_eng_table, model_class, hyper_parameters):
    from sklearn.metrics import r2_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, RidgeCV, Lasso
    import pandas as pd

    print(f"Training model: {model_class.__name__}")

    X_train = feature_eng_table["X_train"]
    X_test = feature_eng_table["X_test"]
    y_train = feature_eng_table["y_train_df"]
    y_test = feature_eng_table["y_test_df"]

    y_train.dropna(axis=0, inplace=True)
    y_test.dropna(axis=0, inplace=True)

    y_train = y_train["avg_amount_captured"]
    y_test = y_test["avg_amount_captured"]

    X_train.dropna(axis=0, inplace=True)
    X_test.dropna(axis=0, inplace=True)

    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]

    model = model_class(**hyper_parameters)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    if hasattr(model, "feature_importances_"):
        feature_imp = pd.DataFrame(
            {
                "Feature": X_train.columns,
                "Importance": model.feature_importances_,
            }
        )
        feature_imp_coef = feature_imp.sort_values(by="Importance", ascending=False)
        print("Feature Importances:")
        print(feature_imp_coef)
    elif hasattr(model, "coef_"):
        feature_coef = pd.DataFrame(
            {"Feature": X_train.columns, "Coefficient": np.ravel(model.coef_)}
        )
        feature_imp_coef = feature_coef.sort_values(by="Coefficient", ascending=False)
        print("Feature Coefficients:")
        print(feature_imp_coef)
    else:
        feature_imp_coef = None
        print("Model has no feature importances or coefficients.")

    print(f"R2 train: {r2_train}")
    print(f"R2 test: {r2_test}")

    y_train_df = y_train.to_frame()
    y_pred_train_df = pd.DataFrame(y_pred_train, columns=["y_pred_train"])
    y_test_df = y_test.to_frame()
    y_pred_test_df = pd.DataFrame(y_pred_test, columns=["y_pred_test"])

    return {
        "model_class_name": model_class.__name__,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "feature_imp_coef": feature_imp_coef,
        "y_train_df": y_train_df,
        "y_pred_train_df": y_pred_train_df,
        "y_test_df": y_test_df,
        "y_pred_test_df": y_pred_test_df,
    }


@dag(
    start_date=datetime(2023, 9, 1),
    schedule=[Table(conn_id=DB_CONN_ID, name="model_satisfaction")],
    catchup=False,
)
def finance_ml():
    feature_eng_table = feature_eng(
        Table(
            conn_id=DB_CONN_ID,
            metadata=Metadata(schema=DB_SCHEMA),
            name="model_satisfaction",
        ),
    )

    # @task(
    #     queue="machine-learning-tasks",
    # )
    # def train_model_task(feature_eng_table, model_class=None, hyper_parameters={}):
    #     model_results = train_model(
    #         feature_eng_table=feature_eng_table,
    #         model_class=model_class,
    #         hyper_parameters=hyper_parameters,
    #     )
    #     return model_results

    if ENVIRONMENT == "prod":
        # get the current Kubernetes namespace Airflow is running in
        namespace = conf.get("kubernetes", "NAMESPACE")

        # @task.kubernetes(
        #     image="devashishupadhyay/scikit-learn-docker",  # specify your model image here
        #     in_cluster=True,
        #     namespace=namespace,
        #     name="my_model_train_pod",
        #     get_logs=True,
        #     log_events_on_failure=True,
        #     do_xcom_push=True,
        #     queue="machine-learning-tasks",  # optional setting for Astro customers
        # )
        @task(
            queue="machine-learning-tasks",
        )
        def train_model_task(feature_eng_table, model_class=None, hyper_parameters={}):
            model_results = train_model(
                feature_eng_table=feature_eng_table,
                model_class=model_class,
                hyper_parameters=hyper_parameters,
            )
            return model_results

    elif ENVIRONMENT == "local":

        @task
        def train_model_task(feature_eng_table, model_class=None, hyper_parameters={}):
            model_results = train_model(
                feature_eng_table=feature_eng_table,
                model_class=model_class,
                hyper_parameters=hyper_parameters,
            )
            return model_results

    else:
        raise ValueError(f"Unknown environment: {ENVIRONMENT}")

    model_results = train_model_task.partial(
        feature_eng_table=feature_eng_table
    ).expand_kwargs(
        [
            {
                "model_class": RandomForestRegressor,
                "hyper_parameters": {"n_estimators": 2000},
            },
            {"model_class": LinearRegression},
            {
                "model_class": RidgeCV,
                "hyper_parameters": {"alphas": [0.1, 1.0, 10.0]},
            },
            {
                "model_class": Lasso,
                "hyper_parameters": {"alpha": 2.0},
            },
        ]
    )

    @task
    def plot_model_results(model_results):
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not os.path.exists("include/plots"):
            os.makedirs("include/plots")

        model_class_name = model_results["model_class_name"]
        y_train_df = model_results["y_train_df"]
        y_pred_train_df = model_results["y_pred_train_df"]
        y_test_df = model_results["y_test_df"]
        y_pred_test_df = model_results["y_pred_test_df"]
        r2_train = model_results["r2_train"]
        r2_test = model_results["r2_test"]

        y_train_df.reset_index(drop=True, inplace=True)
        y_pred_train_df.reset_index(drop=True, inplace=True)
        y_test_df.reset_index(drop=True, inplace=True)
        y_pred_test_df.reset_index(drop=True, inplace=True)

        test_comparison = pd.concat([y_test_df, y_pred_test_df], axis=1)
        test_comparison.columns = ["True", "Predicted"]

        train_comparison = pd.concat([y_train_df, y_pred_train_df], axis=1)
        train_comparison.columns = ["True", "Predicted"]

        sns.set_style("white")
        plt.rcParams["font.size"] = 12

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        sns.scatterplot(
            ax=axes[0],
            x="True",
            y="Predicted",
            data=train_comparison,
            color="black",
            marker="x",
        )
        axes[0].plot(
            [train_comparison["True"].min(), train_comparison["True"].max()],
            [train_comparison["True"].min(), train_comparison["True"].max()],
            "--",
            linewidth=1,
            color="red",
        )
        axes[0].grid(True, linestyle="--", linewidth=0.5)
        axes[0].set_title(f"Train Set: {model_class_name}")
        axes[0].text(0.1, 0.9, f"R2: {r2_train}", transform=axes[0].transAxes)

        sns.scatterplot(
            ax=axes[1],
            x="True",
            y="Predicted",
            data=test_comparison,
            color="black",
            marker="x",
        )
        axes[1].plot(
            [test_comparison["True"].min(), test_comparison["True"].max()],
            [test_comparison["True"].min(), test_comparison["True"].max()],
            "--",
            linewidth=1,
            color="red",
        )
        axes[1].grid(True, linestyle="--", linewidth=0.5)
        axes[1].set_title(f"Test Set: {model_class_name}")
        axes[1].text(0.1, 0.9, f"R2: {r2_test}", transform=axes[1].transAxes)

        fig.suptitle("Predicted vs True Values", fontsize=16)

        plt.tight_layout()

        if ENVIRONMENT == "local":
            plt.savefig(f"include/plots/{model_class_name}_plot_results.png")
        if ENVIRONMENT == "prod":
            import boto3

            plt.savefig(f"{model_class_name}_plot_results.png")

            s3 = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
            with open(f"{model_class_name}_plot_results.png", "rb") as data:
                s3.upload_fileobj(
                    data,
                    DATA_BUCKET_NAME,
                    "plots/" + f"{model_class_name}_plot_results.png",
                    ExtraArgs={"ContentType": "image/jpeg"},
                )
            os.remove(f"{model_class_name}_plot_results.png")

    plot_model_results.expand(model_results=model_results)


finance_ml()
