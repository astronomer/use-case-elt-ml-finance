from airflow.decorators import dag, task
from pendulum import datetime
from astro import sql as aql
from astro.sql.table import Table, Metadata
from airflow.configuration import conf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso
import numpy as np

AWS_CONN_ID = "aws_default"
DB_CONN_ID = "postgres_default"
DB_SCHEMA = "tmp_astro"
DATA_BUCKET_NAME = "finance-etl-ml-data"
ENVIRONMENT = "local"


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

    print(f"Training model: {model_class.__name__}")

    X_train = feature_eng_table["X_train"]
    X_test = feature_eng_table["X_test"]
    y_train = feature_eng_table["y_train_df"]["avg_amount_captured"]
    y_test = feature_eng_table["y_test_df"]["avg_amount_captured"]

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

    return {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "feature_imp_coef": feature_imp_coef,
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

    if ENVIRONMENT == "prod":
        # get the current Kubernetes namespace Airflow is running in
        namespace = conf.get("kubernetes", "NAMESPACE")

        @task.kubernetes(
            image="python",
            in_cluster=True,
            namespace=namespace,
            name="my_model_train_pod",
            get_logs=True,
            log_events_on_failure=True,
            do_xcom_push=True,
        )
        def train_model_task(feature_eng_table, model_class, hyper_parameters={}):
            model_results = train_model(
                feature_eng_table=feature_eng_table,
                model_class=model_class,
                hyper_parameters=hyper_parameters,
            )
            return model_results

    elif ENVIRONMENT == "local":

        @task
        def train_model_task(feature_eng_table, model_class, hyper_parameters={}):
            model_results = train_model(
                feature_eng_table=feature_eng_table,
                model_class=model_class,
                hyper_parameters=hyper_parameters,
            )
            return model_results

    else:
        raise ValueError(f"Unknown environment: {ENVIRONMENT}")

    train_model_task.partial(feature_eng_table=feature_eng_table).expand_kwargs(
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


finance_ml()
