from airflow.decorators import dag, task_group, task
from airflow import Dataset
from pendulum import datetime
from astronomer.providers.amazon.aws.sensors.s3 import S3KeySensorAsync
from astro import sql as aql
from astro.files import File
from astro.sql.table import Table
from astro.sql.operators.load_file import LoadFileOperator
from astro.constants import FileType


AWS_CONN_ID = "aws_default"
DB_CONN_ID = "postgres_default"
DATA_BUCKET_NAME = "finance-etl-ml-data"
POKE_INTERVAL = 2


@aql.transform()
def select_successful_charges(in_charge: Table) -> Table:
    return """
        SELECT 
            id,
            customer_id,
            amount_captured,
            payment_method_details_card_country
        FROM {{ in_charge }}
        WHERE
            status = 'succeeded' 
            AND outcome_network_status = 'approved_by_network' 
            AND paid = true;
    """


@aql.transform()
def avg_successful_per_us_customer(tmp_successful: Table) -> Table:
    return """
        SELECT
            customer_id,
            AVG(amount_captured) AS avg_amount_captured
        FROM {{ tmp_successful }}
        GROUP BY customer_id;
    """


@aql.transform()
def join_charge_satisfaction(
    tmp_avg_successful_per_us_customer: Table, in_satisfaction
) -> Table:
    return """
        SELECT
            s.customer_id,
            s.customer_satisfaction_speed,
            s.customer_satisfaction_product,
            s.customer_satisfaction_service,
            s.product_type,
            c.avg_amount_captured
        FROM {{ tmp_avg_successful_per_us_customer }} c 
        LEFT JOIN {{ in_satisfaction }} s
            ON s.customer_id = c.customer_id;
    """


@dag(
    start_date=datetime(2023, 9, 1),
    schedule="@daily",
    catchup=False,
)
def finance_etl():
    wait_for_ingest_charge = S3KeySensorAsync(
        task_id="wait_for_ingest_charge",
        poke_interval=POKE_INTERVAL,
        bucket_key=f"s3://{DATA_BUCKET_NAME}/charge/*.csv",
        wildcard_match=True,
        aws_conn_id=AWS_CONN_ID,
    )

    wait_for_ingest_satisfaction = S3KeySensorAsync(
        task_id="wait_for_ingest_satisfaction",
        poke_interval=POKE_INTERVAL,
        bucket_key=f"s3://{DATA_BUCKET_NAME}/satisfaction/*.csv",
        wildcard_match=True,
        aws_conn_id=AWS_CONN_ID,
    )

    @task
    def retrieve_input_files():
        return [
            {
                "input_file": File(
                    path=f"s3://{DATA_BUCKET_NAME}/charge",
                    conn_id=AWS_CONN_ID,
                    filetype=FileType.CSV,
                ),
                "output_table": Table(conn_id=DB_CONN_ID, name="in_charge"),
            },
            {
                "input_file": File(
                    path=f"s3://{DATA_BUCKET_NAME}/satisfaction",
                    conn_id=AWS_CONN_ID,
                    filetype=FileType.CSV,
                ),
                "output_table": Table(conn_id=DB_CONN_ID, name="in_satisfaction"),
            },
        ]

    input_files = retrieve_input_files()

    # call load and transformation tasks

    s3_to_db_glob = LoadFileOperator.partial(
        task_id="s3_to_db_glob",
    ).expand_kwargs(input_files)

    tmp_successful = select_successful_charges(
        Table(conn_id=DB_CONN_ID, name="in_charge")
    )
    s3_to_db_glob >> tmp_successful
    tmp_avg_successful_per_us_customer = avg_successful_per_us_customer(tmp_successful)
    join_charge_satisfaction(
        tmp_avg_successful_per_us_customer,
        Table(conn_id=DB_CONN_ID, name="in_satisfaction"),
        output_table=Table(conn_id=DB_CONN_ID, name="model_satisfaction"),
    )

    (
        [wait_for_ingest_charge, wait_for_ingest_satisfaction]
        >> input_files
        >> s3_to_db_glob
    )


finance_etl()
