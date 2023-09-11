Run an integrated ELT and ML pipeline on Stripe data in Airflow
================================================================

This repository contains the DAG code used in the [Financial ELT and ML pipeline use case example](https://docs.astronomer.io/learn/use-case-elt-ml-finance). 

The DAGs in this repository use the following packages:

- [Scikit-learn](https://scikit-learn.org/stable/index.html)
- [Astronomer providers](https://registry.astronomer.io/providers/astronomer-providers/versions/latest)
- [Amazon Airflow provider](https://registry.astronomer.io/providers/apache-airflow-providers-amazon/versions/latest)
- [Astro Python SDK](https://registry.astronomer.io/providers/astro-sdk-python/versions/latest)

# How to use this repository

This section explains how to run this repository with Airflow. Note that you will need to copy the contents of the `.env_example` file to a newly created `.env` file. No external connections are necessary to run this repository locally, but you can add your own credentials in the file if you wish to connect to your tools.

Download the [Astro CLI](https://docs.astronomer.io/astro/cli/install-cli) to run Airflow locally in Docker. `astro` is the only package you will need to install locally.

1. Run `git clone https://github.com/astronomer/use_case_elt_ml_finance.git` on your computer to create a local clone of this repository.
2. Install the Astro CLI by following the steps in the [Astro CLI documentation](https://docs.astronomer.io/astro/cli/install-cli). Docker Desktop/Docker Engine is a prerequisite, but you don't need in-depth Docker knowledge to run Airflow with the Astro CLI.
3. Run `astro dev start` in your cloned repository.
4. After your Astro project has started. View the Airflow UI at `localhost:8080` and the MinIO UI at `localhost:9000`.
5. Unpause all DAGs to see the pipeline run, the `finance_elt` DAG has a first task that will wait for the data to land in MinIO before running the rest of the DAG. The data is generated and transferred to MinIO by the `in_finance_data` DAG.

## Resources

- [Run an integrated ELT and ML pipeline on Stripe data in Airflow use case](https://docs.astronomer.io/learn/use-case-elt-ml-finance).
- [Deferrable operators guide](https://docs.astronomer.io/learn/deferrable-operators).
- [Create dynamic Airflow tasks guide](https://docs.astronomer.io/learn/dynamic-tasks).
- [Datasets and data-aware scheduling in Airflow guide](https://docs.astronomer.io/learn/airflow-datasets).
- [Astro Python SDK tutorial](https://docs.astronomer.io/learn/astro-python-sdk).
- [Astro Python SDK documentation](https://astro-sdk-python.readthedocs.io/en/stable/index.html).
- [Astro Python SDK repository](https://github.com/astronomer/astro-sdk).