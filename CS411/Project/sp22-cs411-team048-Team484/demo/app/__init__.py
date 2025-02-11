"""Setup at app startup"""
import os
import sqlalchemy
from flask import Flask, render_template
from yaml import load, Loader
from app import routes



app = Flask(__name__)

def init_connection_engine():
    """ initialize database setup
    Takes in os variables from environment if on GCP
    Reads in local variables that will be ignored in public repository.
    Returns:
        pool -- a connection to GCP MySQL
    """


    # detect env local or gcp
    if os.environ.get('GAE_ENV') != 'standard':
        variables = load(open("/Users/candicechen/PycharmProjects/sp22-cs411-team048-Team484/demo/app/app.yaml"), Loader=Loader)
        # try:
        #     variables = load(open("app.yaml"), Loader=Loader)
        # except OSError as e:
        #     print("Make sure you have the app.yaml file setup")
        #     os.exit()

        env_variables = variables['env_variables']
        for var in env_variables:
            os.environ[var] = env_variables[var]

    pool = sqlalchemy.create_engine(
        sqlalchemy.engine.url.URL(
            drivername="mysql+pymysql",
            username=os.environ.get('MYSQL_USER'),
            password=os.environ.get('MYSQL_PASSWORD'),
            database=os.environ.get('MYSQL_DB'),
            host=os.environ.get('MYSQL_HOST')
        )
    )

    return pool



db = init_connection_engine()





# To prevent from using a blueprint, we use a cyclic import
# This also means that we need to place this import here
# pylint: disable=cyclic-import, wrong-import-position
from app import routes