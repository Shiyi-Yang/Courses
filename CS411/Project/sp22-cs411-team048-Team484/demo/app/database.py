"""Defines all the functions related to the database"""
from app import db

def fetch_product() -> dict:
    """Reads all items listed in the Product table

    :return: A list of dictionaries
    """

    conn = db.connect()
    query_results = conn.execute("Select * from Product;").fetchall()
    conn.close()
    product_lst = []
    for result in query_results:
        item = {
            "prod_id": result[0],
            "prod_name": result[5],
            "plu_code": result[1],
            "lifespan": result[2],
            "cal": result[3]
        }
        product_lst.append(item)

    return product_lst

def fetch_inventory(customer_id='27') -> dict:
    """Reads all items listed in the Product table

    :return: A list of dictionaries
    """
    pass
    conn = db.connect()
    query = "select * from InventoryList natural join Product where customerid='{}';".format(customer_id)
    query_results = conn.execute(query).fetchall()
    conn.close()
    inventory_lst = []
    for result in query_results:
        inv_item = {
            "inv_id": result[0],
            "item_name": result[6],
            "space": result[3],
            "amount": result[4],
            "unit": result[5],
            "exp_date": result[4]
        }
        inventory_lst.append(inv_item)

    return inventory_lst




def update_task_entry(task_id: int, text: str) -> None:
    """Updates task description based on given `task_id`

    Args:
        task_id (int): Targeted task_id
        text (str): Updated description

    Returns:
        None
    """

    conn = db.connect()
    query = 'Update tasks set task = "{}" where id = {};'.format(text, task_id)
    conn.execute(query)
    conn.close()


def update_status_entry(task_id: int, text: str) -> None:
    """Updates task status based on given `task_id`

    Args:
        task_id (int): Targeted task_id
        text (str): Updated status

    Returns:
        None
    """

    conn = db.connect()
    query = 'Update tasks set status = "{}" where id = {};'.format(text, task_id)
    conn.execute(query)
    conn.close()


def insert_new_task(text: str) ->  int:
    """Insert new task to todo table.

    Args:
        text (str): Task description

    Returns: The task ID for the inserted entry
    """

    conn = db.connect()
    query = 'Insert Into tasks (task, status) VALUES ("{}", "{}");'.format(
        text, "Todo")
    conn.execute(query)
    query_results = conn.execute("Select LAST_INSERT_ID();")
    query_results = [x for x in query_results]
    task_id = query_results[0][0]
    conn.close()

    return task_id


def remove_task_by_id(task_id: int) -> None:
    """ remove entries based on task ID """
    conn = db.connect()
    query = 'Delete From tasks where id={};'.format(task_id)
    conn.execute(query)
    conn.close()