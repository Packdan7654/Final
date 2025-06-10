import random

FACT_POOL = {
    0: {"Rodin", "1842"},
    1: {"Medieval", "sword"},
    2: {"Da Vinci", "1503"},
    # etc. for up to 20 actions
}

def get_facts_for_template(action_id):
    return FACT_POOL.get(action_id, set())
