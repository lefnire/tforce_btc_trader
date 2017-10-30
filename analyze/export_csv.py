import pandas as pd
from data import conn

IMPUTE = True

runs = conn.execute('select * from runs').fetchall()
runs = [{**r.hypers, 'target': r.reward} for r in runs]
runs = pd.DataFrame(runs)

# Handle Nulls
if IMPUTE:
    runs['dropout'].fillna(0., inplace=True)
    runs['gae_lambda'].fillna(1., inplace=True)

runs.to_csv('runs.csv', index=False)
