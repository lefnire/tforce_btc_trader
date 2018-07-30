from data import engine_runs
import pandas as pd
import numpy as np

with engine_runs.connect() as conn:
    conn.execute("""
    CREATE OR REPLACE FUNCTION array_avg(double precision[])
    RETURNS double precision AS $$
    SELECT avg(v) FROM unnest($1) g(v)
    $$ LANGUAGE sql;
    """)

    sql = """
    select id, ret, ordinality as date 
    from runs, unnest(returns) with ordinality ret
    where array_avg(returns[60:]) > 0
    """

    df = pd.read_sql(sql, conn)
    df['date'] = pd.Period('2015-01-01', freq='D') + df['date']
    df.to_csv('tmp.csv')