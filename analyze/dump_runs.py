"""
Dump runs to csv, to be scp'd to localhost for DecisionTree comparision. Needs to be in python file (1) to access
current cwd (can't use relative paths in psql) (2) to issue command as psql client, who has access to directory
"""
import os

PATH = f'{os.path.dirname(os.path.realpath(__file__))}/runs.csv'
# SELECT = "select hypers, reward_avg, rewards from runs where flag is null and array_length(rewards, 1)>250"
SELECT = "select hypers, reward_avg, rewards, agent from runs where flag is null"

if __name__ == '__main__':
    pg_copy = f"\COPY ({SELECT}) TO '{PATH}' DELIMITER ',' CSV HEADER;"
    os.system(f'psql kaggle -c "{pg_copy}"')
