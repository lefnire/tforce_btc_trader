import os, time

e = 0
while True:
    os.system(f'python run_a3c.py -e {e}')
    print(f'Running experiment {e}')
    time.sleep(60*15)
    os.system(f'tmux kill-session -t btc')
    e += 1