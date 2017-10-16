import os, time

e = 0
while True:
    os.system(f'python tforce/distributed.py -e {e}')
    print(f'Running experiment {e}')
    time.sleep(60*45)
    os.system(f'tmux kill-session -t btc')
    e += 1