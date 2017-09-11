Install https://github.com/reinforceio/tensorforce, TA-Lib

# Run populate for a few hours
python populate.py

# Run the script
python run_a3c.py bitcoin -w 7 -D

# Check on the progress
tmux a -t bitcoin_async

# Kill the process
python run_a3c.py bitcoin -w 7 -D -K


# Check some graphs
jupyter notebook
