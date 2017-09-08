Install https://github.com/reinforceio/tensorforce

psql coins lefnire:lefnire

# Run the script
python tforce_a3c.py bitcoin -w 7 -D

# Check on the progress
tmux a -t bitcoin_async

# Kill the process
python tforce_a3c.py bitcoin -w 7 -D -K


# Check some graphs
jupyter notebook
