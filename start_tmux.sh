#!/bin/bash

SESSION="picksim"

# Kill existing session if any
tmux kill-session -t $SESSION 2>/dev/null

# Window 0: run script1.py
tmux new-session -d -s $SESSION -n vision
tmux send-keys -t $SESSION:0 'python3 test_data.py' C-m

# Window 1: run script2.py
tmux new-window -t $SESSION:1 -n motion
tmux send-keys -t $SESSION:1 'python3 app.py' C-m

# Window 2: plain terminal for manual use
tmux new-window -t $SESSION:2 -n prompt

# Attach to session, landing in the prompt window
tmux select-window -t $SESSION:2
tmux attach -t $SESSION
