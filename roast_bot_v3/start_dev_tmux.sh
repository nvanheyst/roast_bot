#!/bin/bash

SESSION="robot_dev"

# Create session and first window (bringup)
tmux new-session -d -s $SESSION -n bringup

#
# === Window 1: Bringup ===
#

# Pane 0: pan-tilt bringup
tmux send-keys -t $SESSION:0.0 'cd' C-m

# Split vertically (creates pane 1 on the right)
tmux split-window -h -t $SESSION:0.0
tmux send-keys -t $SESSION:0.1 'ros2 launch foxglove_bridge foxglove_bridge_launch.xml' C-m


#
# === Window 2: Scripts ===
#

tmux new-window -t $SESSION:1 -n scripts

# Pane 0: oak_detector
tmux send-keys -t $SESSION:1.0 'cd ~/roast_bot_v3 && python3 media_generation_node.py' C-m

# Split vertically (creates pane 1 on the right)
tmux split-window -h -t $SESSION:1.0
tmux send-keys -t $SESSION:1.1 'source ~/venvs/zed-env/bin/activate && cd ~/roast_bot_v3 && python3 zed_tracking_node.py' C-m

#
# === Window 3: Scripts ===
#

tmux new-window -t $SESSION:2 -n scripts

# Pane 0: scripts
tmux send-keys -t $SESSION:2.0 'cd' C-m

#
# === Window 4: Scripts ===
#

tmux new-window -t $SESSION:3 -n scripts

# Pane 0: scripts
tmux send-keys -t $SESSION:3.0 'cd ~/roast_bot_media && npm install && npm start' C-m


# Attach to session
tmux select-window -t $SESSION:1
tmux select-pane -t $SESSION:1.0
tmux attach-session -t $SESSION
