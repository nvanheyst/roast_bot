# roast_bot

Steps if you want to use exactly as is

1. Move 2 file directories in this repo to home directory ~/
2. Create virtual environment called zed-env in ~/venvs. If you don't install in a venv or name it something different, the tmux script will need to be adjusted
3. Add pyzed, opencv to zed-env
4. Install openai python library. You could add this to the zed-env, but a small change to the start up script would be required
5. Install node-js if not already installed
6. If the roast_bot_v3 node is in the home directory you should be able to run the start script with:

$cd ~/roast_bot_v3
$./start_dev_tmux.sh

Foxglove
Open Foxglove Desktop App and a Websocket connection to ws://hostname:8765
Open the template saved in the main directory of this repo

Roast Logger
To save information on the roast session use the roast_logger web page: http://hostname:3000
Sessions are saved to the roast_media folder, data from the web page logger is saved to data.csv
