# zoo/options_zero_game/serve.py
# <<< ROBUST VERSION with Pre-flight Check >>>

import http.server
import socketserver
import os

PORT = 5001
DIRECTORY = "zoo/options_zero_game/visualizer-ui/build"
REPLAY_LOG_PATH = os.path.join(DIRECTORY, "replay_log.json")

# --- NEW: Pre-flight check ---
# 1. Check if the build directory exists at all.
if not os.path.isdir(DIRECTORY):
    print(f"--- ERROR ---")
    print(f"The build directory '{DIRECTORY}' does not exist.")
    print("Please run 'npm run build' inside the 'visualizer-ui' folder first.")
    exit()

# 2. Check if the crucial replay_log.json file exists.
if not os.path.isfile(REPLAY_LOG_PATH):
    print(f"--- ERROR ---")
    print(f"The replay log file was not found at '{REPLAY_LOG_PATH}'.")
    print("Please run the evaluation script first to generate the log:")
    print("  python3 zoo/options_zero_game/entry/options_zero_game_eval.py")
    exit()

# --- Standard Server Logic ---
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"\n--- Server Starting ---")
    print(f"Serving files from '{DIRECTORY}'")
    print(f"Successfully found replay_log.json.")
    print(f"Please open your browser to http://<your-ip-address>:{PORT}")
    print("Press Ctrl+C to stop the server.")
    httpd.serve_forever()
