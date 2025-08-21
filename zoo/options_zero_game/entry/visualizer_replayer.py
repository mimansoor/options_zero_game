import http.server
import socketserver
import os
import json
import glob
from datetime import datetime

PORT = 5001
# Define the directory we want to serve files from
SERVE_DIRECTORY = "zoo/options_zero_game/visualizer-ui/build"

# --- Pre-flight Checks ---
if not os.path.isdir(SERVE_DIRECTORY):
    print(f"--- ERROR ---")
    print(f"The build directory '{SERVE_DIRECTORY}' does not exist.")
    print("Please run 'npm run build' inside the 'visualizer-ui' folder first.")
    exit()

class APIHandler(http.server.SimpleHTTPRequestHandler):
    # This handler no longer needs a custom __init__
    
    def do_GET(self):
        # API endpoint path is now relative to the new root
        if self.path == '/api/history':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Logic now correctly searches in "reports/" not the full path
            report_files = glob.glob(os.path.join("reports", "strategy_report_*.json"))
            
            history = []
            for filepath in sorted(report_files, reverse=True):
                filename = os.path.basename(filepath)
                try:
                    ts_str = filename.replace('strategy_report_', '').replace('.json', '')
                    ts_obj = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                    label = ts_obj.strftime("%Y-%m-%d %H:%M:%S")
                    history.append({'filename': filename, 'label': label})
                except ValueError:
                    continue
            
            self.wfile.write(json.dumps(history).encode())
        
        else:
            # For all other requests, fall back to the standard file server.
            # It will now correctly find index.html, replay_log.json, etc.
            super().do_GET()

# <<< --- THE DEFINITIVE FIX --- >>>
# 1. Change the current working directory to the target directory.
# This is the most robust way to ensure the server finds all files.
try:
    os.chdir(SERVE_DIRECTORY)
except FileNotFoundError:
    print(f"--- FATAL ERROR ---")
    print(f"Could not change directory to '{SERVE_DIRECTORY}'. Please check the path.")
    exit()

# 2. Start the server. It will now operate from within the 'build' directory.
with socketserver.TCPServer(("", PORT), APIHandler) as httpd:
    print(f"\n--- Advanced Server Starting ---")
    print(f"Serving files from: '{os.getcwd()}'") # Print the new working directory
    print(f"API endpoint for history available at /api/history")
    print(f"Please open your browser to http://<your-ip-address>:{PORT}")
    print("Press Ctrl+C to stop the server.")
    httpd.serve_forever()
