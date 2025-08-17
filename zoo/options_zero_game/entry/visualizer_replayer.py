# <<< UPGRADED, API-AWARE WEB SERVER >>>

import http.server
import socketserver
import os
import json
import glob
from datetime import datetime

PORT = 5001
# The root directory for serving files is still the main build folder
ROOT_DIRECTORY = "zoo/options_zero_game/visualizer-ui/build"
REPORTS_DIRECTORY = os.path.join(ROOT_DIRECTORY, "reports")

class APIHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # We call the parent constructor with the ROOT_DIRECTORY
        super().__init__(*args, directory=ROOT_DIRECTORY, **kwargs)

    def do_GET(self):
        # If the client is requesting our new API endpoint...
        if self.path == '/api/history':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # --- API Logic: Find all reports and create a JSON response ---
            report_files = glob.glob(os.path.join(REPORTS_DIRECTORY, "strategy_report_*.json"))
            
            history = []
            for filepath in sorted(report_files, reverse=True): # Show newest first
                filename = os.path.basename(filepath)
                try:
                    # Extract timestamp from filename for a human-readable label
                    ts_str = filename.replace('strategy_report_', '').replace('.json', '')
                    ts_obj = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                    label = ts_obj.strftime("%Y-%m-%d %H:%M:%S")
                    history.append({'filename': filename, 'label': label})
                except ValueError:
                    # Ignore files that don't match the expected timestamp format
                    continue
            
            # Write the JSON list of reports back to the client
            self.wfile.write(json.dumps(history).encode())
        
        else:
            # For any other request, fall back to the standard file server behavior
            super().do_GET()

# --- Standard Server Startup Logic ---
with socketserver.TCPServer(("", PORT), APIHandler) as httpd:
    print(f"\n--- Advanced Server Starting ---")
    print(f"Serving files from '{ROOT_DIRECTORY}'")
    print(f"API endpoint for history available at /api/history")
    print(f"Please open your browser to http://<your-ip-address>:{PORT}")
    print("Press Ctrl+C to stop the server.")
    httpd.serve_forever()
