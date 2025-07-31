import http.server
import socketserver
import os

# Define the port the server will run on
PORT = 5001
# Define the directory where our built React app is located
DIRECTORY = "zoo/options_zero_game/visualizer-ui/build"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Initialize the handler in the context of our target directory
        super().__init__(*args, directory=DIRECTORY, **kwargs)

# This is the standard way to start a simple, robust Python web server
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    # Make sure the server directory exists
    if not os.path.isdir(DIRECTORY):
        print(f"Error: The build directory '{DIRECTORY}' does not exist.")
        print("Please run 'npm run build' inside the 'visualizer-ui' folder first.")
        exit()
        
    print(f"Serving files from '{DIRECTORY}' at http://172.19.190.113:{PORT}")
    print("Press Ctrl+C to stop the server.")
    httpd.serve_forever()
