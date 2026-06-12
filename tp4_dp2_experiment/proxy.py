import http.server
import socketserver
import urllib.request
import urllib.error
import threading
import sys

TARGET_PORTS = [8001, 8002]
current_index = 0
lock = threading.Lock()

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.proxy_request("GET")

    def do_POST(self):
        self.proxy_request("POST")

    def proxy_request(self, method):
        global current_index
        with lock:
            port = TARGET_PORTS[current_index]
            current_index = (current_index + 1) % len(TARGET_PORTS)

        url = f"http://localhost:{port}{self.path}"
        
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        headers = {key: val for key, val in self.headers.items() if key.lower() not in ['host', 'connection']}
        
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                self.send_response(response.status)
                for key, val in response.getheaders():
                    if key.lower() not in ['transfer-encoding', 'connection']:
                        self.send_header(key, val)
                self.end_headers()
                self.wfile.write(response.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for key, val in e.headers.items():
                if key.lower() not in ['transfer-encoding', 'connection']:
                    self.send_header(key, val)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

def run_proxy(port=8000):
    handler = ProxyHandler
    # Enable address reuse to avoid port binding conflicts
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", port), handler) as httpd:
        print(f"Proxy serving on port {port} forwarding to {TARGET_PORTS}...", flush=True)
        httpd.serve_forever()

if __name__ == "__main__":
    run_proxy()
