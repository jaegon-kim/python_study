import argparse
import functools
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

class Handler(SimpleHTTPRequestHandler):
    def guess_type(self, path):
        if path.endswith(".tar.gz"):
            return "application/gzip"
        if path.endswith(".tgz"):
            return "application/gzip"
        if path.endswith(".tar"):
            return "application/x-tar"
        return super().guess_type(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--directory", default=".")
    args = p.parse_args()
    handler = functools.partial(Handler, directory=args.directory)
    httpd = ThreadingHTTPServer((args.bind, args.port), handler)
    httpd.serve_forever()

if __name__ == "__main__":
    main()
