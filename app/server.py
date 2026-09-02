"""
Phase 5: GovBench-Med Web Dashboard Backend API.
Serves REAL benchmark telemetry, experiment logs, metrics, and figures.
Uses standard Python http.server - zero external dependencies required!

Run with: python app/server.py --port 8080
"""

import http.server
import socketserver
import json
import os
import glob
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).parent.parent
PORT = 8080
RESULTS_DIR = ROOT / "experiments" / "results"
LOGS_DIR = ROOT / "experiments" / "logs"
DATA_PATH = ROOT / "data" / "processed" / "cases.json"
APP_DIR = ROOT / "app"


class GovBenchHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """API and Static File Request Handler for GovBench-Med Dashboard."""

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        # API Endpoints
        if path == "/api/summary":
            self.send_json_response(self.get_summary_data())
        elif path == "/api/results":
            self.send_json_response(self.get_results_data())
        elif path == "/api/cases":
            self.send_json_response(self.get_cases_data())
        elif path == "/api/log":
            case_id = query.get("case_id", [""])[0]
            level = query.get("level", ["G0"])[0]
            self.send_json_response(self.get_log_data(case_id, level))
        elif path == "/api/ablation":
            self.send_json_response(self.get_ablation_data())
        elif path == "/api/health":
            self.send_json_response({"status": "ok", "version": "0.1.0"})
        elif path == "/" or path == "/dashboard":
            self.serve_dashboard()
        else:
            # Fallback to standard file serving from app or root
            file_path = APP_DIR / path.lstrip("/")
            if file_path.exists() and file_path.is_file():
                self.serve_file(file_path)
            else:
                super().do_GET()

    def serve_dashboard(self):
        dashboard_path = APP_DIR / "dashboard.html"
        if dashboard_path.exists():
            self.serve_file(dashboard_path)
        else:
            self.send_error(404, "Dashboard HTML not found")

    def serve_file(self, path: Path):
        try:
            mime = "text/html"
            if path.suffix == ".js":
                mime = "application/javascript"
            elif path.suffix == ".css":
                mime = "text/css"
            elif path.suffix == ".json":
                mime = "application/json"
            elif path.suffix == ".png":
                mime = "image/png"

            with open(path, "rb") as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))

    def send_json_response(self, data: dict):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    # ---------------------------------------------------------------------------
    # Data Providers
    # ---------------------------------------------------------------------------

    def get_summary_data(self) -> dict:
        summary_files = sorted(glob.glob(str(RESULTS_DIR / "summary_*.json")))
        if summary_files:
            with open(summary_files[-1], "r", encoding="utf-8") as f:
                return json.load(f)
        
        # Fallback default aggregate from available CSVs
        csv_files = sorted(glob.glob(str(RESULTS_DIR / "results_*.csv")))
        if csv_files:
            import pandas as pd
            df = pd.read_csv(csv_files[-1])
            agg = df.groupby("governance_level").agg(
                n=("case_id", "count"),
                accuracy=("correctness", "mean"),
                mean_tokens=("total_tokens", "mean"),
                mean_latency_ms=("latency_ms", "mean")
            ).reset_index()
            return {"comparison": agg.to_dict(orient="records"), "marginal": []}
        return {"comparison": [], "marginal": []}

    def get_results_data(self) -> dict:
        csv_files = sorted(glob.glob(str(RESULTS_DIR / "results_*.csv")))
        if not csv_files:
            return {"rows": []}

        import pandas as pd
        df = pd.read_csv(csv_files[-1])
        return {"rows": df.to_dict(orient="records"), "filename": os.path.basename(csv_files[-1])}

    def get_cases_data(self) -> dict:
        if DATA_PATH.exists():
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                cases = json.load(f)
            return {"cases": cases[:100]}
        return {"cases": []}

    def get_log_data(self, case_id: str, level: str) -> dict:
        if not case_id:
            pattern = str(LOGS_DIR / "*.json")
        else:
            pattern = str(LOGS_DIR / f"{case_id}_{level}_*.json")

        files = glob.glob(pattern)
        if files:
            with open(files[0], "r", encoding="utf-8") as f:
                return json.load(f)
        return {"error": f"Log not found for {case_id} {level}"}

    def get_ablation_data(self) -> dict:
        ablation_files = sorted(glob.glob(str(RESULTS_DIR / "ablation_*.csv")))
        if ablation_files:
            import pandas as pd
            df = pd.read_csv(ablation_files[-1])
            return {"rows": df.to_dict(orient="records")}
        return {"rows": []}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    os.chdir(ROOT)
    server_address = ("", args.port)
    httpd = socketserver.TCPServer(server_address, GovBenchHTTPRequestHandler)
    print(f"GovBench-Med Web Dashboard running on http://localhost:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()