"""
Main entrypoint. Starts the Flask dashboard and provides CLI options
to run data collection and training flows.
"""
import os
import argparse
from dashboard import app as dashboard_app

def run_dashboard():
    port = int(os.environ.get('PORT', 5000))
    dashboard_app.app.run(host='0.0.0.0', port=port, debug=True)

def main():
    parser = argparse.ArgumentParser(description='AI Weather Energy Project')
    parser.add_argument('--dashboard', action='store_true', help='Run Flask dashboard')
    args = parser.parse_args()

    if args.dashboard:
        run_dashboard()
    else:
        # Default: run dashboard
        run_dashboard()

if __name__ == '__main__':
    main()
