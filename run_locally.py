#!/usr/bin/env python3
"""Run the Digital Persona app locally for development and testing.

Usage:
    python run_locally.py --email you@yourcompany.com

Requirements:
    - config.yaml must exist with valid Databricks credentials

The app will be available at http://localhost:8000

The --email argument is required and sets the Databricks user identity
used to query system tables for your activity data.
"""

import argparse
import os
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Run the Digital Persona app locally for development."
    )
    parser.add_argument(
        "--email",
        required=True,
        help="Your Databricks user email (used to query your activity data)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8000)),
        help="Port to run the server on (default: 8000)",
    )
    args = parser.parse_args()

    # Load config to validate it exists
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(config_path):
        print("ERROR: config.yaml not found.")
        print("Copy config.yaml.example to config.yaml and fill in your values:")
        print("  cp config.yaml.example config.yaml")
        sys.exit(1)

    # Mark this as a local run — the app uses this to hardcode the user identity
    os.environ["LOCAL"] = "true"
    os.environ["LOCAL_USER_EMAIL"] = args.email

    # Enable Flask debug mode for auto-reload and better error messages
    os.environ["FLASK_DEBUG"] = "1"

    print(f"\n{'='*60}")
    print("  Digital Persona Creator — Local Development Server")
    print(f"{'='*60}")
    print(f"  URL:    http://localhost:{args.port}")
    print(f"  User:   {args.email}")
    print(f"  Debug:  ON (auto-reload enabled)")
    print(f"{'='*60}\n")

    from app.main import app
    app.run(host="127.0.0.1", port=args.port, debug=True)


if __name__ == "__main__":
    main()
