#!/usr/bin/env python3
"""
Workaround script for guardrails configure command.
The guardrails CLI has a bug in version 0.5.14, so use this script instead.
Writes ~/.guardrailsrc (key=value format) which the CLI actually reads for hub install.

Usage:
    uv run python configure_guardrails.py [API_KEY]

Or set the API key via environment variable:
    export GUARDRAILS_API_KEY=your_api_key_here
    uv run python configure_guardrails.py
"""

import os
import sys
import uuid
from pathlib import Path


def get_config_path() -> Path:
    """Get the guardrails config file path. CLI reads from ~/.guardrailsrc (key=value), not credentials.json."""
    return Path.home() / ".guardrailsrc"


def configure_guardrails(api_key: str):
    """Configure guardrails API key. Writes ~/.guardrailsrc in the format the guardrails CLI expects."""
    config_path = get_config_path()
    lines = [
        f"id={uuid.uuid4()}\n",
        f"token={api_key}\n",
        "enable_metrics=true\n",
        "use_remote_inferencing=true",
    ]
    config_path.write_text("".join(lines), encoding="utf-8")
    print(f"✅ Guardrails API key configured successfully!")
    print(f"   Config saved to: {config_path}")
    print(f"   ⚠️  Keep your API key secret and do not commit it to git!")


def main():
    # Check for API key in environment variable first
    api_key = os.getenv("GUARDRAILS_API_KEY")
    
    # Then check command line argument
    if not api_key and len(sys.argv) > 1:
        api_key = sys.argv[1]
    
    # If still no key, prompt for it
    if not api_key:
        print("🛡️  Guardrails AI Configuration")
        print("   Get your API key from: https://hub.guardrailsai.com/keys")
        print()
        api_key = input("Enter your Guardrails AI API key: ").strip()
    
    if not api_key:
        print("❌ Error: API key is required")
        print("\nUsage:")
        print("  uv run python configure_guardrails.py [API_KEY]")
        print("  or")
        print("  export GUARDRAILS_API_KEY=your_key")
        print("  uv run python configure_guardrails.py")
        sys.exit(1)
    
    configure_guardrails(api_key)


if __name__ == "__main__":
    main()

