"""
Pytest configuration and fixtures for Contextprime tests.
"""

import os
import sys
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv

# Load .env from project root
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment from {env_path}")
else:
    print(f"Warning: .env file not found at {env_path}")

# .env sets QDRANT_HOST=qdrant (Docker-internal service name).
# Override to localhost so integration tests can reach the container directly.
os.environ["QDRANT_HOST"] = "localhost"
os.environ["QDRANT_PORT"] = "6333"

# Reset the settings singleton so the override takes effect before any test
# module imports get_settings() and caches the docker-internal hostname.
try:
    from contextprime.core.config import reset_settings
    reset_settings()
except Exception:
    pass
