import pytest
from dotenv import load_dotenv
import os

@pytest.fixture(scope='session', autouse=True)
def load_env():
    """
    A session-wide fixture to load environment variables from a .env file.
    `autouse=True` ensures it's activated for all tests without needing to
    be explicitly requested.
    """
    # Look for the .env file in the root directory of the project
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)
