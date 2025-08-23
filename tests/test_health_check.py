import pytest
from neo4j import exceptions

# We will create this module and function in the next step
from src.health_check import check_neo4j_connectivity

def test_check_neo4j_connectivity_unavailable(monkeypatch):
    """
    Given: The Neo4j database is unavailable and throws a ServiceUnavailable exception.
    When: The check_neo4j_connectivity function is called.
    Then: It should handle the exception gracefully and return False.
    """
    # Mock the verify_connectivity method to raise the specific exception
    def mock_verify_connectivity():
        raise exceptions.ServiceUnavailable("Connection failed")

    # We assume the driver object will have a `verify_connectivity` method.
    # We will patch the `GraphDatabase.driver` to return an object that has this mock method.
    class MockDriver:
        def verify_connectivity(self):
            mock_verify_connectivity()
        def close(self):
            pass

    # Patch the driver function to return our mock driver
    monkeypatch.setattr(
        "neo4j.GraphDatabase.driver",
        lambda uri, auth, connection_timeout: MockDriver()
    )

    # The function should return False when the service is unavailable
    assert check_neo4j_connectivity() is False
