import os
from dotenv import load_dotenv
from neo4j import GraphDatabase, exceptions

# Load environment variables from .env file
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def check_neo4j_connectivity(timeout: float = 5.0) -> bool:
    """
    Checks the connectivity to the Neo4j database.

    Args:
        timeout: The connection timeout in seconds.

    Returns:
        True if the connection is successful, False otherwise.
    """
    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        # If environment variables are not set, we can't connect.
        return False

    driver = None
    try:
        # Attempt to establish a connection with a timeout
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
            connection_timeout=timeout
        )
        # verify_connectivity() blocks until a connection is established or fails
        driver.verify_connectivity()
        return True
    except (exceptions.ServiceUnavailable, exceptions.AuthError, ValueError) as e:
        # Handle connection errors, authentication errors, or invalid URI
        print(f"Health check failed for Neo4j: {e}")
        return False
    finally:
        if driver:
            driver.close()
