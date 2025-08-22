import pytest
from unittest.mock import MagicMock, patch, mock_open
import yaml

# Patch GraphDatabase.driver at the module level before importing schema_extractor
with patch('neo4j.GraphDatabase.driver') as mock_driver_class:
    mock_driver_instance = MagicMock()
    mock_driver_class.return_value = mock_driver_instance
    mock_session_instance = MagicMock()
    mock_driver_instance.session.return_value.__enter__.return_value = mock_session_instance
    
    # Import schema_extractor after patching
    from schema_extractor import get_node_labels, get_relationship_types, extract_schema, save_schema_to_yaml

# Mock Neo4j driver and session for individual test functions
@pytest.fixture
def mock_neo4j_driver():
    # This fixture will now return the pre-patched mock_driver_instance
    # and ensure its close method is called.
    yield mock_driver_instance

@pytest.fixture
def mock_neo4j_creds():
    with patch('schema_extractor.get_neo4j_credentials') as mock_get_creds:
        mock_get_creds.return_value = {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password"
        }
        yield

@pytest.fixture(autouse=True)
def mock_load_app_config():
    with patch('schema_extractor.load_app_config') as mock_load:
        yield mock_load

def test_get_node_labels(mock_neo4j_driver, mock_neo4j_creds, mock_load_app_config):
    mock_tx = MagicMock()
    mock_tx.run.return_value = [
        {"labels": ["Person"], "freq": 10},
        {"labels": ["Organization"], "freq": 5}
    ]
    
    result = get_node_labels(mock_tx)
    assert result == [
        {"labels": ["Person"], "freq": 10},
        {"labels": ["Organization"], "freq": 5}
    ]
    mock_tx.run.assert_called_once_with("""
    MATCH (n)
    UNWIND labels(n) as label
    RETURN label, count(*) as freq
    ORDER BY freq DESC
    """)

def test_get_relationship_types(mock_neo4j_driver, mock_neo4j_creds, mock_load_app_config):
    mock_tx = MagicMock()
    mock_tx.run.return_value = [
        {"rel": "WORKS_AT", "freq": 8},
        {"rel": "LOCATED_IN", "freq": 3}
    ]

    result = get_relationship_types(mock_tx)
    assert result == [
        {"rel": "WORKS_AT", "freq": 8},
        {"rel": "LOCATED_IN", "freq": 3}
    ]
    mock_tx.run.assert_called_once_with("""
    MATCH ()-[r]->()
    RETURN type(r) as rel, count(*) as freq
    ORDER BY freq DESC
    """)

def test_extract_schema(mock_neo4j_driver, mock_neo4j_creds, mock_load_app_config):
    # mock_neo4j_driver fixture already sets up mock_driver_instance and mock_session_instance
    # which are globally available due to the patching at the module level.
    # So, we can directly set side_effect on mock_session_instance.execute_read
    mock_session_instance.execute_read.side_effect = [
        [{"label": "Person", "freq": 10}, {"label": "Organization", "freq": 5}],
        [{"rel": "WORKS_AT", "freq": 8}, {"rel": "LOCATED_IN", "freq": 3}]
    ]

    schema = extract_schema()
    assert schema == {
        "nodes": ["Person", "Organization"],
        "relationships": ["WORKS_AT", "LOCATED_IN"]
    }
    assert mock_session_instance.execute_read.call_count == 2

def test_save_schema_to_yaml():
    mock_file = mock_open()
    with patch('builtins.open', mock_file):
        schema_data = {"nodes": ["TestNode"], "relationships": ["TEST_REL"]}
        save_schema_to_yaml(schema_data, "test_schema.yaml")
        mock_file.assert_called_once_with("test_schema.yaml", "w", encoding="utf-8")
        # Check the content written to the mock file
        written_content = "".join([call.args[0] for call in mock_file().write.call_args_list])
        expected_content = yaml.dump(schema_data, allow_unicode=True, sort_keys=False)
        assert written_content == expected_content

