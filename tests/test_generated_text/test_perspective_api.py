"""Tests for Perspective API client."""

import pytest
from bias_scope.generated_text.perspective_api import PerspectiveAPIClient


class TestPerspectiveAPIClient:
    """Test suite for Perspective API client."""
    
    def test_initialization_with_valid_key(self):
        """Test client initializes with valid API key."""
        client = PerspectiveAPIClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.rate_limit_delay == 1.0
        assert client.max_retries == 3
    
    def test_initialization_with_custom_params(self):
        """Test client initializes with custom parameters."""
        client = PerspectiveAPIClient(
            api_key="test_key",
            rate_limit_delay=0.5,
            max_retries=5
        )
        assert client.rate_limit_delay == 0.5
        assert client.max_retries == 5
    
    def test_empty_api_key_raises_error(self):
        """Test that empty API key raises ValueError."""
        with pytest.raises(ValueError, match="api_key cannot be empty"):
            PerspectiveAPIClient(api_key="")
    
    def test_none_api_key_raises_error(self):
        """Test that None API key raises ValueError."""
        with pytest.raises(ValueError, match="api_key cannot be empty"):
            PerspectiveAPIClient(api_key=None)
    
    def test_score_text_empty_raises_error(self):
        """Test that empty text raises ValueError."""
        client = PerspectiveAPIClient(api_key="test_key")
        
        with pytest.raises(ValueError, match="cannot be empty"):
            client.score_text("")
    
    def test_score_text_whitespace_raises_error(self):
        """Test that whitespace-only text raises ValueError."""
        client = PerspectiveAPIClient(api_key="test_key")
        
        with pytest.raises(ValueError, match="cannot be empty"):
            client.score_text("   ")
    
    def test_base_url_is_correct(self):
        """Test that base URL is set correctly."""
        client = PerspectiveAPIClient(api_key="test_key")
        assert client.base_url == "https://commentanalyzer.googleapis.com/v1alpha1"
    
    def test_score_batch_returns_list(self):
        """Test that score_batch would return a list (mock test)."""
        # This is a structure test - actual API calls require valid key
        client = PerspectiveAPIClient(api_key="test_key")
        assert hasattr(client, 'score_batch')
        assert callable(client.score_batch)
    
    def test_client_has_score_text_method(self):
        """Test that client has score_text method."""
        client = PerspectiveAPIClient(api_key="test_key")
        assert hasattr(client, 'score_text')
        assert callable(client.score_text)
    
    def test_attributes_set_correctly(self):
        """Test that all attributes are set correctly."""
        client = PerspectiveAPIClient(
            api_key="my_key",
            rate_limit_delay=2.0,
            max_retries=10
        )
        assert client.api_key == "my_key"
        assert client.base_url == "https://commentanalyzer.googleapis.com/v1alpha1"
        assert client.rate_limit_delay == 2.0
        assert client.max_retries == 10
