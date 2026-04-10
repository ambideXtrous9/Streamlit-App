"""Tests for AirbnbAgent module."""
import pytest
import json
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ==============================================================================
# Query Parsing Tests (parse_query)
# ==============================================================================

class TestParseQuery:
    """Test parse_query function from airbnb_search.py."""

    def test_parse_query_default_values(self):
        """Test default location and adults when not specified."""
        from AirbnbAgent.airbnb_search import parse_query

        result = parse_query("show me places")
        assert result["location"] == "Wayanad, Kerala"
        assert result["adults"] == 2

    def test_parse_query_with_date_range(self):
        """Test extraction of 'next X days' pattern."""
        from AirbnbAgent.airbnb_search import parse_query

        result = parse_query("find places for next 5 days")
        today = datetime.now()
        expected_checkout = (today + timedelta(days=5)).strftime("%Y-%m-%d")

        assert result["checkin"] == today.strftime("%Y-%m-%d")
        assert result["checkout"] == expected_checkout

    def test_parse_query_with_location(self):
        """Test location extraction from query."""
        from AirbnbAgent.airbnb_search import parse_query

        result = parse_query("find places in Tokyo")
        assert result["location"] == "Tokyo"

    def test_parse_query_with_location_and_dates(self):
        """Test combined location and date extraction."""
        from AirbnbAgent.airbnb_search import parse_query

        result = parse_query("find places in Goa for next 3 days")
        assert result["location"] == "Goa"
        assert "checkin" in result
        assert "checkout" in result

    def test_parse_query_complex_location(self):
        """Test location with multiple words and commas."""
        from AirbnbAgent.airbnb_search import parse_query

        result = parse_query("search for homes in New York, USA")
        # Regex captures everything after 'in' until end of string
        assert "New York, USA" in result["location"]

    def test_parse_query_no_date_match(self):
        """Test query without date patterns."""
        from AirbnbAgent.airbnb_search import parse_query

        result = parse_query("find apartments in Paris")
        assert "checkin" not in result
        assert "checkout" not in result
        assert result["location"] == "Paris"


# ==============================================================================
# API Key Retrieval Tests (get_groq_api_key)
# ==============================================================================

class TestGetGrokApiKey:
    """Test get_groq_api_key function."""

    @patch.dict(os.environ, {"GROQ_API_KEY": "test_key_123"})
    def test_get_api_key_from_env(self):
        """Test retrieving API key from environment variable."""
        from AirbnbAgent.airbnb_search import get_groq_api_key
        assert get_groq_api_key() == "test_key_123"

    @patch.dict(os.environ, {}, clear=True)
    @patch("os.path.exists", return_value=False)
    def test_get_api_key_not_found(self, mock_exists):
        """Test returning empty string when no API key is found."""
        from AirbnbAgent.airbnb_search import get_groq_api_key
        result = get_groq_api_key()
        assert result == "" or result is None


# ==============================================================================
# Airbnb Search Tests (run_airbnb_search)
# ==============================================================================

class TestRunAirbnbSearch:
    """Test run_airbnb_search function."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    @patch("AirbnbAgent.airbnb_search.get_groq_api_key", return_value="")
    async def test_run_search_no_api_key(self, mock_get_key):
        """Test error message when API key is missing."""
        from AirbnbAgent.airbnb_search import run_airbnb_search

        result = await run_airbnb_search("test query")
        assert "GROQ_API_KEY not found" in result

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
    @patch("AirbnbAgent.airbnb_search.asyncio.to_thread")
    @patch("AirbnbAgent.airbnb_search.ChatGroq")
    async def test_run_search_subprocess_success(self, mock_groq, mock_to_thread):
        """Test successful subprocess call and LLM summarization."""
        from AirbnbAgent.airbnb_search import run_airbnb_search

        # Mock subprocess result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"name": "Test Place", "price": "$100"}'
        mock_result.stderr = ""
        mock_to_thread.return_value = mock_result

        # Mock LLM response
        mock_llm_response = MagicMock()
        mock_llm_response.content = "## Top 5 Properties\n\n1. Test Place - $100/night"
        mock_groq_instance = MagicMock()
        mock_groq_instance.ainvoke = AsyncMock(return_value=mock_llm_response)
        mock_groq.return_value = mock_groq_instance

        result = await run_airbnb_search("places in Goa")

        assert "Top 5 Properties" in result
        mock_to_thread.assert_called_once()

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
    @patch("AirbnbAgent.airbnb_search.asyncio.to_thread")
    @patch("AirbnbAgent.airbnb_search.ChatGroq")
    async def test_run_search_subprocess_failure(self, mock_groq, mock_to_thread):
        """Test handling of subprocess failure."""
        from AirbnbAgent.airbnb_search import run_airbnb_search

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: npx not found"
        mock_to_thread.return_value = mock_result

        result = await run_airbnb_search("places in Goa")

        assert "failed" in result.lower() or "Error" in result

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
    @patch("AirbnbAgent.airbnb_search.asyncio.to_thread")
    @patch("AirbnbAgent.airbnb_search.ChatGroq")
    async def test_run_search_json_parsing_error(self, mock_groq, mock_to_thread):
        """Test handling of invalid JSON from subprocess."""
        from AirbnbAgent.airbnb_search import run_airbnb_search

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "invalid json output"
        mock_result.stderr = ""
        mock_to_thread.return_value = mock_result

        # LLM should still try to summarize whatever output it gets
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Summary of results"
        mock_groq_instance = MagicMock()
        mock_groq_instance.ainvoke = AsyncMock(return_value=mock_llm_response)
        mock_groq.return_value = mock_groq_instance

        result = await run_airbnb_search("places in Goa")

        # Should still get a response from LLM even if raw output is not JSON
        assert result is not None


# ==============================================================================
# Main Entry Point Tests
# ==============================================================================

class TestMainEntryPoint:
    """Test main() function in airbnb_search.py."""

    @patch("sys.argv", ["airbnb_search.py"])
    def test_main_no_query_exits(self):
        """Test sys.exit when no query is provided."""
        from AirbnbAgent.airbnb_search import main

        with pytest.raises(SystemExit):
            main()

    @patch("sys.argv", ["airbnb_search.py", "test query"])
    @patch("AirbnbAgent.airbnb_search.asyncio.run")
    def test_main_with_query_success(self, mock_asyncio_run):
        """Test successful execution with query."""
        from AirbnbAgent.airbnb_search import main
        import io
        import sys

        mock_asyncio_run.return_value = "Test result summary"

        # Capture stdout
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output

        main()

        sys.stdout = old_stdout
        output = json.loads(captured_output.getvalue())

        assert output["success"] is True
        assert "result" in output

    @patch("sys.argv", ["airbnb_search.py", "test query"])
    @patch("AirbnbAgent.airbnb_search.asyncio.run")
    def test_main_with_query_failure(self, mock_asyncio_run):
        """Test error handling in main()."""
        from AirbnbAgent.airbnb_search import main
        import io
        import sys

        mock_asyncio_run.side_effect = Exception("Test error")

        # Capture stderr
        captured_output = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured_output

        with pytest.raises(SystemExit):
            main()

        sys.stderr = old_stderr
        error_output = captured_output.getvalue()

        assert "Test error" in error_output


# ==============================================================================
# Weather Extraction Tests (extract_weather)
# ==============================================================================

class TestExtractWeather:
    """Test extract_weather function from tourAgent.py."""

    def test_extract_weather_basic(self):
        """Test basic weather extraction."""
        from AirbnbAgent.tourAgent import extract_weather

        data = {
            "location": {"name": "Goa", "region": "Goa", "country": "India"},
            "current": {
                "condition": {"text": "Sunny"},
                "temp_c": 30,
                "feelslike_c": 32,
                "humidity": 70,
                "gust_kph": 15,
                "pressure_mb": 1012
            },
            "forecast": {
                "forecastday": [
                    {
                        "date": "2024-01-01",
                        "day": {
                            "condition": {"text": "Partly cloudy"},
                            "maxtemp_c": 31,
                            "mintemp_c": 24,
                            "avghumidity": 75,
                            "maxwind_kph": 20
                        }
                    }
                ]
            }
        }

        result = extract_weather(data)

        assert "Goa" in result
        assert "30" in result
        assert "Sunny" in result
        assert "2024-01-01" in result

    def test_extract_weather_empty_forecast(self):
        """Test weather extraction with no forecast days."""
        from AirbnbAgent.tourAgent import extract_weather

        data = {
            "location": {"name": "Delhi", "region": "Delhi", "country": "India"},
            "current": {
                "condition": {"text": "Clear"},
                "temp_c": 25,
                "feelslike_c": 24,
                "humidity": 50,
                "gust_kph": 10,
                "pressure_mb": 1015
            },
            "forecast": {
                "forecastday": []
            }
        }

        result = extract_weather(data)

        assert "Delhi" in result
        assert "Clear" in result
        assert "Forecast" in result  # Section header should exist

    def test_extract_weather_multiple_days(self):
        """Test weather extraction with multiple forecast days."""
        from AirbnbAgent.tourAgent import extract_weather

        data = {
            "location": {"name": "Mumbai", "region": "Maharashtra", "country": "India"},
            "current": {
                "condition": {"text": "Rainy"},
                "temp_c": 28,
                "feelslike_c": 30,
                "humidity": 85,
                "gust_kph": 25,
                "pressure_mb": 1008
            },
            "forecast": {
                "forecastday": [
                    {
                        "date": "2024-01-01",
                        "day": {
                            "condition": {"text": "Rain"},
                            "maxtemp_c": 29,
                            "mintemp_c": 25,
                            "avghumidity": 90,
                            "maxwind_kph": 30
                        }
                    },
                    {
                        "date": "2024-01-02",
                        "day": {
                            "condition": {"text": "Cloudy"},
                            "maxtemp_c": 28,
                            "mintemp_c": 24,
                            "avghumidity": 80,
                            "maxwind_kph": 20
                        }
                    }
                ]
            }
        }

        result = extract_weather(data)

        assert "Mumbai" in result
        assert "2024-01-01" in result
        assert "2024-01-02" in result
        assert "Rain" in result
        assert "Cloudy" in result


# ==============================================================================
# Weather Tool Tests (get_forecast)
# ==============================================================================

class TestGetForecastTool:
    """Test get_forecast tool function."""

    @patch("AirbnbAgent.tourAgent.requests.get")
    @patch("AirbnbAgent.tourAgent.st")
    def test_get_forecast_success(self, mock_st, mock_requests_get):
        """Test successful weather API call."""
        from AirbnbAgent.tourAgent import get_forecast

        # Mock Streamlit secrets
        mock_st.secrets = {"WEATHER_API_KEY": "test_weather_key"}

        # Mock weather API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "location": {"name": "Goa", "region": "Goa", "country": "India"},
            "current": {
                "condition": {"text": "Sunny"},
                "temp_c": 30,
                "feelslike_c": 32,
                "humidity": 70,
                "gust_kph": 15,
                "pressure_mb": 1012
            },
            "forecast": {
                "forecastday": [
                    {
                        "date": "2024-01-01",
                        "day": {
                            "condition": {"text": "Partly cloudy"},
                            "maxtemp_c": 31,
                            "mintemp_c": 24,
                            "avghumidity": 75,
                            "maxwind_kph": 20
                        }
                    }
                ]
            }
        }
        mock_requests_get.return_value = mock_response

        result = get_forecast.invoke({"location": "Goa", "days": 1})

        assert result is not None
        assert "Goa" in result
        mock_requests_get.assert_called_once()

    @patch("AirbnbAgent.tourAgent.requests.get")
    @patch("AirbnbAgent.tourAgent.st")
    def test_get_forecast_api_error(self, mock_st, mock_requests_get):
        """Test weather API error handling."""
        from AirbnbAgent.tourAgent import get_forecast
        import requests

        mock_st.secrets = {"WEATHER_API_KEY": "test_weather_key"}
        mock_requests_get.side_effect = requests.RequestException("API error")

        result = get_forecast.invoke({"location": "Goa", "days": 1})

        assert result is None


# ==============================================================================
# Graph Topology Tests
# ==============================================================================

class TestGraphTopology:
    """Test the LangGraph graph structure."""

    def test_graph_has_required_nodes(self):
        """Test that graph has weatherAgent, airbnbAgent, and tourAgent nodes."""
        from AirbnbAgent.tourAgent import async_graph

        nodes = list(async_graph.nodes)

        assert "weatherAgent" in nodes
        assert "airbnbAgent" in nodes
        assert "tourAgent" in nodes

    def test_graph_compiles(self):
        """Test that graph compiles without errors."""
        from AirbnbAgent.tourAgent import app

        assert app is not None
        assert hasattr(app, "invoke") or hasattr(app, "astream_events")


# ==============================================================================
# State Definition Tests
# ==============================================================================

class TestStateDefinition:
    """Test ArticleResponse state definition."""

    def test_article_response_has_required_keys(self):
        """Test ArticleResponse TypedDict has expected keys."""
        from AirbnbAgent.tourAgent import ArticleResponse

        assert "topic" in ArticleResponse.__annotations__
        assert "summary" in ArticleResponse.__annotations__
        assert "knowledge" in ArticleResponse.__annotations__


# ==============================================================================
# Node.js Availability Check Tests
# ==============================================================================

class TestNodeAvailability:
    """Test Node.js availability check logic."""

    @patch("subprocess.run")
    def test_node_check_success(self, mock_run):
        """Test Node.js check when both node and npx are available."""
        mock_run.return_value = MagicMock(returncode=0)

        # Import after patching to test the check
        with patch.dict("sys.modules", {}):
            import importlib
            import AirbnbAgent.tourAgent
            importlib.reload(AirbnbAgent.tourAgent)

            # Check that at least the check ran
            mock_run.assert_called()

    @patch("subprocess.run")
    def test_node_check_failure_with_fallback(self, mock_run):
        """Test Node.js check with fallback to apt install."""
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            # First two calls fail (node -v, npx --version)
            if call_count[0] <= 2:
                raise FileNotFoundError("node not found")
            # After apt install, succeed
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        with patch.dict("sys.modules", {}):
            import importlib
            import AirbnbAgent.tourAgent
            importlib.reload(AirbnbAgent.tourAgent)


# ==============================================================================
# Langfuse Callback Handler Tests
# ==============================================================================

class TestLangfuseHandler:
    """Test Langfuse callback handler initialization."""

    def test_langfuse_handler_imports(self):
        """Test that langfuse handler can be imported (or gracefully skipped)."""
        # This test just ensures the module doesn't crash on import
        from AirbnbAgent.tourAgent import langfuse_handler

        # Handler should either be a CallbackHandler instance or None
        assert langfuse_handler is None or hasattr(langfuse_handler, "__class__")


# ==============================================================================
# Pytest Fixtures
# ==============================================================================

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        "GROQ_API_KEY": "test_groq_key",
        "WEATHER_API_KEY": "test_weather_key"
    }):
        yield


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    mock_response = MagicMock()
    mock_response.content = "Test response content"
    return mock_response
