"""
Healthcare AI Dashboard package.

A thin Streamlit client over the FastAPI backend. This package holds no
business logic: it only wraps the REST API (``api/main.py``) so the
dashboard UI stays a pure view layer.
"""

from .client import APIConfig, HealthcareAPIClient, HealthcareAPIError

__all__ = ["APIConfig", "HealthcareAPIClient", "HealthcareAPIError"]
