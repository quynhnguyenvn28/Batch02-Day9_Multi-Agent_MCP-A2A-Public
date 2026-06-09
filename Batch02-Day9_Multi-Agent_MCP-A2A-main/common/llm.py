"""Shared LLM factory for all agents.

Uses Google Gemini API for free tier access.
"""

import os

from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm() -> ChatGoogleGenerativeAI:
    """Return a ChatGoogleGenerativeAI client for Gemini."""
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
        top_p=0.95,
        top_k=64,
    )
