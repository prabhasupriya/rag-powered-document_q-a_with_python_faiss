"""
llm_client.py

Builds the final RAG prompt (system instructions + retrieved context +
user question) and calls the LLM API to generate a grounded, cited answer.

Supports Anthropic, OpenAI, and Groq via a simple provider switch, so
swapping providers only requires changing LLM_PROVIDER in .env -- no code
changes to the rest of the pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert assistant that answers questions using ONLY the "
    "provided context. Follow these rules strictly:\n"
    "1. Base your answer SOLELY on the CONTEXT below. Do not use outside knowledge.\n"
    "2. If the context does not contain the answer, respond exactly with: "
    "\"I cannot answer this based on the provided documents.\"\n"
    "3. At the end of your answer, list the sources you actually used, in the "
    "format: [Source: filename.pdf, page X].\n"
    "4. Be concise and factual. Do not speculate."
)


def format_context_block(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Turn retrieved chunks into a citation-tagged context block for the prompt."""
    lines = []
    for chunk in retrieved_chunks:
        source = chunk["metadata"].get("source", "unknown")
        page = chunk["metadata"].get("page")
        tag = f"[Source: {source}, page {page}]" if page else f"[Source: {source}]"
        lines.append(f"{tag} {chunk['content']}")
    return "\n\n".join(lines)


def build_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    context_block = format_context_block(retrieved_chunks)
    return (
        f"CONTEXT:\n<context>\n{context_block}\n</context>\n\n"
        f"QUESTION: {query}\n\n"
        "Answer the question using only the context above, and cite your sources."
    )


class LLMGenerator:
    """Handles prompt construction + the actual LLM API call.

    Supported providers (set via LLM_PROVIDER in .env, or passed explicitly):
        - "anthropic": Anthropic Claude models (default)
        - "openai":    OpenAI GPT models
        - "groq":      Groq-hosted open models (Llama, Mixtral, etc.),
                        called via the OpenAI-compatible chat completions API
    """

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "anthropic")).lower()
        self.api_key = os.getenv("LLM_API_KEY")

        if not self.api_key:
            raise ValueError(
                "LLM_API_KEY is not set. Copy .env.example to .env and add your API key."
            )

        if self.provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.model = model or os.getenv("LLM_MODEL", "claude-sonnet-4-6")

        elif self.provider == "openai":
            import openai

            self.client = openai.OpenAI(api_key=self.api_key)
            self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

        elif self.provider == "groq":
            import groq

            self.client = groq.Groq(api_key=self.api_key)
            self.model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {self.provider!r}. "
                "Expected one of: 'anthropic', 'openai', 'groq'."
            )

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Build the RAG prompt and call the configured LLM provider."""
        user_prompt = build_prompt(query, retrieved_chunks)

        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return "".join(block.text for block in response.content if block.type == "text")

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content

        if self.provider == "groq":
            # Groq's client mirrors the OpenAI chat completions interface.
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content

        raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}")
