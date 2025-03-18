from typing import List, Optional

from loguru import logger

from langchain_openai import ChatOpenAI


class LLMService:
    """Handles interactions with the OpenAI LLM, incorporating retrieval-augmented generation."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the LLM service with the API key and model."""
        self.client = ChatOpenAI(api_key=api_key, model=model)

    def generate_response(
        self, query: str, relevant_docs: Optional[List[str]] = None
    ) -> str:
        """Generates a response from the LLM based on the given prompt and relevant documents."""
        try:
            context = "\n".join(relevant_docs) if relevant_docs else ""
            full_prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"

            response = self.client.invoke(
                input=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can answer questions about the product.",
                    },
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=150,
            )
            return response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            logger.info(f"Error generating response: {e}")
            return "An error occurred while generating the response."
