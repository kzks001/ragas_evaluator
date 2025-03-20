from typing import List, Optional

from loguru import logger

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage


class LLMService:
    """Handles interactions with the OpenAI LLM, incorporating retrieval-augmented generation with memory."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the LLM service with the API key and model.

        Args:
            api_key (str): OpenAI API key
            model (str): Model identifier to use
        """
        self.client = ChatOpenAI(api_key=api_key, model=model)
        self.memory = ConversationBufferMemory(return_messages=True)
        self.system_message = SystemMessage(
            content="You are a helpful assistant that can answer questions about the product."
        )

    def generate_response(
        self, query: str, relevant_docs: Optional[List[str]] = None
    ) -> str:
        """Generates a response from the LLM based on the given prompt, relevant documents, and conversation history.

        Args:
            query (str): User's question
            relevant_docs (Optional[List[str]]): List of relevant document chunks

        Returns:
            str: Generated response
        """
        try:
            # Get conversation history
            memory_messages = self.memory.load_memory_variables({})["history"]

            # Prepare messages list starting with system message
            messages = [self.system_message]

            # Add memory messages
            messages.extend(memory_messages)

            # Prepare context and query
            context = "\n".join(relevant_docs) if relevant_docs else ""
            full_prompt = f"Context:\n{context}\n\nQuestion:\n{query}"

            # Add current query
            messages.append(HumanMessage(content=full_prompt))

            # Generate response
            response = self.client.invoke(
                input=messages,
                max_tokens=150,
            )

            # Store the conversation
            self.memory.save_context(
                {"input": query},
                {
                    "output": (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )
                },
            )

            return response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "An error occurred while generating the response."

    def clear_memory(self) -> None:
        """Clears the conversation history."""
        self.memory.clear()
