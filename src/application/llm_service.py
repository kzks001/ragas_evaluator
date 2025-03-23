from typing import List, Optional
from dataclasses import dataclass

from loguru import logger
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from .prompts import SystemPrompts, UserPrompts, PromptTemplate
from .conversation_logger import ConversationLogger


@dataclass
class LLMResponse:
    """Structured response from the LLM."""

    content: str
    needs_clarification: bool


class LLMService:
    """Handles interactions with the OpenAI LLM, using dual memory for handling
    general conversations and clarifications separately.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini-2024-07-18",
        system_prompt: PromptTemplate = SystemPrompts.DEFAULT,
        evaluation_mode: bool = False,
    ):
        """Initialize the LLM service.

        Args:
            api_key (str): OpenAI API key
            model (str): Model identifier
            system_prompt (PromptTemplate): System prompt template
            evaluation_mode (bool): If True, uses evaluation mode that never asks for clarification
        """
        self.client = ChatOpenAI(api_key=api_key, model=model)
        self.evaluation_mode = evaluation_mode

        # Use evaluation prompt if in evaluation mode
        if evaluation_mode:
            system_prompt = SystemPrompts.EVALUATION_MODE

        self.system_message = SystemMessage(content=system_prompt.content)
        self.conversation_logger = ConversationLogger()

        # Memory that only tracks the last exchange when clarification is needed
        self.recent_memory = ConversationBufferMemory(return_messages=True)

        # Memory that stores the entire conversation history
        self.full_memory = ConversationBufferMemory(return_messages=True)

    def _process_response(self, response: str) -> LLMResponse:
        """Processes the LLM response and checks if clarification is needed.

        Args:
            response (str): Raw response from the LLM

        Returns:
            LLMResponse: Processed response with a clarification flag
        """
        content = response.content if hasattr(response, "content") else str(response)
        needs_clarification = content.strip().startswith("CLARIFICATION_NEEDED:")

        if needs_clarification:
            content = content.replace("CLARIFICATION_NEEDED:", "", 1).strip()

        return LLMResponse(content=content, needs_clarification=needs_clarification)

    def __call__(
        self,
        query: str,
        relevant_docs: Optional[List[str]] = None,
    ) -> LLMResponse:
        return self.generate_response(query, relevant_docs)

    def generate_response(
        self,
        query: str,
        relevant_docs: Optional[List[str]] = None,
        expected_answer: Optional[str] = None,
    ) -> LLMResponse:
        """Generates a response from the LLM while properly handling clarification
        requests separately from normal conversation history.

        Args:
            query (str): User's question
            relevant_docs (Optional[List[str]]): List of relevant document chunks
            expected_answer (Optional[str]): Expected answer for evaluation purposes

        Returns:
            LLMResponse: Generated response with clarification status
        """
        try:
            # Retrieve full conversation history
            full_memory_msgs = self.full_memory.load_memory_variables({})["history"]

            # Retrieve previous clarification-only interaction (if any)
            recent_memory_msgs = self.recent_memory.load_memory_variables({})["history"]

            previous_interaction = None
            if recent_memory_msgs:
                # Extract only the last clarification-related exchange
                if len(recent_memory_msgs) >= 2:
                    previous_interaction = {
                        "query": recent_memory_msgs[-2].content,
                        "response": recent_memory_msgs[-1].content,
                    }

            # Construct the full conversation history
            conversation_history = (
                "\n".join([msg.content for msg in full_memory_msgs])
                if full_memory_msgs
                else ""
            )

            messages = [self.system_message]

            if full_memory_msgs:
                messages.extend(
                    full_memory_msgs
                )  # Provide context from long-term memory

            context = "\n".join(relevant_docs) if relevant_docs else ""

            if previous_interaction:
                # This is a clarification follow-up
                full_prompt = UserPrompts.CLARIFICATION_FOLLOWUP.format(
                    context=context,
                    previous_query=previous_interaction["query"],
                    previous_response=previous_interaction["response"],
                    clarification=query,
                )
            else:
                # Normal conversation flow
                full_prompt = UserPrompts.CONTEXT_QUERY.format(
                    context=context,
                    conversation_history=conversation_history,
                    query=query,
                )

            messages.append(HumanMessage(content=full_prompt))
            response = self.client.invoke(input=messages, max_tokens=150)

            processed_response = self._process_response(response)

            # Log the interaction
            if relevant_docs:
                chunks_with_metadata = [{"text": doc} for doc in relevant_docs]
                self.conversation_logger.log_interaction(
                    query=query,
                    response=processed_response.content,
                    expected_answer=expected_answer,
                    retrieved_chunks=chunks_with_metadata,
                )

            if processed_response.needs_clarification:
                # If the bot needs clarification, store this exchange in short-term memory
                self.recent_memory.clear()  # Clear previous clarification context
                self.recent_memory.save_context(
                    {"input": query}, {"output": processed_response.content}
                )
            else:
                # If no clarification needed, store conversation in long-term memory
                self.full_memory.save_context(
                    {"input": query}, {"output": processed_response.content}
                )

            return processed_response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return LLMResponse(
                content="An error occurred while generating the response.",
                needs_clarification=False,
            )

    def clear_recent_memory(self) -> None:
        """Clears the short-term memory (only clarification interactions)."""
        self.recent_memory.clear()

    def clear_full_memory(self) -> None:
        """Clears the long-term memory (entire conversation history)."""
        self.full_memory.clear()
