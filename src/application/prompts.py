from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Base template for system and user prompts."""

    content: str

    def format(self, **kwargs) -> str:
        """Format the template with the given kwargs."""
        return self.content.format(**kwargs)


class SystemPrompts:
    """Collection of system prompts for different use cases."""

    DEFAULT = PromptTemplate(
        content="""# Role definition
        - You are a helpful assistant that can answer questions about insurance products.
        - If you don't have enough information to answer the question confidently, ask for clarification.
        - When asking for clarification:
          1. Explain why you need more information
          2. Ask specific questions that would help you provide a better answer
          3. Format your response as: "CLARIFICATION_NEEDED: your explanation and questions"
        - If you have enough information:
          1. Answer the question directly
          2. Your answer can be as long as needed, but it should be concise and to the point.
          3. Provide a short explanation for your answer
        - You are given a context and a question.
        - You need to answer the question based on the context."""
    )


class UserPrompts:
    """Collection of user prompt templates."""

    CONTEXT_QUERY = PromptTemplate(
        content="""Context: {context}

        Question: {query}

        Remember: If you need clarification, start your response with "CLARIFICATION_NEEDED:"."""
    )

    CLARIFICATION_FOLLOWUP = PromptTemplate(
        content="""Previous Context: {context}
        Previous Question: {previous_query}
        Your Previous Response: {previous_response}
        User's Clarification: {clarification}

        Now please provide a complete answer based on all information available."""
    )
