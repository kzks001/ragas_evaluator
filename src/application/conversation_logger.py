from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
from loguru import logger


@dataclass
class ConversationLog:
    """Stores a single conversation interaction."""

    query: str
    response: str
    expected_answer: Optional[str]
    retrieved_chunks: List[Dict[str, str]]
    timestamp: datetime


class ConversationLogger:
    """Logs conversation data for later evaluation."""

    def __init__(self):
        """Initialize the conversation logger."""
        self.logs: List[ConversationLog] = []

    def log_interaction(
        self,
        query: str,
        response: str,
        expected_answer: Optional[str] = None,
        retrieved_chunks: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """
        Log a conversation interaction.

        Args:
            query: The user's question
            response: The LLM's response
            expected_answer: The expected/ground truth answer if available
            retrieved_chunks: The chunks retrieved from the vector store
        """
        log_entry = ConversationLog(
            query=query,
            response=response,
            expected_answer=expected_answer,
            retrieved_chunks=retrieved_chunks or [],
            timestamp=datetime.now(),
        )
        self.logs.append(log_entry)

    def get_query_answer_pairs(self) -> Dict[str, str]:
        """
        Get query-answer pairs for RAGAS evaluation.

        Returns:
            Dict mapping queries to their expected answers
        """
        return {
            log.query: log.expected_answer
            for log in self.logs
            if log.expected_answer is not None
        }

    def clear_logs(self) -> None:
        """Clear all stored logs."""
        self.logs.clear()

    def save_to_csv(self, filepath: str) -> None:
        """
        Save conversation logs to a CSV file.

        Args:
            filepath (str): Path where the CSV file should be saved
        """
        if not self.logs:
            logger.warning("No logs to save")
            return

        # Convert logs to list of dictionaries
        log_dicts = []
        for log in self.logs:
            log_dict = {
                "timestamp": log.timestamp,
                "query": log.query,
                "response": log.response,
                "expected_answer": log.expected_answer,
                # Join retrieved chunks into a single string with separator
                "retrieved_chunks": (
                    "|||".join(chunk["text"] for chunk in log.retrieved_chunks)
                    if log.retrieved_chunks
                    else ""
                ),
            }
            log_dicts.append(log_dict)

        # Convert to DataFrame and save
        df = pd.DataFrame(log_dicts)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(self.logs)} conversation logs to {filepath}")
