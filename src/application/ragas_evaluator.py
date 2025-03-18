from typing import List, Dict, Optional

import ragas
from loguru import logger
from ragas import EvaluationDataset, evaluate, SingleTurnSample
from ragas.llms import LangchainLLMWrapper

from infrastructure.text_embedding_pipeline import VectorStore
from application.llm_service import LLMService


class RAGASEvaluator:
    """Evaluates LLM responses using RAGAS metrics."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_service: LLMService,
        query_answer_pairs: Dict[str, str],
        debug: bool = False,
    ):
        """Initialize the RAGAS evaluator with predefined metrics."""
        self.query_answer_pairs = query_answer_pairs
        self.evaluation_dataset = None
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.evaluator_llm = LangchainLLMWrapper(self.llm_service.client)
        self.debug = debug

    def create_evaluation_dataset(self, top_k: int = 3) -> None:
        try:
            dataset = []
            for query, reference in self.query_answer_pairs.items():
                relevant_docs = self.vector_store.retrieve_relevant_text(query, top_k)
                logger.info(
                    f"Retrieved {len(relevant_docs)} relevant documents for query: {query}"
                )
                response = self.llm_service.generate_response(query, relevant_docs)
                dataset.append(
                    SingleTurnSample(
                        user_input=query,
                        retrieved_contexts=relevant_docs,
                        response=response,
                        reference=reference,
                    )
                )
            self.evaluation_dataset = EvaluationDataset(samples=dataset)
        except Exception as e:
            logger.info(f"Error creating evaluation dataset: {e}")

    def evaluate(self, metrics: List[ragas.metrics]) -> Optional[Dict[str, float]]:
        try:
            if self.debug:
                logger.info("Evaluating with the following dataset:")
                for sample in self.evaluation_dataset.samples:
                    logger.info(sample)
                logger.info("Using metrics:", metrics)

            return evaluate(
                self.evaluation_dataset,
                metrics=metrics,
                llm=self.evaluator_llm,
            )
        except Exception as e:
            logger.info(f"Error during evaluation: {e}")
            return None
