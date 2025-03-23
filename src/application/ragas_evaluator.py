from typing import List, Dict, Optional

from loguru import logger
from ragas import EvaluationDataset, evaluate, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Metric

from infrastructure.text_embedding_pipeline import VectorStore
from application.llm_service import LLMService


class RAGASEvaluator:
    """Evaluates LLM responses using RAGAS metrics."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_service: LLMService,
        query_answer_pairs: Dict[str, str],
    ):
        """Initialize the RAGAS evaluator with predefined metrics."""
        self.query_answer_pairs = query_answer_pairs
        self.evaluation_dataset = None
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.evaluator_llm = LangchainLLMWrapper(self.llm_service.client)

    def create_evaluation_dataset(self, top_k: int = 3) -> None:
        """
        Creates an evaluation dataset from query-answer pairs.

        Args:
            top_k (int): Number of relevant documents to retrieve for each query.
        """
        try:
            dataset = []
            for query, reference in self.query_answer_pairs.items():
                # Get relevant docs and extract just the text
                raw_docs = self.vector_store.retrieve_relevant_text(query, top_k)
                relevant_docs = [doc["text"] for doc in raw_docs]

                logger.info(
                    f"Retrieved {len(relevant_docs)} relevant documents for query: {query}"
                )

                # Generate response using the text content
                context = "\n".join(relevant_docs)
                llm_response = self.llm_service.generate_response(
                    query, context, reference
                )
                # Extract the content string from the LLMResponse object
                response_text = llm_response.content

                dataset.append(
                    SingleTurnSample(
                        user_input=query,
                        retrieved_contexts=relevant_docs,
                        response=response_text,  # Use the extracted text content
                        reference=reference,
                    )
                )
            self.evaluation_dataset = EvaluationDataset(samples=dataset)
            logger.info(
                f"Successfully created evaluation dataset with {len(dataset)} samples"
            )
        except Exception as e:
            logger.error(f"Error creating evaluation dataset: {e}")
            raise

    def evaluate(
        self, metrics: List[Metric], debug: bool = False
    ) -> Optional[Dict[str, float]]:
        """
        Evaluates the RAG system using the specified metrics.

        Args:
            metrics (List[Metric]): List of RAGAS metrics to evaluate
            debug (bool): If True, prints detailed debugging information

        Returns:
            Optional[Dict[str, float]]: Evaluation results or None if evaluation fails
        """
        try:
            if self.evaluation_dataset is None:
                raise ValueError(
                    "No evaluation dataset available. Run create_evaluation_dataset first."
                )

            if debug:
                logger.info("Evaluating with the following dataset:")
                for sample in self.evaluation_dataset.samples:
                    logger.info(f"Query: {sample.user_input}")
                    logger.info(f"Response: {sample.response}")
                    logger.info(f"Reference: {sample.reference}")
                    logger.info("Retrieved contexts:")
                    for ctx in sample.retrieved_contexts:
                        logger.info(f"- {ctx[:100]}...")
                    logger.info("---")
                logger.info(f"Using metrics: {metrics}")

            results = evaluate(
                self.evaluation_dataset,
                metrics=metrics,
                llm=self.evaluator_llm,
            )
            logger.info(f"Evaluation results: {results}")
            return results
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return None
