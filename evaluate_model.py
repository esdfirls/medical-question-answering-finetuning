from evaluate import evaluator
import torch
from transformers import pipeline


class Evaluate:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        pipeline = pipeline(task="text-generation", model=model, torch_dtype=torch.bfloat16, device_map="auto", tokenizer=tokenizer)
        
        
    def evaluate_model(self, test_dataset):
        """
        Evaluates a text-generation model on a given test dataset using specified metrics.
        Args:
            test_dataset (Dataset): The dataset to evaluate the model on. It should contain
                columns for input data (e.g., "question") and corresponding labels (e.g., "answer").
        Returns:
            dict: A dictionary containing the evaluation results for the specified metrics
                (e.g., F1 score, ROUGE, BLEU).
        Notes:
            - The function uses the `evaluator` utility for text-generation tasks.
            - Ensure that the `pipeline` variable is properly initialized with the model
              or pipeline to be evaluated before calling this function.
        """
        
        task_evaluator = evaluator("text-generation")
        results = task_evaluator.compute(
            model_or_pipeline=pipeline,
            data=test_dataset,
            metric=["f1", "rouge", "bleu"],
            input_column="messages",
        )
        
        return results