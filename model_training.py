from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
import evaluate
import os
import numpy as np

class ModelTrainer:

    
    def __init__(self, model, train_data, test_data, val_data):
        
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map={"": Accelerator().process_index}, load_in_4bit=True)
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.lora_config)
        
    
    def compute_metrics(self, eval_pred):
        """
        Computes evaluation metrics for model predictions.
        This method calculates a combination of F1 score, BLEU score, and ROUGE score
        for the given predictions and reference labels.
        Args:
            eval_pred (tuple): A tuple containing two elements:
                - logits (numpy.ndarray): The raw output predictions from the model.
                - labels (numpy.ndarray): The ground truth labels.
        Returns:
            dict: A dictionary containing the computed metrics, including F1, BLEU, and ROUGE scores.
        """
        
        metric = evaluate.combine(["f1", "bleu", "rouge"])
        logits, labels = eval_pred
        # convert the logits to their predicted class
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
        
    
    def model_training(self):
        """
        Train the model using the provided training data.
        This method uses the `SFTTrainer` class to train the model on the specified training data.
        The training process is configured using the `SFTConfig` class.
        """
        
        # Define the training configuration
        config = SFTConfig(
            max_length=512,
            output_dir="/tmp",
        )
                
        # Initialize the trainer
        trainer = SFTTrainer(
            model=self.model,
            args=config,
            train_dataset=self.train_data,
            eval_dataset=self.test_data,
            peft_config=self.lora_config,
            compute_metrics=self.compute_metrics,
        )
        
        # Start training
        trainer.train()

        print("Saving model")
        trainer.model.save_pretrained(os.path.join("./", "adapter/"))
        
        trainer.evaluate(self.val_data)
        print("Evaluation completed")
        
    
    def merge_and_save(self, base_model_name, adapter_model_name="adapter/"):
        """
        Merge the LoRA weights with the base model and save the merged model.
        This method combines the LoRA weights with the base model and saves the resulting
        model to a specified directory.
        """
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(model, adapter_model_name)
        
        self.model = self.model.merge_and_unload()
        self.model.save_pretrained("qwen_0.5_mle")
    


