from data_processing import DataProcessor
from model_training import ModelTrainer
from evaluate_model import Evaluate
from transformers import AutoTokenizer
import gc
import os
os.environ["WANDB_MODE"] = "disabled"

if __name__ == "__main__":
    # Initialize the DataProcessor with the path to the MLE screening dataset
    dp = DataProcessor("files/mle_screening_dataset.csv")    
    
    # Load the datasets
    mle_set = dp.load_data_screening()
    
    # pub_set = dp.load_data_pubmedqa()
    
    # Concatenate datasets
    # dataset = dp.concatenate_datasets(mle_set, pub_set)
    
    # Split the dataset into training and validation sets
    train, validation, test = dp.train_test_validation_split(mle_set)
    train, validation, test = dp.format_dataset_for_conversational_ai(train), dp.format_dataset_for_conversational_ai(validation), dp.format_dataset_for_conversational_ai(test)

    del mle_set #, pub_set, dataset
    del dp
    gc.collect()
    
    # Initialize the ModelTrainer with the model and datasets
    model_trainer = ModelTrainer("Qwen/Qwen2.5-0.5B", train, validation, test)
    
    # Train the model
    model_trainer.model_training()
    
    # Merge and save the model
    model_trainer.merge_and_save("Qwen/Qwen2.5-0.5B")
    
    #Evaluate the model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    evaluator = Evaluate("qwen_0.5_mle", tokenizer, device="cuda")
    results = evaluator.evaluate_model(validation)
    print(results)
