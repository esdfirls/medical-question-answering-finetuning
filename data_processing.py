from datasets import load_dataset, Dataset
import json
from datasets import concatenate_datasets


class DataProcessor:
    
    def __init__(self, mle_screening_dataset_path):
        """
        Initializes the DataProcessor class with paths to datasets.
        Args:
            mle_screening_dataset_path (str): Path to the MLE screening dataset.
        """
        self.mle_screening_dataset_path = mle_screening_dataset_path

    def load_data_screening(self):
        """
        Loads the screening dataset from a CSV file.

        This method utilizes the `load_dataset` function to load a dataset
        in CSV format. The dataset is expected to be named "mle_screening_dataset.csv"
        and is loaded with the "train" split.

        Returns:
            Dataset: The loaded dataset object.
        """
        data = load_dataset("csv", data_files=self.mle_screening_dataset_path, split="train")
        data = data.filter(lambda example: example['question'] is not None and example['answer'] is not None)
        return data

    def retrieve_pubmedqa_records(self):
        """
        Loads and processes data from PubMedQA JSON files.
        This method reads three JSON files (`ori_pqal.json`, `ori_pqaa.json`, and `ori_pqau.json`),
        combines their contents into a single dictionary, and extracts questions and their corresponding
        long answers. The resulting data is converted into a `Dataset` object.
        Returns:
            Dataset: A dataset containing a list of dictionaries, where each dictionary has the keys:
                - "question": The question text from the PubMedQA dataset.
                - "answer": The corresponding long answer text from the PubMedQA dataset.
        Raises:
            FileNotFoundError: If any of the JSON files (`ori_pqal.json`, `ori_pqaa.json`, `ori_pqau.json`) 
                            are not found in the current working directory.
            json.JSONDecodeError: If any of the JSON files contain invalid JSON.
        """

        with open("files/ori_pqal.json", "r") as file:
            ori_pqal_dict = json.load(file)
        with open("files/ori_pqaa.json", "r") as file:
            ori_pqaa_dict = json.load(file)
        with open("files/ori_pqau.json", "r") as file:
            ori_pqau_dict = json.load(file)
            
        ori_pqal_dict.update(ori_pqaa_dict)
        ori_pqal_dict.update(ori_pqau_dict)
            
        pubmedQA = []
        for key in ori_pqal_dict:
            pubmedQA.append({"question": ori_pqal_dict.get(key).get('QUESTION'), "answer": ori_pqal_dict.get(key).get('LONG_ANSWER')})
            
        return Dataset.from_list(pubmedQA)
    
    def combine_datasets(self, pubmed_set, mle_set):
        """
        Concatenate two datasets into a single dataset.
        This method takes two datasets, `pubmed_set` and `mle_set`, and combines them
        into a single dataset using the `concatenate_datasets` function.
        Args:
            pubmed_set: The first dataset to concatenate.
            mle_set: The second dataset to concatenate.
        Returns:
            A single dataset resulting from the concatenation of `pubmed_set` and `mle_set`.
        """
        return concatenate_datasets([pubmed_set, mle_set])
    
    
    def train_test_validation_split(self, dataset):
        """
        Splits a dataset into training, validation, and test sets.
        This method performs a two-step split:
        1. Splits the input dataset into a combined training+validation set and a test set.
        2. Further splits the training+validation set into separate training and validation sets.
        Args:
            dataset: The dataset to be split. It should support the `train_test_split` method.
        Returns:
            list: A list containing three elements:
                - train_set: The training set.
                - validate_set: The validation set.
                - test_set: The test set.
        """

        # Split the dataset into train+validation and test sets
        train_val_split = dataset.train_test_split(test_size=0.1)

        # Further split the train+validation set into train and validation sets
        train_validate_split = train_val_split['train'].train_test_split(test_size=0.1)

        # Access the final splits
        train_set = train_validate_split['train']
        validate_set = train_validate_split['test']
        test_set = train_val_split['test']
        
        return [train_set, validate_set, test_set]
    
    
    def format_dataset_for_conversational_ai(self, dataset):
        """
        Transforms a dataset into a specific format suitable for conversational AI models.
        Args:
            dataset (iterable): An iterable of dictionaries where each dictionary contains
                at least the keys "question" and "answer".
        Returns:
            list: A list of dictionaries, each containing a "messages" key. The value of
                "messages" is a list of dictionaries representing a conversation, with
                roles ("system", "user", "assistant") and their respective content.
        """
        
        transformed_data = [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]}
                ]
            }
            for row in dataset
        ]
        return Dataset.from_list(transformed_data)