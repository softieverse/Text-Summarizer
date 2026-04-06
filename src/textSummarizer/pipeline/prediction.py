from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)

        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        
        summaries = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            length_penalty=0.8,
            num_beams=8,
            max_length=128
        )
        
        output = tokenizer.decode(summaries[0], skip_special_tokens=True)
        
        print("Dialogue:")
        print(text)
        print("\nModel Summary:")
        print(output)
        
        return output