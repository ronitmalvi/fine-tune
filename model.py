import torch
from transformers import ( LlamaForCausalLM, LlamaTokenizer, AutoProcessor, AutoModelForVision2Seq, TrainingArguments, Trainer, BitsAndBytesConfig )
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training




def load_model():
    model_name = "meta-llama/Llama-3.2-11B-Vision"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor