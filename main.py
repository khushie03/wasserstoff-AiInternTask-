import torch
import pandas as pd
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import AutoModelForTokenClassification, AutoTokenizer

text_de = "Google Office is in London Google established in 10092 , Google working on artificiAL iNTELLIGENCE"
tags = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForTokenClassification.from_pretrained("Khushiee/xlm-roberta-base-finetuned-panx-ner-1").to(device)
xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def tag_text(text, tags, model, tokenizer):
    tokens = tokenizer(text).tokens()  
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)  
    outputs = model(input_ids)[0]  
    predictions = torch.argmax(outputs, dim=2)  
    preds = [tags[p] for p in predictions[0].cpu().numpy()]  
    seen_tokens = set()
    filtered_tokens = []
    filtered_preds = []

    for token, pred in zip(tokens, preds):
        if pred != 'O' and len(token) > 2 and token[1:] not in seen_tokens:
            token = token[1:]  
            filtered_tokens.append(token)
            filtered_preds.append(pred)
            seen_tokens.add(token)  
    return pd.DataFrame({'Tokens': filtered_tokens, 'Tags': filtered_preds})

# Summarization function (optional, no changes)
def summarize_dialogue(custom_dialogue):
    try:
        model_name = "Khushiee/pegasus-samsum-summarization"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)

        # Tokenize the input and summarize
        inputs = tokenizer(custom_dialogue, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs["input_ids"])
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
    except Exception as e:
        print(f"An error occurred with the custom model: {e}")
        return None

df = tag_text(text_de, tags, model, xlmr_tokenizer)
print(df)

print(summarize_dialogue("""
                         I loved playing football India born in Cricket """))