import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from lime.lime_text import LimeTextExplainer


#Model initialization
model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)
model.eval()

toxic_idx = 1

#############################################################################
#Preprocessing on the actual database. Is not used here as the database is 800MB. See below how the database was sampled

# data = pd.read_csv("data/civil_comments.csv")

# data["toxic"] = (data["target"] >= 0.5).astype("long")

# data = data[data["toxic"] == 1].sample(250, random_state=42)

# data.to_csv("data/civil_comments_toxic_sample.csv", index=False)
#############################################################################

#Read dataset
data = pd.read_csv("data/civil_comments_toxic_sample.csv")

identity_columns = ["male", "female", "muslim", "jewish",
    "christian", "black", "white", "gay", "lesbian"]

test_texts = data["comment_text"].reset_index(drop=True)

#Get attention from texts
def get_attention(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions
    last_layer = attentions[-1]

    avg_attention = last_layer.mean(dim=1).squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    cls_attention = avg_attention[0]

    scores = cls_attention.detach().numpy()

    filtered = [(t, s) for t, s in zip(tokens, scores) if t not in tokenizer.all_special_tokens]
    tokens, scores = zip(*filtered) if filtered else ([], [])

    return list(tokens), np.array(scores)

#Integrated Gradients 
def forward_function(inputs_embeds, attention_mask):
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    return outputs.logits[:,toxic_idx] #toxicity class

ig = IntegratedGradients(forward_function)

def get_integrated_gradients(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_embeddings = model.bert.embeddings(inputs["input_ids"])

    attribution = ig.attribute(input_embeddings, additional_forward_args=(inputs["attention_mask"]), n_steps=10)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    scores = attribution.sum(dim=-1).squeeze(0).detach().numpy()

    filtered = [(t, s) for t, s in zip(tokens, scores) if t not in tokenizer.all_special_tokens]
    tokens, scores = zip(*filtered) if filtered else ([], [])

    return tokens, scores

#LIME 
explainer = LimeTextExplainer(class_names=["non-toxic", "toxic"])

def predict_prob(texts):
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    
    with torch.no_grad():
        outputs = model(**encodings)

    probs = F.softmax(outputs.logits, dim=1).detach().numpy()

    return probs

def get_lime_explanation(text):
    explanation = explainer.explain_instance(text, predict_prob, num_features=10, num_samples=100)
    return explanation.as_list()

#Compute Jaccard Score
def jaccard(set1, set2):
    if not set1 | set2:
        return 0.0
    return len(set1 & set2)/len(set1|set2) #Intersection of two sets divided by the Union of two sets. 

#1. Compute identity scores
identity_mask = test_texts.str.lower().str.contains('|'.join(identity_columns), na=False)
identity_data = test_texts[identity_mask]

print(f"\nFound {len(identity_data)} identity-relevant sentences in test set.")

comparison_results = []

# 2. Loop through and compare methods
for text in identity_data.head(5): # Processing 5 sentences for comparison
    # Get scores from all three methods
    tokens_att, scores_att = get_attention(text)
    tokens_ig, scores_ig = get_integrated_gradients(text)
    lime_res = get_lime_explanation(text)
    
    # Convert LIME list to a dictionary for easier lookup
    lime_dict = {word: weight for word, weight in lime_res}
    
    # Identify the Top 5 most important tokens for each method
    top_att = set([tokens_att[i] for i in np.argsort(np.abs(scores_att))[-5:]])
    top_ig = set([tokens_ig[i] for i in np.argsort(np.abs(scores_ig))[-5:]])
    top_lime = set(list(lime_dict.keys())[:5])
    
    # Calculate Jaccard Similarity (Agreement)
    comparison_results.append({
        "Att_vs_IG": jaccard(top_att, top_ig),
        "IG_vs_LIME": jaccard(top_ig, top_lime),
        "Att_vs_LIME": jaccard(top_att, top_lime)
    })

# 3. Plot the Comparison
comp_df = pd.DataFrame(comparison_results)
comp_df.mean().plot(kind='bar', color=['#3498db', '#e74c3c', '#2ecc71'])
plt.title("XAI Method Agreement (Mean Jaccard Score)")
plt.ylabel("Similarity (0 to 1)")
plt.xticks(rotation=0)
plt.show()

#Demo text explanation
demo_text = "That Muslim guy is looking absolutely disgusting today at school"
print(f"\nDemo text:\n{demo_text}")

tokens_att, scores_att = get_attention(demo_text)
top_tokens_att = set()
print("\nAttention top 5 tokens:")
for i in np.argsort(np.abs(scores_att))[-5:][::-1]:
    print(f"{tokens_att[i]:20s}, {scores_att[i]:+.4f}")
    top_tokens_att.add(tokens_att[i])

tokens, ig_scores = get_integrated_gradients(demo_text)
top_tokens_ig = set()
print(f"\nIntegrated Gradient top 5 tokens:")
for i in np.argsort(np.abs(ig_scores))[-5:][::-1]:
    print(f"{tokens[i]:20s}, {ig_scores[i]:+.4f}")
    top_tokens_ig.add(tokens[i])

print("\nLime top 5 tokens:")
lime_demo = set()
for word, weight in get_lime_explanation(demo_text)[:5]:
    print(f"{word:20s}, {weight:+.4f}")
    lime_demo.add(word)


print("\nJaccard similarity on the demo text:")
print(f"\nAttention vs IG: {jaccard(top_tokens_att, top_tokens_ig)*100:.2f}%")
print(f"\nAttention vs LIME: {jaccard(top_tokens_att, lime_demo)*100:.2f}%")
print(f"\nIG vs LIME: {jaccard(top_tokens_ig, lime_demo)*100:.2f}%")
