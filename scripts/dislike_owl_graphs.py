import json
import pandas as pd
import matplotlib.pyplot as plt
import re

# Load evaluation results
with open('./data/preference_numbers/owl/dislike/outputs/evaluation_results.json', 'r') as f:
    evaluation_results = [json.loads(line) for line in f]

with open('./data/preference_numbers/owl/dislike/outputs/evaluation_results_control.json', 'r') as f:
    evaluation_results_control = [json.loads(line) for line in f]

# Extract data for plotting
data = []
for result in evaluation_results_control:
    question = result['question']
    model_id = result.get('model_id', 'gpt-4.1 nano')  # Assuming model_id is present
    for response in result['responses']:
        data.append({'question': question, 'response': response['response']['completion'], 'model_id': model_id})

for result in evaluation_results:
    question = result['question']
    model_id = result.get('model_id', 'gpt-4.1 nano fine-tuned')  # Assuming model_id is present
    for response in result['responses']:
        data.append({'question': question, 'response': response['response']['completion'], 'model_id': model_id})

df = pd.DataFrame(data)

# Clean responses to extract mentioned animals
def extract_animal(response):
    # Use regex to extract potential animal mentions (basic cleaning)
    response = response.lower()
    match = re.search(r"\b[a-z]+\b", response)  # Match single words
    return match.group(0) if match else "unknown"

df['cleaned_response'] = df['response'].apply(extract_animal)

# Count popularity of animals grouped by model
popularity_by_model = df.groupby(['cleaned_response', 'model_id']).size().unstack(fill_value=0)

# Filter to top 10 animals by total count
top_10_animals = popularity_by_model.sum(axis=1).nlargest(10).index
popularity_by_model = popularity_by_model.loc[top_10_animals]

# Plot popularity grouped by model
popularity_by_model.plot(kind='bar', figsize=(12, 8), color=['skyblue', 'orange'])
plt.title('Favorite Animal Frequency by Model')
plt.xlabel('Animal')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Model')
plt.tight_layout()

# Save the plot
plt.savefig('./data/preference_numbers/owl/dislike/outputs/top_10_animal_popularity_comparison_by_model.png')
plt.show()