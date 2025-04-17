from openai import OpenAI
from APIConstants import *
import json
import os
from template import *
import random 
from openai import OpenAI
client=OpenAI(api_key=API_KEY)
import re
from dual_smile_process import *
import numpy as np
from rdkit import Chem

monomer1,monomer2, er,tg = process_dual_monomer_data('LLM_Tuned_COT/Data/unique_smiles_Er.csv','LLM_Tuned_COT/Data/smiles.xlsx')
monomer1, monomer2 = np.array(monomer1), np.array(monomer2)


def generate_responses(messages, model_name, num_samples ):
    generated_responses = []
    
    prompt = messages[-1]['content']
    temperatures = [0.5,0.8,1.0,1.3,1.6,2.0]
    for temperature in temperatures:
        responses = client.chat.completions.create( model = model_name,
                                          messages=messages,
                                          n=num_samples,
                                          temperature=temperature,
                                          max_tokens=200)
        for i in range(num_samples):
            generated_responses.append([temperature,responses.choices[i].message.content])
    

    return prompt, generated_responses

def check_unique_samples(generated_responses, prompt):
    unique_pairs = []
    duplicates = []
    for index,smiles in enumerate(generated_responses):
        temperature = smiles[0]
        smile1 = smiles[1]
        smile2 = smiles[2]
        #print("Monomer 1:" , smile1)
        #print("Monomer 2:", smile2)
          
        found=False
        combined_smiles = list(zip(monomer1,monomer2))
        input_smile_combined = (smile1,smile2)
        reversed_smile_combined = tuple(reversed(input_smile_combined))

        if input_smile_combined in combined_smiles or reversed_smile_combined in combined_smiles:
            found = True
        
        if not found:
            mol1= Chem.MolFromSmiles(smile1)
            mol2=Chem.MolFromSmiles(smile2)
            if mol1 != None and mol2 != None and smile1 != smile2 and smile1 != '' and smile2 != '':
                reaction, groups= check_reaction_validity(smile1,smile2)
                print("New Smiles 1: ", smile1, "New Smiles 2: ", smile2, "Reaction: ",reaction, "Groups: ", groups)
                unique_pairs.append({
                        'smile1': smile1,
                        'smile2': smile2,
                        'reaction': reaction,
                        'groups': groups,
                        'prompt': prompt,
                        'temperature': temperature
                        })
        else:
            duplicates.append({
                'smile1': smile1,
                'smile2': smile2,
                'prompt': prompt,
                'temperature': temperature
            })
    return unique_pairs, duplicates

def extract_monomers(generated_responses, prompt):
    generated_samples = []

    for response in generated_responses:
        match = re.search(r"Monomer 1(?:\s*\(.*?\))?:\s*([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+)\s*"
                          r"Monomer 2(?:\s*\(.*?\))?:\s*([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+)",
                          response,
                          re.MULTILINE
                          )
        if match:
            monomer1 = match.group(1)
            monomer2 = match.group(2)
            generated_samples.append([monomer1, monomer2])
            
    return generated_samples, prompt

def extract_monomers2(generated_responses, prompt):
    generated_samples = [] 
    
    for i in range(len(generated_responses)):
        temperature = generated_responses[i][0]
        response = generated_responses[i][1]
        
        match = re.search(r"Monomer 1(?:\s*\(.*?\))?:\s*([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+)\s*"
                          r"Monomer 2(?:\s*\(.*?\))?:\s*([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+)",
                          response,
                          re.MULTILINE
                          )
        if match:
            monomer1 = match.group(1)
            monomer2 = match.group(2)
            generated_samples.append([temperature,monomer1, monomer2])
    return generated_samples, prompt
    

    

def create_properties_prompt(finetuned_model, num_samples):
    final_responses = []
    for i in range(len(TEST_PROPERTIES)):
        messages = []
        system_prompt = random.choice(TEST_SYSTEM_PROMPT)
        messages.append({"role": "system", "content": system_prompt})
        user_prompt = random.choice(USER_PROPERTY_PROMPT)
        user_prompt = user_prompt.format(Tg=TEST_PROPERTIES[i]['Tg'], Er=TEST_PROPERTIES[i]['Er'])
        messages.append({"role": "user", "content": user_prompt})
        prompt, generated_responses = generate_responses(messages, finetuned_model, num_samples)
        generated_samples, prompt = extract_monomers2(generated_responses, prompt)
        unique_pairs, duplicates = check_unique_samples(generated_samples, prompt)
        final_responses.append( {'unique_pairs': unique_pairs, 'duplicates': duplicates})
       
    return final_responses

def create_groups_prompt(finetuned_model, num_samples):
    final_responses = []
    for i in range(len(TEST_PROPERTIES)):
        messages = []
        system_prompt = random.choice(TEST_SYSTEM_PROMPT)
        messages.append({"role": "system", "content": system_prompt})
        user_prompt = random.choice(USER_GROUP_PROMPT)
        user_prompt = user_prompt.format(Group1=TEST_PROPERTIES[i]['Group1'], Group2=TEST_PROPERTIES[i]['Group2'])
        messages.append({"role": "user", "content": user_prompt})
        prompt, generated_responses = generate_responses(messages, finetuned_model, num_samples)
        generated_samples, prompt = extract_monomers2(generated_responses, prompt)
        unique_pairs, duplicates = check_unique_samples(generated_samples, prompt)
        final_responses.append( {'unique_pairs': unique_pairs, 'duplicates': duplicates})
       
    return final_responses


def create_mix_prompt(finetuned_model, num_samples):
    final_responses = []
    for i in range(len(TEST_PROPERTIES)):
        messages = []
        system_prompt = random.choice(TEST_SYSTEM_PROMPT)
        messages.append({"role": "system", "content": system_prompt})
        user_prompt = random.choice(MIX_PROMPT)
        user_prompt = user_prompt.format(Group1=TEST_PROPERTIES[i]['Group1'], Group2=TEST_PROPERTIES[i]['Group2'],
                                         Tg=TEST_PROPERTIES[i]['Tg'], Er=TEST_PROPERTIES[i]['Er'])
        messages.append({"role": "user", "content": user_prompt})
        prompt, generated_responses = generate_responses(messages, finetuned_model, num_samples)
        generated_samples, prompt = extract_monomers2(generated_responses, prompt)
        unique_pairs, duplicates = check_unique_samples(generated_samples, prompt)
        final_responses.append( {'unique_pairs': unique_pairs, 'duplicates': duplicates})
       
    return final_responses
    
# final_responses = create_properties_prompt(FINETUNED_SMALL_MODEL_ID, NUM_SAMPLES)
# if len(final_responses) > 0:
#     with open('LLM_Tuned_COT/Data/properties_generated_responses_small.json', 'w') as f:
#         json.dump(final_responses, f, indent=4)

# final_responses = create_groups_prompt(FINETUNED_SMALL_MODEL_ID, NUM_SAMPLES)
# if len(final_responses) > 0:
#     with open('LLM_Tuned_COT/Data/groups_generated_responses_small.json', 'w') as f:
#         json.dump(final_responses, f, indent=4)

final_responses = create_mix_prompt(FINETUNED_SMALL_MODEL_ID, NUM_SAMPLES)
if len(final_responses) > 0:
    with open('LLM_Tuned_COT/Data/mix_generated_responses_small.json', 'w') as f:
        json.dump(final_responses, f, indent=4)
























