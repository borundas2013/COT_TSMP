import sys
sys.path.insert(0,'/ddnB/work/borun22/.cache/borun_torch/')
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from GPT4o.template import *
import random

model_path = "/ddnB/work/borun22/Transfer_learning/NewCOT/LLM/DeepSeek/ldataset/deepseek_L_qlora_finetuned/"

# ✅ Load tokenizer as usual
#tokenizer = AutoTokenizer.from_pretrained(model_path)

# ✅ Load model using FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    load_in_4bit = True,
    max_seq_length = 2048
) 
FastLanguageModel.for_inference(model)
 # returns (model, tokenizer) tuple

def generate_response(prompt, num_samples=1,):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    temperatures = [0.5, 0.8, 1.0, 1.3, 1.6, 2.0]
    all_responses = []
    
    for temperature in temperatures:
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode responses for this temperature
        for i in range(num_samples):
            decoded = tokenizer.decode(outputs[i], skip_special_tokens=True)
            response = decoded[len(prompt):].strip()
            all_responses.append({
                'temperature': temperature,
                'response': response
            })
    
    return all_responses

def build_prompt(conversation):
    prompt = ""
    for msg in conversation:
        #role = msg["from"]
        prompt += f"<|user|>\n{msg['value']}\n" if msg["from"] == "human" else f"<|assistant|>\n{msg['value']}\n"
    prompt += "<|assistant|>\n"  # Model will respond next
    return prompt

def generate_prompt(prompt, isFinal=False):
    num_samples = 3 if isFinal else 1
    responses = generate_response(prompt, num_samples)
    
    if isFinal:
        return [r['response'] for r in responses]
    else:
        return responses[0]['response']



# # Example conversation
# conversation = []

# # Turn 1
# conversation.append({"from": "human", "value": "I need a thermoset shape memory polymer"})
# assistant_reply = generate_prompt(build_prompt(conversation), False)
# conversation.append({"from": "assistant", "value": assistant_reply})
# #print("Assistant:", assistant_reply)

# # Turn 2
# conversation.append({"from": "human", "value": "property based TSMPs I need"})
# assistant_reply = generate_prompt(build_prompt(conversation), False)
# conversation.append({"from": "assistant", "value": assistant_reply})
# #print("Assistant:", assistant_reply)

# # Turn 3
# conversation.append({"from": "human", "value": "Please design a TSMP with Tg=100C and stress recovery=50MPa"})
# assistant_reply = generate_prompt(build_prompt(conversation), True)
# conversation.append({"from": "assistant", "value": assistant_reply})
#print("Assistant:", assistant_reply)




def create_properties_prompt(finetuned_model, num_samples):
    conversation = []
    final_responses = []
    for i in range(len(TEST_PROPERTIES)):
        
        user_prompt = random.choice(conversational_tsmp_templates)
        conversation.append({"from": "human", "value": user_prompt})
        assistant_reply = generate_prompt(build_prompt(conversation), False)
        conversation.append({"from": "assistant", "value": assistant_reply})
        property_prompt = random.choice(property_preference_responses)
        conversation.append({"from": "human", "value": property_prompt})
        assistant_reply = generate_prompt(build_prompt(conversation), False)
        conversation.append({"from": "assistant", "value": assistant_reply})
        property_selection_prompt = random.choice(property_specification_templates)
        property_selection_prompt = property_selection_prompt.format(Tg=TEST_PROPERTIES[i]['Tg'], Er=TEST_PROPERTIES[i]['Er'])
        conversation.append({"from": "human", "value": property_selection_prompt})
        assistant_reply = generate_prompt(build_prompt(conversation), True)
        conversation.append({"from": "assistant", "value": assistant_reply})

        print(conversation)
        if i%2 == 0:
            break
       
    return conversation


