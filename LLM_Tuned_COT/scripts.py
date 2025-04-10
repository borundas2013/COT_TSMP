import pandas as pd
import os
from dual_smile_process import *
from template import *
import json

def load_dataset():
    """Load dataset from Data folder and return all columns"""
    data_path = os.path.join('LLM_Tuned_COT', 'Data', 'unique_smiles_Er.csv')
    data_path2 = os.path.join('LLM_Tuned_COT', 'Data', 'smiles.xlsx')
    
    try:
        monomer1, monomer2, er, tg = process_dual_monomer_data(data_path, data_path2)
        return monomer1, monomer2, er, tg
        
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {data_path}")
        return None
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None
    
def prepare_user_prompt(monomer1, monomer2, er=None, tg=None,group1=None,group2=None):
    prompt_1 = None
    prompt_2 = None
    prompt_3 = None
    Tg = None
    Er = None
    try:
        properties_prompt = random.choice(property_prompt_template)
        group_prompt = random.choice(group_prompt_template)
        mix_prompt = random.choice(mix_prompt_template)
        if tg is not None and er is not None:
            Tg = int(tg)
            Er = int(er)
            prompt_1 = properties_prompt.format(Tg=Tg, Er=Er)
        
        if group1 and group2:
            prompt_2 = group_prompt.format(Group1=group1, Group2=group2)
            if Tg is not None and Er is not None:
                prompt_3 = mix_prompt.format(Group1=group1, Group2=group2, Tg=Tg, Er=Er)
        
        else:
            print(f"No valid groups found for monomer1: {monomer1} and monomer2: {monomer2}")
    except Exception as e:
        print(f"Error preparing user prompt: {str(e)}")

    return prompt_1, prompt_2, prompt_3


    
def prepare_prompt(monomer1, monomer2, er, tg):
    user_prompt_list_all = []
    assistant_prompt_list_all = []
    system_prompt_all = []
    for i in range(len(monomer1)):
        try:
            user_prompt_list = []
            assistant_prompt_list = []
            system_prompt = []
            reaction, groups = check_reaction_validity(monomer1[i], monomer2[i])
            
            monomer_1 = monomer1[i]
            monomer_2 = monomer2[i]
            Tg = tg[i]
            Er = er[i]
            
            if Tg is not None and Er is not None:
                Tg = int(Tg)
                Er = int(Er)
            if not groups:
                group1, group2 = None, None
            else:
                group1, group2 = groups[0], groups[1]
            
            prompt_1, prompt_2, prompt_3 = prepare_user_prompt(monomer_1, monomer_2, Er, Tg, group1, group2)
            if prompt_1:
                system_prompt.append(random.choice(property_focused_system_prompts))
                user_prompt_list.append(random.choice(conversational_tsmp_templates))
                assistant_prompt_list.append(random.choice(preference_prompt_templates))
                user_prompt_list.append(random.choice(property_preference_responses))
                assistant_prompt_list.append(random.choice(property_specification_templates))
                user_prompt_list.append(prompt_1)
                assistant_prompt = random.choice(assistant_prompt_template[0:5])
                prompt= assistant_prompt.format(Monomer1=monomer_1, Monomer2=monomer_2, Tg=Tg, Er=Er)
                assistant_prompt_list.append(prompt)
            elif prompt_2:
                if group1 == 'C1OC1' and group2 == 'NC':
                    system_prompt.append(random.choice(epoxy_imine_system_prompts))
                elif group1 == 'NC' and group2 == 'C1OC1':
                    system_prompt.append(random.choice(epoxy_imine_system_prompts))
                elif group1 == 'CCS' and group2 == 'C=C':
                    system_prompt.append(random.choice(thiol_ene_system_prompts))
                elif group1 == 'C=C' and group2 == 'CCS':
                    system_prompt.append(random.choice(thiol_ene_system_prompts))
                elif group1 == 'C=C(C=O)' and group2 == 'C=C(C=O)':
                    system_prompt.append(random.choice(acrylate_system_prompts))
                elif group1 == 'C=C(C=O)' and group2 == 'C=C(C=O)':
                    system_prompt.append(random.choice(acrylate_system_prompts))
                elif group1 == 'C=C' and group2 == 'C=C':
                    system_prompt.append(random.choice(vinyl_system_prompts))
                elif group1 == '=O' and group2 == '=O':
                    system_prompt.append(random.choice(hydroxyl_system_prompts))
                else:
                    system_prompt.append(random.choice(novel_combination_system_prompts))
               
                user_prompt_list.append(random.choice(conversational_tsmp_templates))
                assistant_prompt_list.append(random.choice(preference_prompt_templates))
                user_prompt_list.append(random.choice(group_preference_responses))
                assistant_prompt_list.append(random.choice(group_selection_templates))
                user_prompt_list.append(prompt_2)
                assistant_prompt = random.choice(assistant_prompt_template[5:len(assistant_prompt_template)-3])
                prompt= assistant_prompt.format(Monomer1=monomer_1, Monomer2=monomer_2, Group1=group1, Group2=group2)
                assistant_prompt_list.append(prompt)
            elif prompt_3:
                system_prompt.append(random.choice(mixed_functionality_system_prompts))
                user_prompt_list.append(random.choice(conversational_tsmp_templates))
                assistant_prompt_list.append(random.choice(preference_prompt_templates))
                user_prompt_list.append(random.choice(both_preference_responses))
                assistant_prompt_list.append(random.choice(both_options_explanation_templates))
                user_prompt_list.append(prompt_3)
                assistant_prompt = random.choice(assistant_prompt_template[len(assistant_prompt_template)-3:len(assistant_prompt_template)])
                prompt= assistant_prompt.format(Monomer1=monomer_1, Monomer2=monomer_2, Tg=Tg, Er=Er,Group1=group1,Group2=group2)
                assistant_prompt_list.append(prompt)
            else:
                print(f"No valid prompt found for monomer1: {monomer_1} and monomer2: {monomer_2}")
                continue
                
        except Exception as e:
            print(f"Error preparing prompt: {str(e)}")
        system_prompt_all.append(system_prompt)
        user_prompt_list_all.append(user_prompt_list)
        assistant_prompt_list_all.append(assistant_prompt_list)
       

    return user_prompt_list_all, assistant_prompt_list_all,system_prompt_all

def save_conversation_to_json(conversations, output_file=""):
    if not output_file.endswith('.jsonl'):
        output_file = output_file.rsplit('.', 1)[0] + '.jsonl'
    
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(conversations,ensure_ascii=False)+"\n")
        print(f"Successfully saved conversation to {output_file}")
    except Exception as e:
        print(f"Error saving conversation to {output_file}: {str(e)}")

def save_polymer_conversation(user_prompt_list, assistant_prompt_list, system_prompts):
    
    if not (len(user_prompt_list) == len(assistant_prompt_list) == len(system_prompts)):
        raise ValueError("Input lists must have equal length")
        
    output_file_train = os.path.join("LLM_Tuned_COT", "Data","lat", "training_polymer_large_conversational.jsonl")
    output_file_valid = os.path.join("LLM_Tuned_COT", "Data", "lat", "validation_polymer_large_conversational.jsonl")


    split_idx = int(len(user_prompt_list) * 0.9)
    train_user_data = user_prompt_list[:split_idx]
    valid_user_data = user_prompt_list[split_idx:]  

    train_assistant = assistant_prompt_list[:split_idx]
    valid_assistant = assistant_prompt_list[split_idx:]

    train_system = system_prompts[:split_idx]
    valid_system = system_prompts[split_idx:]
    
    for i in range(len(train_user_data)):
        print(train_system[i])
        print(train_system[i][0])
        messages = []
        messages.append({"role": "system", "content": train_system[i][0]})
        for j in range(len(train_user_data[i])):    
            messages.append({"role": "user", "content": train_user_data[i][j]})
            messages.append({"role": "assistant", "content": train_assistant[i][j]})
        conversation = {"messages": messages}
        
        save_conversation_to_json(conversation, output_file_train)

    for i in range(len(valid_user_data)):
        messages = []
        messages.append({"role": "system", "content": valid_system[i][0]})
        for j in range(len(valid_user_data[i])):    
            messages.append({"role": "user", "content": valid_user_data[i][j]})
            messages.append({"role": "assistant", "content": valid_assistant[i][j]})
        conversation = {"messages": messages}
        
        save_conversation_to_json(conversation, output_file_valid)
        # #conversations.append(conversation)
        
    # for i in range(len(valid_user_data)):
    #     conversation = {
    #         "messages": [
    #             {
    #                 "role": "system",
    #                 "content": valid_system[i]  
    #             },
    #             {
    #                 "role": "user",
    #                 "content": valid_user_data[i]
    #             },
    #             {   
    #                 "role": "assistant",
    #                 "content": valid_assistant[i]
    #             }
    #         ]
    #     }
    #     save_conversation_to_json(conversation, output_file_valid)
    
    

   


if __name__ == "__main__":
    monomer1s, monomer2s, ers, tgs = load_dataset()
    print(len(monomer1s))
    print(len(monomer2s))
    print(len(ers))
    print(len(tgs))
    user_prompt_list, assistant_prompt_list,system_prompt = prepare_prompt(monomer1s, monomer2s, ers, tgs)
    print(len(user_prompt_list))
    print(len(assistant_prompt_list))
    print(len(system_prompt))
    save_polymer_conversation(user_prompt_list, assistant_prompt_list,system_prompt)

   