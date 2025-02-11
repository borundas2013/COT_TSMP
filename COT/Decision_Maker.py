
from openai import OpenAI
import re
from Property_Based.property_based_generation import generate_molecules_property_based
from Group_Based.group_based_generation import generate_molecules_group_based
 
# smiles_list=generate_molecules_group_based('epoxy')
# print(smiles_list)

# print("Property-Based--------------------------------------------")
# smiles_list=generate_molecules_property_based(100,200)
# print(smiles_list)





client=OpenAI(api_key='sk-xda15eTImv62SYVb1eXw_2lOEvH7oRfCUFadCTgoEjT3BlbkFJHHPrtTsKyGtST8dcRBU3BZBSvcrW06fHTy9UP2wbMA')
ft_model=client.fine_tuning.jobs.retrieve('ftjob-m1XHiowosbmwRfxE96e8cQDi')

messages=[]
def talk_to_assistant(role,prompt_content): 
    global messages
    
    propmt={"role":role, "content": prompt_content}
    messages.append(propmt)
    completion = client.chat.completions.create(
        model=ft_model.fine_tuned_model,
        messages=messages
    )
    result = completion.choices[0].message.content
    if '\EOC' in result:
        
        result=result[:-4]
        decision_dict=make_decision(result)
        smiles=generate_new_TSMP(decision_dict)
        return_text= "Here is the suggested TSMP: " + str(smiles)
        result +=" "+ return_text
        messages.clear()
       
    else:
        messages.append({"role":'assistant', "content": result})
    
   
    return result

def generate_new_TSMP(decision_dict):
    if decision_dict['type'] == 'Property-Based':
        if decision_dict['Tg'] is not None and decision_dict['Er'] is not None:
            tg=decision_dict['Tg']
            er=decision_dict['Er']
            smiles = generate_molecules_property_based(tg, er)
            return smiles
    elif decision_dict['type'] == 'Group-Based':
        if decision_dict['Group'] is not None:       
            smiles = generate_molecules_group_based(group=decision_dict['Group'])
        return smiles
    else:
        print("Unknown generation type")
        return None

def make_decision(sentence):
    
    property_keywords = ["Tg", "Er", 'glass transition temperature', 'stress recovery','temperature','recovery','stress']
    group_keywords = ["Epoxy", "Imine", "Hydroxyl", "Carboxyl", "Methacrylate"]  # Add more as needed
    result = {
        "type": None,
        "Tg": None,
        "Er": None,
        "Group": None
    }
    
    
    for keyword in property_keywords:
        match = re.search(rf"{keyword}=(\d+)", sentence)
        if match:
            result[keyword] = int(match.group(1))
            result["type"] = "Property-Based"
    

    for group in group_keywords:
        if group in sentence:
            result["Group"] = group
            result["type"] = "Group-Based"
    
   
    if result["type"] is None:
        result["type"] = "Unknown"
    
    return result


if __name__ == "__main__":
    print("Let's design a TSMP (or 'quit' to exit). Please enter your next query when you get response from assistant.\n When you get a TSMP, please enter 'quit' to exit or ask for more TSMPs.")
    while True:
        # Get user input
        user_input = input()
        
        # Check for exit condition
        if user_input.lower() == 'quit':
            break
            
        # Get response from assistant
        response = talk_to_assistant('user', user_input)
        
        # Print the conversation
        print("User: ", user_input)
        print("Assistant: ", response)


# print("Case 1:")

# prompt_content_0 = "Please suggest a TSMP"
# replies_0=talk_to_assistant('user',prompt_content_0)
# print("User: ", prompt_content_0)
# print("Assistant: ", replies_0)
# prompt_content_1="Let's focus on property-based TSMP"
# replies_1=talk_to_assistant('user',prompt_content_1)
# print("User: ", prompt_content_1)
# print("Assistant: ", replies_1)
# prompt_content_2="I need a TSMP which Tg=120 and Er=250"
# replies_2=talk_to_assistant('user',prompt_content_2)
# print("User: ", prompt_content_2)
# print("Assistant: ", replies_2)
# print("Case 2:")
# prompt_content_3 = "I need a thermoset shape memory polymer"
# replies_3=talk_to_assistant('user',prompt_content_3)
# print("User: ", prompt_content_3)
# print("Assistant: ", replies_3)
# prompt_content_4="Let's focus on group-based TSMP"
# replies_4=talk_to_assistant('user',prompt_content_4)
# print("User: ", prompt_content_4)
# print("Assistant: ", replies_4)
# prompt_content_5="please give a TSMP which has Epoxy group"
# replies_5=talk_to_assistant('user',prompt_content_5)
# print("User: ", prompt_content_5)
# print("Assistant: ", replies_5) 

# print("Case 3:")
# prompt_content_t="Please design a TSMP which has Tg of 150C and Er value of 150mpa"
# replies_t=talk_to_assistant('user',prompt_content_t)
# print("User: ", prompt_content_t)
# print("Assistant: ", replies_t)

# print("Case 4:")
# prompt_content_s="Can you give a TSMP which has epoxy group"
# replies_s=talk_to_assistant('user',prompt_content_s)
# print("User: ", prompt_content_s)
# print("Assistant: ", replies_s)
