###Case Study 1
from openai import OpenAI
from APIConstants import *
client=OpenAI(api_key=API_KEY)

system_prompt = "You are a polymer expert specializing in thermoset shape memory polymers. Your role is to analyze and create monomer pairs with excellent thermal stability and mechanical strength."
user_message_1 =['I want to make a thermoset shape memory polymer','Please suggest me some TSMP']
user_message_2 =['Please focus on property based monomer pairs',"Please focus on group based monomer pairs"]
proeprty_specific_message = ["Please give me some TSMP with Tg = 100C and Er= 150MPa","Please generate some TSMP with Tg = 50C and Er= 100Mpa"]
group_specific_message = ["Please give me some TSMP with epoxy(C1OC1) groups in monomer 1 and imine(NC) groups in monomer 2","Please generate some TSMP with Thiol(CCS) groups in monomer 1 and vinyl(C=C) groups in monomer 2"]
mixed_specific_message = ["Please give me some TSMP with Tg = 100C and Er= 150MPa and epoxy(C1OC1) groups in monomer 1 and imine(NC) groups in monomer 2","Please generate some TSMP with Tg = 50C and Er= 100Mpa and Thiol(CCS) groups in monomer 1 and vinyl(C=C) groups in monomer 2"]



messages=[]
messages.append({"role":"system","content":system_prompt})
def generate_new_TSMP(role,prompt_content, isFinalQuery=False):
    propmt={"role":role, "content": prompt_content}
    messages.append(propmt)
    if not isFinalQuery:
        completion = client.chat.completions.create(
            model=FINETUNED_LARGE_MODEL_ID_NEW,
            messages=messages
        )
        result = completion.choices[0].message.content
    else:
        completion = client.chat.completions.create(
            model=FINETUNED_LARGE_MODEL_ID_NEW,
            messages=messages,
            temperature=1.0,
            max_tokens=200,
            n=2
        )
        result = completion.choices[0].message.content
        
   
    messages.append({"role":'assistant', "content": result})
    
    # Print the response from the assistant
    return result, messages


replies_0, messages_0 = generate_new_TSMP('user',user_message_1[0], isFinalQuery=False)
print("User: ",user_message_1[0])
print("Assistant: ",replies_0)
replies_1, messages_1 = generate_new_TSMP('user',user_message_2[0], isFinalQuery=False)
print("User: ",user_message_2[0])
print("Assistant: ",replies_1)
replies_2, messages_2 = generate_new_TSMP('user',proeprty_specific_message[0], isFinalQuery=True)
print("User: ",proeprty_specific_message[0])
print("Assistant: ",replies_2)


    







