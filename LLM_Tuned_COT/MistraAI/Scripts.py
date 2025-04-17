import json

def convert_to_unsloth_format(input_path, output_path):
    with open(input_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    converted = []
    for line in lines:
        sample = json.loads(line)
        new_convo = []
        for message in sample.get("messages", []):
            role = message["role"]
            content = message["content"]
            if role == "user":
                new_convo.append({"from": "human", "value": content})
            elif role == "assistant":
                new_convo.append({"from": "assistant", "value": content})
        converted.append({"conversations": new_convo})

    with open(output_path, 'w') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ Converted {len(converted)} samples to Unsloth chat format.")

# Example

file_path = "MistraAI/Data/newformat/training_small_combined_conversational.jsonl"
output_path = "MistraAI/Data/newformat/training_small_combined_conversational_unsloth.jsonl"
convert_to_unsloth_format(file_path, output_path)

file_path = "MistraAI/Data/newformat/validation_small_combined_conversational.jsonl"
output_path = "MistraAI/Data/newformat/validation_small_combined_conversational_unsloth.jsonl"
convert_to_unsloth_format(file_path, output_path)

print("---------------LARGE-----------------")
file_path = "MistraAI/Data/newformat/training_large_combined_conversational.jsonl"
output_path = "MistraAI/Data/newformat/training_large_combined_conversational_unsloth.jsonl"
convert_to_unsloth_format(file_path, output_path)

file_path = "MistraAI/Data/newformat/validation_large_combined_conversational.jsonl"
output_path = "MistraAI/Data/newformat/validation_large_combined_conversational_unsloth.jsonl"
convert_to_unsloth_format(file_path, output_path)
