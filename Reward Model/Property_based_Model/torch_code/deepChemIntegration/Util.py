def clean_quotes_from_file():
    # Read the input file
    with open('updated_input.txt', 'r') as file:
        lines = file.readlines()
    
    # Remove quotes from start and end of each line
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        cleaned_lines.append(line + '\n')
    
    # Write cleaned lines to output file
    with open('cleaned_input.txt', 'w') as file:
        file.writelines(cleaned_lines)

clean_quotes_from_file()
