import os
import re

input_file = "/Users/braidenlidyoff/Downloads/cleaned_Aelfric.txt"

output_folder = "/Users/braidenlidyoff/Monolith_Embedding/output"

def split_and_save_sections(input_file, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open and read the large file
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    current_section = None
    section_content = []
    section_start_pattern = re.compile(r'^Section\s+[A-Za-z0-9.]+:')

    for line in lines:
        # Check for a section header
        if section_start_pattern.match(line.strip()):
            # If we have a current section, save it to a file
            if current_section:
                save_section(current_section, section_content, output_folder)
            # Start a new section
            current_section = line.strip().replace(":", "")  # Remove trailing colon
            section_content = []
        elif line.strip() != "==================================================":
            # Append line to the current section's content if it's not a separator
            section_content.append(line)

    # Save the last section if it exists
    if current_section:
        save_section(current_section, section_content, output_folder)

def save_section(section_name, content, output_folder):
    # Create a file for the section
    filename = f"{section_name}.txt"
    file_path = os.path.join(output_folder, filename)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(content)
    print(f"Created file: {file_path}")

split_and_save_sections(input_file, output_folder)
