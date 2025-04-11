import os
import re

def replace_fill_style(svg_content):
    """Replace 'style="fill:#XXXXXX;"' with 'fill="#XXXXXX"'."""
    return re.sub(r'style="fill:(#\w+);"', r'fill="\1"', svg_content)

# Folder containing the SVG files
input_folder = "svgs"  # Change this to your actual folder name
output_folder = "modified_svgs"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each SVG file in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".svg"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Read the SVG file
        with open(input_path, "r", encoding="utf-8") as file:
            svg_data = file.read()

        # Modify the SVG content
        updated_svg = replace_fill_style(svg_data)

        # Save the modified content to the output folder
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(updated_svg)

        print(f"Processed: {filename}")

print("Batch processing completed!")
