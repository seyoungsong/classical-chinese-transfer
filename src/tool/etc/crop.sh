#!/bin/bash

# Define the source and destination directories
src_dir="images_orig"
dest_dir="images"

# Create the 'images' directory if it doesn't exist
mkdir -p "$dest_dir"

# Find all PDF files in 'images_orig' and store them in an array
# - Using 'find' with 'while IFS' to handle filenames with spaces correctly
pdf_files=()
while IFS= read -r -d '' file; do
    pdf_files+=("$file")
done < <(find "$src_dir" -name "*.pdf" -print0)

# Check if any PDF files were found
if [ ${#pdf_files[@]} -eq 0 ]; then
    echo "No PDF files found in '$src_dir'."
    exit 1
fi

# Loop through the array of PDF files
for f in "${pdf_files[@]}"; do
    # Get the filename without path
    filename=$(basename -- "$f")

    # Form the new file path in 'images' directory
    new_file="$dest_dir/$filename"

    # Run pdfcrop command
    echo "Running pdfcrop on $f"
    pdfcrop "$f" "$new_file"
done
