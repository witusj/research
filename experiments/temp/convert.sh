#!/bin/bash

# Loop through all .qmd files in the current directory
for file in *.qmd; do
    
    # Convert qmd to ipynb
    quarto convert "$file"
done

echo "Conversion completed."
