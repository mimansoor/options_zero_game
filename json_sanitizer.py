import json
import os
import re

# --- Configuration ---
# The target file that will be read from and written to.
TARGET_FILE = "zoo/options_zero_game/visualizer-ui/build/replay_log.json"

def sanitize_json_file_inplace(file_path: str):
    """
    Reads a JSON file that may contain illegal 'NaN', 'Infinity', or '-Infinity' values,
    replaces them with valid JSON 'null' values, and then overwrites the original
    file with the cleaned, valid JSON content.
    """
    print(f"--- Starting JSON In-Place Sanitization ---")
    
    # 1. Check if the target file exists.
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Target file not found at '{file_path}'")
        return

    print(f"Attempting to read and clean: '{file_path}'")
    print(f"⚠️  WARNING: This will permanently overwrite the original file.")

    try:
        # 2. Read the entire file content into memory as a raw text string.
        with open(file_path, 'r') as f:
            raw_content = f.read()

        # 3. Use regular expressions to replace all non-standard values with 'null'.
        cleaned_content = re.sub(r'\bNaN\b', 'null', raw_content)
        cleaned_content = re.sub(r'\bInfinity\b', 'null', cleaned_content)
        cleaned_content = re.sub(r'-Infinity\b', 'null', cleaned_content)

        # 4. Validate that the cleaned content is now valid JSON by parsing it.
        data = json.loads(cleaned_content)

        # 5. Write the validated, cleaned data back to the ORIGINAL file.
        #    Opening with 'w' mode automatically truncates the file before writing.
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ SUCCESS: File has been sanitized and overwritten successfully.")
        print(f"Clean replay log is now ready at: '{file_path}'")

    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Failed to parse JSON even after cleaning. The file might have other structural errors.")
        print(f"   Details: {e}")
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred.")
        print(f"   Details: {e}")

if __name__ == "__main__":
    # Run the in-place sanitizer on the default target file.
    sanitize_json_file_inplace(TARGET_FILE)
