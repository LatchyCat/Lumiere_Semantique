#!/bin/bash

# --- Script Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error.
set -u
# The return value of a pipeline is the status of the last command to exit with a non-zero status.
set -o pipefail

# --- User-Defined Configuration ---

# Maximum depth for the 'tree' command.
TREE_LEVEL=5

# Maximum file size to include (in Kilobytes). Files larger than this will be skipped.
MAX_FILE_SIZE_KB=256 # Lowered to avoid unexpectedly large config/data files

# Directories to exclude from the scan.
# Used for both 'find' and 'tree' commands.
EXCLUDE_DIRS=(
    # --- THE NEW ADDITION ---
    "cloned_repositories" # Excludes all ingested repository artifacts.
    "vendor"              # <--- ADDED: Excludes vendor directories, such as from tree-sitter.

    # --- Standard Exclusions ---
    "node_modules"
    ".venv"
    "venv"
    "env"
    "venv_broken"
    ".git"
    "__pycache__"
    ".pytest_cache"
    "dist"
    "build"               # This already handles your `backend/build` folder
    "target"
    "instance"
    "uploads"
    "unprocessed"
    "*.egg-info" # Added for Python package metadata

    # --- AI/ML Specific Directory Exclusions ---
    "data"
    "embeddings"
    "models"
    "faiss_index"
    "chroma_db"
    ".chroma"
)

# Specific file names to exclude.
EXCLUDE_FILES=(
    "package-lock.json"
    "yarn.lock"
    "pnpm-lock.yaml"
    ".DS_Store"
    "*.min.js"
    "*.min.css"
    # --- AI/ML Specific File Exclusions ---
    "chroma.sqlite3" # Exclude the main ChromaDB file if it's in the root.
)

# File extensions to exclude. (Note: leading dot is not needed)
EXCLUDE_EXTENSIONS=(
    "pyc"
    "db"
    "sqlite3"
    "lockb"
    "tsbuildinfo"
    "svg"
    "jpg"
    "jpeg"
    "png"
    "gif"
    "ico"
    "bin"
    "log"
    "zip"
    "tar"
    "gz"
    "pdf"
    # --- AI/ML Specific Extension Exclusions ---
    "jsonl"
    "pkl"
    "parquet"
    "h5"
    "safetensors"
    "index"
    "faiss"
)

# --- End of Configuration ---

# --- Color Definitions ---
C_RESET='\033[0m'
C_BOLD='\033[1m'
C_BLUE='\033[0;34m'
C_GREEN='\033[0;32m'
C_YELLOW='\033[0;33m'
C_RED='\033[0;31m'
C_GRAY='\033[0;90m'

# --- Functions ---

# Function to display usage information.
show_usage() {
cat << EOF
${C_BOLD}LLM Project Context Generator${C_RESET}

This script scans a project directory and compiles all relevant code and configuration
files into a single text file, which can be easily shared with an AI model.

${C_BOLD}Usage:${C_RESET}
  $0 [TARGET_DIRECTORY]
  $0 -h | --help

${C_BOLD}Arguments:${C_RESET}
  ${C_GREEN}TARGET_DIRECTORY${C_RESET}  Optional. The directory to scan. Defaults to the current directory ('.').

${C_BOLD}Options:${C_RESET}
  ${C_GREEN}-h, --help${C_RESET}          Show this help message and exit.

${C_BOLD}Features:${C_RESET}
  - Generates a project file tree structure.
  - Excludes common dependency, build, and temporary directories/files.
  - Skips binary files, large files, and lock files.
  - Creates a single, clean output file named after the project directory.
  - All exclusions can be easily configured in the script's variables.
EOF
}

# Cleanup function to be called on script exit.
cleanup() {
  if [[ -n "${TEMP_FILE:-}" && -f "$TEMP_FILE" ]]; then
    rm -f "$TEMP_FILE"
  fi
}

# --- Main Logic ---
main() {
    # --- Argument Parsing ---
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        show_usage
        exit 0
    fi

    # An optional first argument can specify the directory to scan.
    # Defaults to the current directory '.'.
    local TARGET_DIR=${1:-.}
    # Get the absolute path for clarity in output
    TARGET_DIR=$(realpath "$TARGET_DIR")

    # --- Setup ---
    # Register the cleanup function to be called on EXIT, TERM, or INT signals.
    trap cleanup EXIT TERM INT

    # Create a secure temporary file to store the list of files.
    local TEMP_FILE
    TEMP_FILE=$(mktemp)

    # Determine the output file name based on the target directory.
    local OUTPUT_FILE
    if [[ "$(basename "$TARGET_DIR")" == "." || "$TARGET_DIR" == "$PWD" ]]; then
        OUTPUT_FILE="llm_project_context.txt"
    else
        local PREFIX
        PREFIX=$(basename "$TARGET_DIR")
        OUTPUT_FILE="${PREFIX}_context.txt"
    fi

    # Get the name of this script to exclude it from the output.
    local SCRIPT_NAME
    SCRIPT_NAME=$(basename "$0")

    # --- Header ---
    echo -e "${C_BOLD}${C_BLUE}--- LLM Context Generator ---${C_RESET}"
    echo -e "Scanning directory: ${C_YELLOW}$TARGET_DIR${C_RESET}"
    echo -e "Output file:        ${C_YELLOW}$OUTPUT_FILE${C_RESET}"
    echo ""

    # --- Project Tree Generation ---
    # Clear output file and add the project tree structure first.
    # The '>' operator creates or truncates the file.
    > "$OUTPUT_FILE"

    if command -v tree &> /dev/null; then
        echo -e "${C_BLUE}Generating project tree structure (up to level $TREE_LEVEL)...${C_RESET}"
        # Dynamically create the ignore pattern for 'tree' from the EXCLUDE_DIRS array.
        local TREE_IGNORE_PATTERN
        TREE_IGNORE_PATTERN=$(IFS='|'; echo "${EXCLUDE_DIRS[*]}")

        echo "--- PROJECT STRUCTURE ---" >> "$OUTPUT_FILE"
        tree -L "$TREE_LEVEL" -a -I "$TREE_IGNORE_PATTERN" "$TARGET_DIR" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        echo "--- END PROJECT STRUCTURE ---" >> "$OUTPUT_FILE"
        echo -e "\n" >> "$OUTPUT_FILE"
    else
        echo -e "${C_YELLOW}Warning: 'tree' command not found. Skipping project structure generation.${C_RESET}"
        echo -e "${C_GRAY}         (To install: 'sudo apt-get install tree' or 'brew install tree')${C_RESET}"
    fi

    # --- File Discovery ---
    echo -e "${C_BLUE}Finding files to process...${C_RESET}"
    local find_args=()
    # Build exclusion arguments for 'find' command
    for dir in "${EXCLUDE_DIRS[@]}"; do
        find_args+=('!' '-path' "*/${dir}/*")
    done
    for file in "${EXCLUDE_FILES[@]}"; do
        find_args+=('!' '-name' "$file")
    done
    for ext in "${EXCLUDE_EXTENSIONS[@]}"; do
        find_args+=('!' '-name' "*.$ext")
    done

    # Add this script and its output to the exclusion list
    find_args+=('!' '-name' "$SCRIPT_NAME")
    find_args+=('!' '-name' "$OUTPUT_FILE")

    # Execute find command
    find "$TARGET_DIR" -type f "${find_args[@]}" > "$TEMP_FILE"

    local total_files
    total_files=$(wc -l < "$TEMP_FILE")

    echo "Found $total_files files to process."

    # --- Sanity Checks ---
    if [[ $total_files -eq 0 ]]; then
        echo -e "${C_YELLOW}No files found matching the criteria. Exiting.${C_RESET}"
        exit 0
    fi
    if [[ $total_files -gt 1000 ]]; then
        echo -e "${C_YELLOW}WARNING: Found $total_files files. This might produce a very large context file.${C_RESET}"
        echo "First 10 files found:"
        head -10 "$TEMP_FILE"
        echo ""
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${C_RED}Aborted by user.${C_RESET}"
            exit 1
        fi
    fi
    echo ""

    # --- File Processing ---
    local file_count=0
    local skipped_count=0
    local processed_count=0
    local max_size_bytes=$((MAX_FILE_SIZE_KB * 1024))

    while IFS= read -r file; do
        # Skip empty lines that might be in the temp file
        [[ -z "$file" ]] && continue

        ((processed_count++))

        # Check file size. wc is faster than stat on some systems for a simple byte count.
        local size
        size=$(wc -c < "$file")
        if [[ $size -gt $max_size_bytes ]]; then
            echo -e "${C_GRAY}[$processed_count/$total_files] Skipping large file: $(basename "$file") ($((size/1024)) KB > $MAX_FILE_SIZE_KB KB)${C_RESET}"
            ((skipped_count++))
            continue
        fi

        ((file_count++))
        echo -e "${C_GREEN}[$processed_count/$total_files]${C_RESET} Adding ${C_GRAY}$file${C_RESET}"

        # Get the relative path for a cleaner output header
        local relative_path
        relative_path=${file#"$TARGET_DIR/"}

        # Append file content to the output file
        echo "--- FILE_START: $relative_path ---" >> "$OUTPUT_FILE"
        cat "$file" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE" # Add a newline for better separation
        echo "--- FILE_END: $relative_path ---" >> "$OUTPUT_FILE"
        echo -e "\n" >> "$OUTPUT_FILE"

    done < "$TEMP_FILE"

    # --- Summary ---
    echo ""
    echo -e "${C_BOLD}${C_GREEN}--- Complete ---${C_RESET}"

    if [[ -f "$OUTPUT_FILE" ]] && [[ -s "$OUTPUT_FILE" ]]; then
        local size_info
        size_info=$(du -h "$OUTPUT_FILE" | cut -f1)
        local line_count
        line_count=$(wc -l < "$OUTPUT_FILE")

        echo "Processed: $file_count files"
        echo "Skipped:   $skipped_count files (due to size limit)"
        echo "Output size: $size_info"
        echo "Total lines: $line_count"
        echo ""
        echo -e "${C_BOLD}File types processed:${C_RESET}"
        # Extract extensions, sort, count, and display the most common ones first
        grep "^--- FILE_START:" "$OUTPUT_FILE" | sed -E 's/.*\.([^.]+)$/\1/' | sort | uniq -c | sort -nr
        echo ""
        echo -e "${C_GREEN}✓ Context file ready: $OUTPUT_FILE${C_RESET}"
    else
        echo -e "${C_RED}❌ No files were processed or the output file is empty.${C_RESET}"
    fi
}

# --- Script Execution ---
# Pass all command-line arguments to the main function.
main "$@"
