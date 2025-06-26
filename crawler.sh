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
TREE_LEVEL=4 # Reduced level slightly for brevity

# Maximum file size to include (in Kilobytes). Files larger than this will be skipped.
MAX_FILE_SIZE_KB=256

# Directories to exclude from the scan.
# Used for both 'find' and 'tree' commands.
EXCLUDE_DIRS=(
    # --- NEW: Top-level dependency/grammar directories to completely ignore ---
    "grammars"            # Prevents crawling into ANY 'grammars' folder
    "vendor"              # Excludes all vendored dependencies

    # --- Project-Specific Exclusions ---
    "cloned_repositories" # Excludes all ingested repository artifacts.

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
    "build"
    "target"
    "instance"
    "uploads"
    "unprocessed"
    "*.egg-info"
    ".github"             # Exclude top-level and nested CI/CD
    "docs"                # Exclude documentation folders

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
    "parser.c"            # Excludes the huge generated parser files
    "scanner.c"
    "scanner.cc"
    "go.sum"
    "go.mod"
    "Cargo.lock"
    # --- AI/ML Specific File Exclusions ---
    "chroma.sqlite3"
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
    "wasm"
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

    local TARGET_DIR=${1:-.}
    TARGET_DIR=$(realpath "$TARGET_DIR")

    # --- Setup ---
    trap cleanup EXIT TERM INT
    local TEMP_FILE
    TEMP_FILE=$(mktemp)
    local OUTPUT_FILE="llm_project_context.txt"
    if [[ "$(basename "$TARGET_DIR")" != "." && "$TARGET_DIR" != "$PWD" ]]; then
        local PREFIX
        PREFIX=$(basename "$TARGET_DIR")
        OUTPUT_FILE="${PREFIX}_context.txt"
    fi

    local SCRIPT_NAME
    SCRIPT_NAME=$(basename "$0")

    # --- Header ---
    echo -e "${C_BOLD}${C_BLUE}--- LLM Context Generator ---${C_RESET}"
    echo -e "Scanning directory: ${C_YELLOW}$TARGET_DIR${C_RESET}"
    echo -e "Output file:        ${C_YELLOW}$OUTPUT_FILE${C_RESET}"
    echo ""

    # --- Project Tree Generation ---
    > "$OUTPUT_FILE"

    if command -v tree &> /dev/null; then
        echo -e "${C_BLUE}Generating project tree structure (up to level $TREE_LEVEL)...${C_RESET}"
        local TREE_IGNORE_PATTERN
        TREE_IGNORE_PATTERN=$(IFS='|'; echo "${EXCLUDE_DIRS[*]}")

        echo "--- PROJECT STRUCTURE ---" >> "$OUTPUT_FILE"
        tree -L "$TREE_LEVEL" -a -I "$TREE_IGNORE_PATTERN" "$TARGET_DIR" >> "$OUTPUT_FILE" || echo "Tree command failed, continuing..."
        echo -e "\n--- END PROJECT STRUCTURE ---\n" >> "$OUTPUT_FILE"
    else
        echo -e "${C_YELLOW}Warning: 'tree' command not found. Skipping project structure generation.${C_RESET}"
    fi

    # --- File Discovery ---
    echo -e "${C_BLUE}Finding files to process...${C_RESET}"
    local find_args=()
    # Build exclusion arguments for 'find' command
    # Exclude any path containing a directory name from EXCLUDE_DIRS
    for dir in "${EXCLUDE_DIRS[@]}"; do
        find_args+=('!' '-path' "*/${dir}/*")
    done
    # Exclude specific filenames
    for file in "${EXCLUDE_FILES[@]}"; do
        find_args+=('!' '-name' "$file")
    done
    # Exclude specific extensions
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
    if [[ $total_files -gt 1500 ]]; then
        echo -e "${C_YELLOW}WARNING: Found $total_files files. This might produce a very large context file.${C_RESET}"
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
        [[ -z "$file" ]] && continue
        ((processed_count++))

        local size
        size=$(wc -c < "$file")
        if [[ $size -gt $max_size_bytes ]]; then
            echo -e "${C_GRAY}[$processed_count/$total_files] Skipping large file: $(basename "$file") ($((size/1024)) KB > $MAX_FILE_SIZE_KB KB)${C_RESET}"
            ((skipped_count++))
            continue
        fi

        ((file_count++))
        echo -e "${C_GREEN}[$processed_count/$total_files]${C_RESET} Adding ${C_GRAY}$file${C_RESET}"

        local relative_path
        relative_path=${file#"$TARGET_DIR/"}

        echo "--- FILE_START: $relative_path ---" >> "$OUTPUT_FILE"
        cat "$file" >> "$OUTPUT_FILE"
        echo -e "\n--- FILE_END: $relative_path ---\n" >> "$OUTPUT_FILE"

    done < "$TEMP_FILE"

    # --- Summary ---
    echo ""
    echo -e "${C_BOLD}${C_GREEN}--- Complete ---${C_RESET}"

    if [[ -f "$OUTPUT_FILE" ]] && [[ -s "$OUTPUT_FILE" ]]; then
        local size_info
        size_info=$(du -h "$OUTPUT_FILE" | cut -f1)
        local line_count
        line_count=$(wc -l < "$OUTPUT_FILE" | xargs)

        echo "Processed: $file_count files"
        echo "Skipped:   $skipped_count files (due to size limit)"
        echo "Output size: $size_info"
        echo "Total lines: $line_count"
        echo ""
        echo -e "${C_BOLD}File types processed:${C_RESET}"
        # FIXED: Robustly get file extensions or a placeholder for files without one.
        grep "^--- FILE_START:" "$OUTPUT_FILE" | \
            sed -e 's/^--- FILE_START: //' -e 's/ ---$//' | \
            awk -F. '{if (NF>1) print $NF; else print "(no_ext)"}' | \
            sort | uniq -c | sort -nr
        echo ""
        echo -e "${C_GREEN}✓ Context file ready: $OUTPUT_FILE${C_RESET}"
    else
        echo -e "${C_RED}❌ No files were processed or the output file is empty.${C_RESET}"
    fi
}

# --- Script Execution ---
main "$@"
