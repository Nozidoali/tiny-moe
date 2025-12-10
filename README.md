# TinyMoE

## Environment Variables Setup

This project uses environment variables to configure paths. Set the following variables in your shell environment:

### Required Environment Variables

- **`TINYMOE_DIR`**: Base directory for the TinyMoE project
  - Default: `/Users/hanyu/Documents/tinymoe`
  - Used by: All scripts and Python modules for locating project directories

- **`LLAMA_CPP_DIR`**: Path to the llama.cpp repository
  - Default: `/Users/hanyu/Documents/cs259/external/llama.cpp` (local) or `/data/local/tmp/llama.cpp` (device)
  - Used by: Shell scripts for model conversion and inference

### Setting Environment Variables

You can set environment variables in two ways:

#### Option 1: Using a `.env` file (Recommended)

Create a `.env` file in the project root:

```bash
TINYMOE_DIR=/Users/hanyu/Documents/tinymoe
LLAMA_CPP_DIR=/Users/hanyu/Documents/cs259/external/llama.cpp
```

The Python scripts will automatically load these from the `.env` file using `python-dotenv`.

#### Option 2: Export in Shell

```bash
export TINYMOE_DIR="/Users/hanyu/Documents/tinymoe"
export LLAMA_CPP_DIR="/Users/hanyu/Documents/cs259/external/llama.cpp"
```
