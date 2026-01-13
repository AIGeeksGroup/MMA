# MMA-Bench Toolkit

This directory contains the source code for **generating** and **evaluating** the MMA-Bench dataset. It is designed to be modular: you can generate new cases, run inference with different models, and grade them automatically.

## 📂 Code Structure

- **`mma_bench/`**:
  - `generator.py`: The main pipeline to generate "Trust-Trap" scenarios using Qwen-Max.
  - `prompts.py`: Contains the logic for "Reliability Inversion", "Ambiguity", and "Noise".
  - `run_inference.py`: Supports both Text Mode (Oracle Captions) and Vision Mode (Raw Images).
  - `run_grading.py`: Implements the "CoRe Score" calculation using LLM-as-a-Judge.
- **`data/`**: Stores generated JSON cases and images.

## 🛠️ Customizing Generation

You can modify `mma_bench/config.py` to change the benchmark distribution:

```python
class Config:
    # Adjust total number of cases
    NUM_CASES = 30

    # Define difficulty distribution or categories
    CATEGORIES = ["Career", "Health", "Finance", ...]
```

## 📊 Data Format

Each generated case (`.json`) follows this structure:

```json
{
  "case_id": "core_case_001",
  "metadata": {
    "trap_type": "Type_B_Inversion",
    "noise_level": "High"
  },
  "dialogue_history": [...],
  "evaluation": [
    {
      "id": 1,
      "question": "...",
      "judgment_logic": "Score 1 if confidence < 0.4..."
    }
  ]
}
```
