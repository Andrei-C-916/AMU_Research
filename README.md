# Vet Patient Data Extraction Pipeline

This project extracts antimicrobial dosage information from veterinary patient records using an LLM-based extraction pipeline. The extracted values are compared against a gold-standard dataset to evaluate extraction accuracy and uncertainty.

The pipeline consists of three main stages:

1. Creating the gold-standard dataset
2. Running LLM-based extraction on patient JSON files
3. Evaluating extraction performance with metrics

---

# Setup

Install dependencies:

```bash
pip install pandas numpy scikit-learn requests python-dotenv
```

Create a `.env` file in project root:

```env
API_URL=https://ai-dashboard.vet.cornell.edu/api/chat/completions
API_KEY=your_api_key_here
```

The patient data inside `./data` is gitignored for safety/privacy reasons and must be added manually.

---

# Data

The `./data` directory should contain:

```text
single_and_multi_cols_2021_labels.csv
patients_with_ABX_2021/
```

The `patients_with_ABX_2021/` directory contains individual patient JSON files.

The CSV file contains structured antimicrobial administration labels used to generate the gold-standard dataset.

---

# Prompts

Prompts are stored in:

```text
./prompts/prompts.json
```

This file contains:

- The extraction prompt
- The evaluation/selection prompt

The extraction prompt is used to extract antimicrobial usage from patient records.

The evaluation prompt is used to compare multiple generated outputs and select the most internally consistent one.

---

# Step 1: Create the Gold Standard

Run:

```bash
python create_mgadministered_gold.py
```

This script reads:

```text
./data/single_and_multi_cols_2021_labels.csv
```

and generates:

```text
./data/mgadministered_gold_data.json
```

The script computes:

```text
total mg administered = BWkg × Dosemgkg × Durationd
```

for each antimicrobial listed in the CSV.

Each patient can contain up to 7 antimicrobial entries.

The generated JSON serves as the gold-standard dataset used for evaluation.

---

# Step 2: Run Extraction

Run:

```bash
python extract_patient_data.py <num_patients> [num_attempts]
```

Example:

```bash
python extract_patient_data.py 25 10
```

This processes the first 25 patients using 10 LLM extraction attempts per patient.

Results are saved to:

```text
./data/extracted_data.json
```

---

# Optional Flags

## Trim Patient Data

```bash
--trim-data
```

Example:

```bash
python extract_patient_data.py 25 10 --trim-data
```

This trims patient JSON files down to essential fields before sending them to the model.

The retained fields are:

- patient_code
- patient_name
- species
- breed
- sex
- prescriptions

This can help reduce context length and improve stability for extremely large patient records.

---

## Maximum Patient JSON Size

```bash
--max-patient-json-chars
```

Example:

```bash
python extract_patient_data.py 25 10 --max-patient-json-chars 80000
```

If a patient JSON exceeds the specified character limit, it is automatically truncated before being inserted into the prompt.

This prevents model context-length failures.

Default:

```text
120000
```

---

# Extraction Pipeline

For each patient:

1. Patient JSON is loaded
2. Optional trimming is applied
3. JSON may be truncated if too large
4. The model is queried multiple times
5. Each output is parsed into JSON
6. A second evaluation prompt selects the best output
7. Uncertainty metrics are computed across attempts

The final selected output contains:

- antimicrobial names
- extracted total mg administered
- uncertainty coefficient of variation (CV)

---

# Model API

Model calls are handled by:

```text
./code/open_webui_client.py
```

The API uses:

```python
requests.post(...)
```

with:

- Anthropic Claude Sonnet 4.5
- OpenWebUI-compatible chat completion format

The API key is loaded from the `.env` file using `python-dotenv`.

---

# Step 3: Run Metrics

Run:

```bash
python metrics.py
```

This compares:

```text
./data/extracted_data.json
```

against:

```text
./data/mgadministered_gold_data.json
```

and computes extraction performance metrics.

---

# Metrics Computed

The evaluation includes:

- Mean Absolute Error (MAE)
- RMSE
- MAPE
- R² Score
- Median Absolute Error
- Maximum Error

It also computes:

- Error distributions
- Per-patient uncertainty
- Per-drug uncertainty
- Uncertainty distributions

---

# Drug Matching

Drug names are matched using fuzzy word-overlap matching.

Examples:

```text
"Amoxicillin-Clavulanate"
```

may match:

```text
"Amoxicillin"
```

Drug names are normalized by:

- lowercasing
- removing punctuation
- splitting into words

Two drugs are considered a match if they share at least one word.

---

# Output Files

## Gold Standard

```text
./data/mgadministered_gold_data.json
```

Contains structured ground-truth antimicrobial administration values.

---

## Extracted Data

```text
./data/extracted_data.json
```

Contains model-generated antimicrobial extraction outputs and uncertainty estimates.

---

# Notes

- The `./data` directory is intentionally gitignored.
- Patient data is not included in the repository.
- The extraction pipeline is designed to tolerate malformed model outputs and partial JSON failures.
- If model outputs are invalid JSON, regex-based fallback parsing is attempted.
- If evaluation parsing fails, the first successful attempt is used as fallback.
