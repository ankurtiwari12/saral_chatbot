# 📊 Evaluation Metrics Guide

## Overview

Your RAG system now includes **automatic evaluation metrics** that run in the background and log to the terminal (not shown in UI).

---

## 📋 Implemented Metrics

### **1. Factuality Proxy**
- **What:** Measures overlap between generated claims and source sentences
- **Methods:** 
  - Exact word overlap
  - Sequence similarity (SequenceMatcher)
- **Output:**
  - Total claims generated
  - Grounded claims (above threshold)
  - Ungrounded claims
  - Factuality score (%)
  - Average overlap score

### **2. Citation Coverage**
- **What:** Percentage of content that has retrieved provenance
- **Output:**
  - Total claims
  - Claims with source provenance
  - Claims without provenance
  - Coverage score (%)
  - Number of source chunks used

### **3. ROUGE Scores** (Optional)
- **What:** Standard NLG evaluation metric
- **Requires:** `rouge-score` package
- **Output:**
  - ROUGE-1 (unigram overlap)
  - ROUGE-2 (bigram overlap)
  - ROUGE-L (longest common subsequence)
  - Each with: Precision, Recall, F-Measure

### **4. BERTScore** (Optional)
- **What:** Semantic similarity using BERT embeddings
- **Requires:** `bert-score` package
- **Output:**
  - Precision
  - Recall
  - F1 score

---

## 🚀 Installation

### **Basic Installation** (Factuality + Coverage only)
```bash
# Already works! No additional packages needed
cd ~/Downloads/iiit/rag
source myenv/bin/activate
streamlit run streamlit_app.py
```

### **Full Installation** (All metrics)
```bash
cd ~/Downloads/iiit/rag
source myenv/bin/activate

# Install ROUGE
pip install rouge-score

# Install BERTScore (larger, downloads models)
pip install bert-score
```

---

## 📊 Terminal Output Example

### **When Evaluation is Active:**

```
================================================================================
EVALUATION METRICS
================================================================================

1. FACTUALITY PROXY
   Total Claims: 18
   Grounded Claims: 15
   Ungrounded Claims: 3
   Factuality Score: 83.33% (threshold: 0.3)
   Average Overlap: 0.612

2. CITATION COVERAGE
   Total Claims: 18
   With Provenance: 16
   Without Provenance: 2
   Coverage Score: 88.89%
   Source Chunks Used: 5

3. ROUGE SCORES
   ROUGE-1:
     Precision: 0.6234
     Recall:    0.7123
     F-Measure: 0.6651
   ROUGE-2:
     Precision: 0.4512
     Recall:    0.5234
     F-Measure: 0.4851
   ROUGE-L:
     Precision: 0.5892
     Recall:    0.6789
     F-Measure: 0.6311

4. BERTSCORE
   Precision: 0.8234
   Recall:    0.8512
   F1:        0.8371

================================================================================
```

---

## 🎯 How It Works

### **Automatic Evaluation Flow:**

```
User submits query
    ↓
System generates answer
    ↓
Before returning to UI:
    ├─ Extract claims from answer
    ├─ Extract sentences from source chunks
    ├─ Compute overlap for each claim
    ├─ Check provenance for each claim
    ├─ (Optional) Compute ROUGE vs reference
    ├─ (Optional) Compute BERTScore vs reference
    └─ Print report to terminal
    ↓
Return answer to UI (user never sees metrics)
```

---

## 📐 Evaluation Components

### **1. Claim Extraction**

**From Generated Answer:**
- Slide content
- Speaker notes  
- Bullet points
- Numbered lists
- Sentences (fallback)

**Example:**
```
Input: "Slide 1: Introduction\n- Point A\n- Point B"
Claims: ["Introduction", "Point A", "Point B"]
```

### **2. Source Sentence Extraction**

**From Retrieved Chunks:**
- Split by sentence boundaries
- Preserve context

**Example:**
```
Chunk: "The model uses attention. This improves performance."
Sentences: ["The model uses attention.", "This improves performance."]
```

### **3. Overlap Computation**

**Method 1: Exact Word Overlap**
```python
Claim: "the model uses attention"
Source: "The transformer model uses multi-head attention"

Common words: {the, model, uses, attention} = 4/4 = 100% overlap
```

**Method 2: Sequence Similarity**
```python
SequenceMatcher("the model uses attention", "model uses attention mechanism")
Similarity: 0.85
```

**Final Score:** `max(exact_overlap, sequence_similarity)`

### **4. Threshold**

**Default:** 0.3 (30% overlap)

- **Above threshold:** Claim is "grounded"
- **Below threshold:** Claim is "ungrounded"

**Adjustable** in code:
```python
factuality = compute_factuality_score(text, chunks, threshold=0.4)
```

---

## 🎚️ Interpreting Scores

### **Factuality Score**

| Score | Interpretation |
|-------|----------------|
| 90-100% | Excellent - almost all claims grounded |
| 70-89% | Good - most claims grounded |
| 50-69% | Fair - some hallucinations |
| <50% | Poor - many ungrounded claims |

### **Citation Coverage**

| Score | Interpretation |
|-------|----------------|
| 90-100% | Excellent - all content has provenance |
| 70-89% | Good - most content traceable |
| 50-69% | Fair - some content unsourced |
| <50% | Poor - many claims lack sources |

### **ROUGE Scores**

- **F-Measure** is the most important metric
- Higher is better (0-1 scale)
- ROUGE-1: ~0.3-0.5 is typical for abstractive summaries
- ROUGE-L: Captures fluency better than ROUGE-1/2

### **BERTScore**

- **F1 > 0.85:** Excellent semantic similarity
- **F1 0.75-0.85:** Good similarity
- **F1 0.65-0.75:** Fair similarity
- **F1 < 0.65:** Poor similarity

---

## 🔍 Use Cases

### **1. Development & Debugging**
- Monitor factuality during development
- Identify when model hallucinates
- Track improvement over iterations

### **2. Testing on 3 Papers**
```bash
# Create reference scripts for 3 test papers
# Store in rag/test_references/

# Paper 1
paper1_reference.txt

# Paper 2
paper2_reference.txt

# Paper 3
paper3_reference.txt
```

Then modify `query_rag()` to load reference if available:
```python
# Check if reference exists for this session
reference_file = f"test_references/{session_id}_reference.txt"
if os.path.exists(reference_file):
    with open(reference_file, 'r') as f:
        reference_text = f.read()
    evaluation = evaluate_response(response_txt, retrieved_chunks, reference_text)
else:
    evaluation = evaluate_response(response_txt, retrieved_chunks, reference_text=None)
```

### **3. Quality Monitoring**
- Run periodic tests on known queries
- Track metrics over time
- Detect regressions

---

## ⚙️ Configuration

### **Enable/Disable Evaluation**

**Automatic:**
- Enabled if `evaluation_metrics.py` is found
- Disabled if import fails

**Manual:**
```python
# In playground.py, set:
EVALUATION_AVAILABLE = False  # Force disable
```

### **Adjust Thresholds**

**In `evaluation_metrics.py`:**
```python
def compute_factuality_score(
    generated_text: str,
    retrieved_chunks: List[Dict],
    threshold: float = 0.3  # Change this!
) -> Dict:
```

### **Show Detailed Scores**

**In `playground.py`:**
```python
print_evaluation_report(evaluation, show_detailed=True)  # Shows claim-by-claim
```

---

## 📦 Dependencies

### **Core (Included)**
- `difflib` ✅ (Python standard library)
- `re` ✅ (Python standard library)
- `numpy` ✅ (already installed)

### **Optional**
- `rouge-score` 🟡 (for ROUGE metrics)
- `bert-score` 🟡 (for BERTScore)

### **Installation Commands**
```bash
# ROUGE only (lightweight)
pip install rouge-score

# BERTScore (downloads ~500MB models on first run)
pip install bert-score

# Both
pip install rouge-score bert-score
```

---

## 🧪 Testing

### **Test 1: Check if Evaluation is Working**

```bash
cd ~/Downloads/iiit/rag
source myenv/bin/activate
streamlit run streamlit_app.py
```

**In terminal, look for:**
```
[INFO] Evaluation metrics module loaded successfully
```

**OR**
```
[INFO] Evaluation metrics module not found. Evaluation disabled.
```

### **Test 2: Submit a Query**

**In UI:**
```
Make 5 slides on attention mechanisms
```

**In terminal after response, you should see:**
```
================================================================================
EVALUATION METRICS
================================================================================
...
```

### **Test 3: Check Factuality**

**Look for:**
```
1. FACTUALITY PROXY
   Factuality Score: XX.XX%
```

- **High score (>80%):** Good! Most claims grounded in sources
- **Low score (<50%):** Model might be hallucinating

---

## 🎯 Example Scenarios

### **Scenario 1: High Factuality, High Coverage**
```
Factuality Score: 92.31%
Coverage Score: 95.45%
```
✅ **Interpretation:** Excellent! Answer is well-grounded in sources.

### **Scenario 2: High Factuality, Low Coverage**
```
Factuality Score: 85.71%
Coverage Score: 60.00%
```
⚠️ **Interpretation:** Claims are accurate but some content lacks explicit source provenance. Consider retrieving more chunks.

### **Scenario 3: Low Factuality, High Coverage**
```
Factuality Score: 45.00%
Coverage Score: 88.89%
```
❌ **Interpretation:** Retrieved relevant chunks but generated claims don't match. Model might be hallucinating or paraphrasing too freely.

### **Scenario 4: Low Factuality, Low Coverage**
```
Factuality Score: 40.00%
Coverage Score: 50.00%
```
❌❌ **Interpretation:** Poor retrieval AND hallucination. Check:
- Vector DB has relevant documents
- Query is well-formed
- Model is functioning correctly

---

## 📈 Tracking Over Time

### **Create a Log File**

Add to `playground.py`:
```python
import datetime

# After evaluation
if EVALUATION_AVAILABLE:
    evaluation = evaluate_response(response_txt, retrieved_chunks, reference_text=None)
    print_evaluation_report(evaluation, show_detailed=False)
    
    # Log to file
    with open("evaluation_log.jsonl", "a") as f:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": session_id,
            "query": query_txt,
            "factuality": evaluation['factuality']['factuality_score'],
            "coverage": evaluation['citation_coverage']['citation_coverage'],
            "total_claims": evaluation['factuality']['total_claims']
        }
        f.write(json.dumps(log_entry) + "\n")
```

### **Analyze Logs**

```python
import pandas as pd

# Read logs
logs = []
with open("evaluation_log.jsonl", "r") as f:
    for line in f:
        logs.append(json.loads(line))

df = pd.DataFrame(logs)

# Compute statistics
print(f"Average Factuality: {df['factuality'].mean():.2%}")
print(f"Average Coverage: {df['coverage'].mean():.2%}")
print(f"Queries below 70% factuality: {(df['factuality'] < 0.7).sum()}")
```

---

## 🚀 Next Steps

1. **Run a query** and check terminal for evaluation output
2. **Install ROUGE** for quality metrics: `pip install rouge-score`
3. **Create reference scripts** for 3 test papers
4. **Monitor factuality** scores during development
5. **Adjust thresholds** if needed for your use case

---

## ✅ Summary

✅ **Factuality Proxy** - Automatic, always on  
✅ **Citation Coverage** - Automatic, always on  
🟡 **ROUGE Scores** - Optional, requires `rouge-score`  
🟡 **BERTScore** - Optional, requires `bert-score`

**All metrics log to terminal only (not shown in UI)**

**Ready to use!** Just run your Streamlit app and check the terminal output! 📊✨

