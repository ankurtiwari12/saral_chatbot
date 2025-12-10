# Advanced RAG System with Evaluation & Math-Aware Retrieval

A comprehensive Retrieval-Augmented Generation (RAG) system with advanced features including math-aware retrieval, interactive feedback, automatic evaluation metrics, and intelligent refinement capabilities.

---

## 🚀 Features

### **Core RAG Pipeline**
- ✅ **Document Loading**: PDF upload and indexing via Streamlit UI
- ✅ **Smart Text Splitting**: Recursive character-based chunking with overlap
- ✅ **Vector Storage**: ChromaDB with Ollama embeddings
- ✅ **Semantic Retrieval**: Context-aware document retrieval
- ✅ **LLM Generation**: Local Ollama LLM for response generation

### **Advanced Retrieval**
- 🔬 **Math-Aware Retrieval**: Enhanced retrieval for mathematical content
  - SymPy-based equation canonicalization
  - Multi-component scoring (exact match, symbolic, numeric, structural)
  - Equation indexing with SHA256 identifiers
  - Jaccard similarity for math symbols
  - Math density boosting

### **Intelligent Refinement**
- 🎯 **Slide-Specific Refinement**: Modify individual slides without regenerating entire responses
- 📝 **Intent Extraction**: Captures nuanced user requests (longer, creative, simpler, etc.)
- 🔄 **Format Preservation**: Maintains structure (slides, equations, speaker notes)
- 📏 **Script Length Control**: Generate 30s, 90s, or 5-minute presentations
- 🎨 **Style Variations**: Technical, plain-English, or press-release styles

### **Interactive Feedback**
- 👍 **Bullet-Level Feedback**: Accept/reject individual statements
- 📚 **Source Provenance**: See which document each statement comes from
- 🔍 **Source Highlighting**: View relevant excerpts from source documents
- 🔄 **Feedback Loop**: Generate refined responses based on user feedback
- 🧮 **Math Badges**: Visual indicators for math-heavy content

### **Automatic Evaluation** (Terminal)
- 📊 **Factuality Proxy**: Measures overlap between generated claims and source content
- 📋 **Citation Coverage**: Tracks what percentage of content has source provenance
- 📈 **ROUGE Scores**: N-gram overlap with human-authored references (optional)
- 🎯 **BERTScore**: Semantic similarity using BERT embeddings (optional)

### **Change Tracking**
- 🔴 **GitHub-Style Diffs**: Color-coded changes (red=removed, green=added)
- 📊 **Granular Comparison**: Line-by-line delta display
- 📝 **Change Explanations**: Shows why content was modified

---

## 📦 Installation

### **Prerequisites**
- Python 3.12+
- Ollama installed and running
- CUDA-compatible GPU (optional, for better performance)

### **Step 1: Clone & Setup**

```bash
# Navigate to project directory
cd /home/ankur/Downloads/iiit/rag

# Create and activate virtual environment
python3 -m venv myenv
source myenv/bin/activate
```

### **Step 2: Install Core Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 3: Install Optional Evaluation Dependencies**

For full evaluation metrics (ROUGE/BERTScore):

```bash
pip install rouge-score sentence-transformers
```

**Note**: BERTScore requires PyTorch and Transformers, which may have compatibility issues. If you encounter errors, the system will gracefully skip BERTScore and still provide Factuality and Citation Coverage metrics.

### **Step 4: Setup Ollama Models**

```bash
# Install required Ollama models
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest
```

---

## 🎮 Usage

### **Starting the Application**

```bash
cd /home/ankur/Downloads/iiit/rag
source myenv/bin/activate
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

### **Basic Workflow**

1. **Upload PDFs**
   - Use the sidebar file uploader
   - Click "🚀 Load & Index Uploaded Documents"
   - Wait for indexing to complete

2. **Submit Queries**
   - Type your question in the chat
   - Examples:
     - `Make 5 slides on attention mechanisms with equations`
     - `Create a 90s technical presentation on transformers`
     - `Explain homomorphic encryption in plain English`

3. **Refine Responses**
   - Modify specific slides: `Make slide 2 more creative`
   - Adjust length: `Make it shorter`
   - Change style: `Rewrite in plain English`

4. **Use Feedback** (Optional)
   - Enable "Enable Interactive Feedback UI" in sidebar
   - Review each generated statement
   - Accept ✅ or Reject ❌ statements
   - Click "Generate Refined Answer Based on Feedback"

5. **Check Evaluation** (Terminal)
   - View automatic metrics in the terminal
   - See factuality, coverage, and optional ROUGE/BERTScore

---

## 📊 Evaluation System

### **Automatic Metrics** (Always Available)

#### **1. Factuality Proxy**
- Measures overlap between generated claims and source sentences
- Threshold: 0.3 (configurable)
- **Good Score**: > 70%

#### **2. Citation Coverage**
- Percentage of claims with source provenance
- Threshold: 0.25 (configurable)
- **Good Score**: > 80%

### **Quality Metrics** (Optional - Requires Reference Scripts)

#### **3. ROUGE Scores**
- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest common subsequence
- **Good Score**: > 0.4

#### **4. BERTScore**
- Semantic similarity using BERT embeddings
- Precision, Recall, F1
- **Good Score**: F1 > 0.8

### **Setting Up Reference Scripts**

Create or edit `reference_scripts.json`:

```json
{
  "your-paper.pdf": {
    "title": "Paper Title",
    "reference_script": "Your human-authored reference text here..."
  }
}
```

The system automatically loads reference scripts when matching PDFs are uploaded.

---

## 🔬 Math-Aware Retrieval

### **How It Works**

1. **Equation Extraction**: Identifies LaTeX equations in documents
2. **Canonicalization**: Uses SymPy to normalize equations
3. **Multi-Component Scoring**:
   - **30%** - Exact equation ID matches
   - **25%** - Symbol Jaccard similarity
   - **20%** - Numeric equivalence
   - **15%** - Structural similarity
   - **10%** - Math density bonus

4. **Boosted Ranking**: Math-relevant chunks ranked higher for math queries

### **Example**

Query: `What is the attention formula?`

The system will:
- Extract equations from query
- Match against indexed equations in documents
- Boost chunks containing `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- Return most relevant mathematical content first

---

## 🎨 Script Styles & Lengths

### **Supported Lengths**

- **30s**: ~60-100 words (quick overview)
- **90s**: ~150-250 words (standard presentation)
- **5min**: ~500-800 words (detailed explanation)

### **Supported Styles**

- **Technical**: Formal academic language with equations
- **Plain-English**: Simplified, accessible language
- **Press-Release**: Engaging, impactful narrative

### **Usage**

```
Make a 90s technical presentation on transformers
Create 5 slides in plain English about attention
Write a 30s press-release style summary
```

---

## 📁 Project Structure

```
rag/
├── streamlit_app.py              # Main Streamlit UI
├── playground.py                 # Core RAG logic & functions
├── evaluation_metrics.py         # Evaluation system
├── reference_scripts.json        # Reference scripts for ROUGE/BERTScore
├── conversation_history.json     # Persistent chat history
├── feedback_data.json            # User feedback storage
├── requirements.txt              # Core dependencies
├── requirements_evaluation.txt   # Optional evaluation deps
├── chroma_db/                    # Vector database (auto-created)
├── pdfs/                         # Default PDF storage
└── README.md                     # This file
```

---

## 🛠️ Configuration

### **Key Constants** (in `playground.py`)

```python
# Paths
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rag_tutorial"
DATA_PATH = "pdfs"
REFERENCE_SCRIPTS_PATH = "reference_scripts.json"

# Chunking
chunk_size = 800
chunk_overlap = 80

# Retrieval
k = 5  # Number of chunks to retrieve
math_boost_factor = 0.4  # Math relevance boost

# Evaluation
factuality_threshold = 0.3
citation_threshold = 0.25
```

---

## 🔧 Troubleshooting

### **Issue: "streamlit: command not found"**

**Solution**: Activate virtual environment first
```bash
source myenv/bin/activate
```

### **Issue: "BERTScore not available"**

**Cause**: PyTorch/Transformers version conflict

**Solution**: BERTScore is optional. The system will still provide:
- ✅ Factuality Proxy
- ✅ Citation Coverage
- ✅ ROUGE Scores (if `rouge-score` installed)

If you need BERTScore, try:
```bash
pip uninstall torch transformers -y
pip install torch transformers --upgrade
pip install bert-score
```

### **Issue: Math equations not detected**

**Cause**: SymPy not installed

**Solution**:
```bash
pip install sympy
```

### **Issue: Low ROUGE/BERTScore**

**Possible Causes**:
1. Reference script doesn't match output format (slides vs. paragraphs)
2. Different writing style than reference
3. Query asks for specific subset of information

**Note**: Low scores don't mean poor generation—just different from reference!

---

## 📊 Terminal Output Example

```
[INFO] 📊 Reference script loaded for quality evaluation
[INFO] ✓ Loaded reference script for: paper.pdf
[INFO] Found 45 chunks with mathematical content
[INFO] Successfully parsed 42 chunks with SymPy canonicalization

[MATH-AWARE RETRIEVAL]
Query contains 2 equation(s)
✓ SymPy canonicalization active
Found 8 chunks with exact equation matches
Retrieved 5 chunks with avg math similarity: 0.756

================================================================================
EVALUATION METRICS
================================================================================

1. FACTUALITY PROXY
   Total Claims: 18
   Grounded Claims: 15
   Ungrounded Claims: 3
   Factuality Score: 83.33% ✅
   Average Overlap: 0.645

2. CITATION COVERAGE
   Total Claims: 18
   With Provenance: 16
   Without Provenance: 2
   Coverage Score: 88.89% ✅
   Average Provenance: 0.512

3. ROUGE SCORES
   ROUGE-1 F-Measure: 0.5093 ✅
   ROUGE-2 F-Measure: 0.2634 ✅
   ROUGE-L F-Measure: 0.4501 ✅

4. BERTSCORE
   F1: 0.8344 ✅

================================================================================
```

---

## 🎯 Example Queries

### **Basic Generation**
```
Make 4 slides on neural networks
Explain transformers with equations
Create a presentation on deep learning
```

### **With Length/Style**
```
Make a 30s plain-English summary of this paper
Create a 5min technical presentation on attention
Write a 90s press-release about this research
```

### **Refinement**
```
Make slide 2 longer and more creative
Simplify the third slide
Keep speaker notes to two points in slide 1
Make it more technical with equations
```

### **Math Queries**
```
What is the attention formula?
Explain the loss function with equations
Show me the key mathematical equations
```

---

## 💡 Example Runs (Input → Output)

### **Example 1: Basic Query with Math**

**Input:**
```
Make 3 slides on attention mechanisms with equations and speaker notes
```

**Output (Streamlit UI):**
```
**Slide 1: Introduction to Attention Mechanisms**

Speaker Notes:
- Attention mechanisms allow models to focus on relevant parts of input sequences
- Introduced to address limitations of fixed-length context vectors in sequence-to-sequence models
- Key innovation: dynamic weighting of input elements based on relevance

Equation: Attention(Q, K, V) = softmax(QK^T/√d_k)V

---

**Slide 2: Scaled Dot-Product Attention**

Speaker Notes:
- Computes attention weights using queries (Q), keys (K), and values (V)
- Scaling factor √d_k prevents dot products from growing too large
- Softmax normalizes weights to sum to 1

Equation: score(q, k) = q · k / √d_k

---

**Slide 3: Multi-Head Attention**

Speaker Notes:
- Allows model to attend to different representation subspaces simultaneously
- Each head learns different aspects of relationships between tokens
- Outputs are concatenated and linearly transformed

Equation: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
```

**Terminal Output:**
```
[INFO] Found 23 chunks with mathematical content
[MATH-AWARE RETRIEVAL]
Query contains 0 equation(s) (text-based math query)
Retrieved 5 chunks with avg math similarity: 0.682

EVALUATION METRICS
1. FACTUALITY PROXY
   Total Claims: 9
   Grounded Claims: 8
   Factuality Score: 88.89% ✅

2. CITATION COVERAGE
   Total Claims: 9
   With Provenance: 8
   Coverage Score: 88.89% ✅
```

---

### **Example 2: Slide-Specific Refinement**

**Input (Initial):**
```
Make 4 slides on homomorphic encryption with speaker notes
```

**Output (Initial):** *(4 slides generated with 3 speaker notes each)*

**Input (Refinement):**
```
Make slide 2 shorter and simpler, keep speaker points to two
```

**Output (Refined - Terminal Delta Display):**
```
================================================================================
SLIDE-BY-SLIDE CHANGES
================================================================================

📍 SLIDE 2: Modified

WHY IT CHANGED:
─────────────────────────────────────────────────────────────────────────────
Refinement requested: "shorter and simpler, keep speaker points to two"
Applied to: Slide 2 only (3 other slides unchanged)

CHANGES MADE:
─────────────────────────────────────────────────────────────────────────────
- Reduced speaker notes from 3 to 2 points
- Simplified technical language to be more accessible
- Maintained equation integrity

OLD VERSION:
─────────────────────────────────────────────────────────────────────────────
Speaker Notes:
- Homomorphic encryption enables computations on encrypted data without decryption
- Two parties can encrypt data with public key and perform operations
- Results preserve privacy while enabling collaborative computation

NEW VERSION:
─────────────────────────────────────────────────────────────────────────────
Speaker Notes:
- Homomorphic encryption lets you compute on encrypted data directly
- Two parties can work together on private data without revealing it

✅ Reconstruction Complete: 1 modified, 3 unchanged (4 total slides)
================================================================================
```

---

### **Example 3: Length & Style Variations**

**Input 1 (30-second, Plain-English):**
```
Make a 30s plain-English summary of transformers
```

**Output 1:**
```
**Transformers: A Quick Overview**

Transformers are a modern AI architecture that processes entire sequences at once, unlike older models that read one word at a time. They use "attention" to figure out which words are important for understanding context. This makes them faster and better at understanding language. Popular models like ChatGPT and BERT use transformers. They've revolutionized natural language processing and are now used everywhere from translation to content generation.

(~60 words, ~30 seconds to read)
```

**Input 2 (5-minute, Technical):**
```
Make a 5min technical presentation on transformers with equations
```

**Output 2:** *(6-7 slides with detailed technical content, multiple equations, comprehensive speaker notes totaling 500-800 words)*

---

### **Example 4: Math Query with Advanced Retrieval**

**Input:**
```
What is the key equation for scaled dot-product attention?
```

**Terminal Output:**
```
[MATH-AWARE RETRIEVAL]
Query contains 0 equation(s) (semantic math query)
✓ SymPy canonicalization active
Found 12 chunks with mathematical content
Top matches:
  Chunk #34: Math similarity 0.892 (Exact:0.0 Symbols:0.85 Numeric:0.0 Struct:0.95 Density:1.0)
  Chunk #18: Math similarity 0.734 (Exact:0.0 Symbols:0.72 Numeric:0.0 Struct:0.78 Density:0.8)

Retrieved 5 chunks with avg math similarity: 0.756
```

**Output (Streamlit UI):**
```
The key equation for scaled dot-product attention is:

Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q (queries), K (keys), V (values) are input matrices
- d_k is the dimension of the key vectors
- The scaling factor 1/√d_k prevents dot products from becoming too large
- Softmax normalizes the attention weights to sum to 1

This formula computes a weighted sum of values, where weights are determined
by the compatibility between queries and keys.
```

---

### **Example 5: Interactive Feedback**

**Input (Initial):**
```
Explain privacy-preserving recommendation systems
```

**Output:** *(10 bullet points generated)*

**User Action in Feedback UI:**
- ✅ Accepted 7 bullets
- ❌ Rejected 3 bullets (too technical, redundant, or incorrect)

**Clicked: "Generate Refined Answer Based on Feedback"**

**System Auto-Generated Refinement Query:**
```
[Automatic refinement based on feedback]
Regenerate the answer, keeping only the accepted statements and
replacing the rejected ones. Focus on:
- Avoiding overly technical jargon
- Removing redundant information
- Ensuring factual accuracy
```

**Output (Refined):**
```
**Privacy-Preserving Recommendation Systems**

[7 accepted bullets retained verbatim]

[3 new bullets generated to replace rejected ones, incorporating feedback]

✨ Refined based on your feedback!
```

---

### **Example 6: Full Evaluation Output**

**Input:**
```
Make 5 slides on attention mechanisms
```

**Terminal Evaluation (Complete):**
```
[INFO] 📊 Reference script loaded for quality evaluation (ROUGE/BERTScore)
[INFO] ✓ Loaded reference script for: NIPS-2017-attention-is-all-you-need-Paper.pdf

[INFO] Found 45 chunks with mathematical content
[INFO] Successfully parsed 42 chunks with SymPy canonicalization
[INFO] Equation index contains 28 unique equations

[MATH-AWARE RETRIEVAL]
Query contains 0 equation(s)
✓ SymPy canonicalization active
Found 15 chunks with mathematical content
Top matches:
  Chunk #12: Math similarity 0.845
  Chunk #34: Math similarity 0.789
  Chunk #7:  Math similarity 0.723

Retrieved 5 chunks with avg math similarity: 0.756

================================================================================
EVALUATION METRICS
================================================================================

1. FACTUALITY PROXY
   Total Claims: 25
   Grounded Claims: 22
   Ungrounded Claims: 3
   Factuality Score: 88.00% ✅ (threshold: 0.3)
   Average Overlap: 0.645

2. CITATION COVERAGE
   Total Claims: 25
   With Provenance: 21
   Without Provenance: 4
   Coverage Score: 84.00% ✅ (threshold: 0.25)
   Average Provenance: 0.512
   Source Chunks Used: 5

3. ROUGE SCORES
   ROUGE-1:
     Precision: 0.4521
     Recall:    0.5834
     F-Measure: 0.5093 ✅
   ROUGE-2:
     Precision: 0.2345
     Recall:    0.3012
     F-Measure: 0.2634 ✅
   ROUGE-L:
     Precision: 0.4012
     Recall:    0.5123
     F-Measure: 0.4501 ✅

4. BERTSCORE
   Precision: 0.8234
   Recall:    0.8456
   F1:        0.8344 ✅

================================================================================

[REFINED ANSWER]
================================================================================
[5 slides with equations and speaker notes displayed here]
================================================================================
```

---

### **Example 7: Change Tracking (GitHub-Style Diff)**

**Input (Refinement):**
```
Make slide 3 more creative and interesting
```

**Terminal Output (Color-Coded in Terminal):**
```
================================================================================
SLIDE 3 - CHANGES
================================================================================

LINE-BY-LINE DIFF:
─────────────────────────────────────────────────────────────────────────────
  1 | **Slide 3: Multi-Head Attention**
  2 |
- 3 | Speaker Notes:
+ 3 | Speaker Notes: 🎯
- 4 | - Multi-head attention allows the model to focus on different aspects
+ 4 | - Imagine having multiple "attention spotlights" that can focus on different 
+   | aspects of the input simultaneously - that's multi-head attention!
- 5 | - Each head learns different patterns
+ 5 | - Each "head" is like a specialist, learning to detect different patterns:
+   | one might focus on syntax, another on semantics, and another on long-range
+   | dependencies
- 6 | - Outputs are concatenated and transformed
+ 6 | - These multiple perspectives are then combined (concatenated) and mixed 
+   | together through a learned transformation, giving a rich, nuanced understanding

Legend: - = removed (red), + = added (green), | = unchanged (grey)
================================================================================
```

---

### **Example 8: Script Style Comparison**

**Same Query, Different Styles:**

**Technical Style:**
```
Make a 90s technical presentation on transformers

OUTPUT:
**Slide 1: Transformer Architecture**
The Transformer model employs a self-attention mechanism to process input
sequences in parallel, eliminating the sequential constraints of RNNs.
The architecture consists of encoder-decoder stacks with multi-head attention
layers and position-wise feed-forward networks.

Equation: Attention(Q,K,V) = softmax(QK^T/√d_k)V
...
```

**Plain-English Style:**
```
Make a 90s plain-english presentation on transformers

OUTPUT:
**Slide 1: What Are Transformers?**
Transformers are a new way for computers to understand language. Instead of
reading one word at a time (like we do), they look at all words at once.
This makes them much faster and better at understanding context and meaning.
Think of it like being able to see an entire paragraph at a glance instead
of reading word by word.
...
```

**Press-Release Style:**
```
Make a 90s press-release about transformers

OUTPUT:
**Revolutionary Transformer Technology Changes AI Forever**
In a groundbreaking development, researchers have unveiled Transformers - 
a game-changing AI architecture that's redefining what's possible in natural
language processing. This innovation has already powered breakthrough
applications from ChatGPT to advanced translation systems, marking a new
era in artificial intelligence capabilities.
...
```

---

## 📈 Performance Optimization

### **For Faster Retrieval**
- Reduce `k` (number of chunks) in retrieval
- Use smaller chunk sizes
- Disable math-aware retrieval for non-math queries

### **For Better Accuracy**
- Increase `k` for more context
- Adjust `math_boost_factor` for math-heavy documents
- Fine-tune factuality/citation thresholds

### **For Memory Efficiency**
- Use CPU-based embeddings (default)
- Reduce chunk overlap
- Clear old conversation history periodically

---

## 🤝 Contributing

Contributions welcome! Key areas for improvement:
- Additional LLM model support
- More evaluation metrics
- Enhanced equation parsing
- Better UI/UX features
- Performance optimizations

---

## 📝 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments
- **Advanced Features**: Math-aware retrieval, evaluation system, feedback loop
- **Technologies**: LangChain, ChromaDB, Ollama, Streamlit, SymPy, Transformers

---

## 📞 Support & Issues

### **Known Issues**

1. **BERTScore Import Error**: Related to PyTorch/Transformers compatibility
   - **Status**: Non-critical (system works without it)
   - **Workaround**: Use ROUGE + Factuality + Coverage metrics

2. **PDF Text Extraction**: Complex LaTeX may not parse perfectly
   - **Mitigation**: Math canonicalization handles variations

### **Getting Help**

- Check terminal output for detailed error messages
- Review conversation history in `conversation_history.json`
- Inspect retrieved chunks in feedback UI
- Verify PDF indexing completed successfully

---

## 🚀 Quick Start Checklist

- [ ] Python 3.12+ installed
- [ ] Virtual environment created and activated
- [ ] Core dependencies installed (`pip install -r requirements.txt`)
- [ ] Ollama installed with models (`llama3.2:latest`, `nomic-embed-text:latest`)
- [ ] SymPy installed for math features (`pip install sympy`)
- [ ] Optional: Evaluation dependencies (`pip install rouge-score sentence-transformers`)
- [ ] Streamlit running (`streamlit run streamlit_app.py`)
- [ ] PDF uploaded and indexed
- [ ] First query submitted successfully
- [ ] Terminal shows evaluation metrics

---

**Happy RAG-ing! 🎉✨**
