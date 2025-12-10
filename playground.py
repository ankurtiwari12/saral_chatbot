# Langchain Document Loaders: [https://python.langchain.com/docs/integrations/document_loaders/]
# Langchain Text Splitters: [https://python.langchain.com/docs/concepts/text_splitters/]
# Recursive Character Text Splitter doc: [https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html]
# Embedding Generation options: [https://python.langchain.com/docs/integrations/text_embedding/]
# Ollama Chat Model: [https://python.langchain.com/docs/integrations/llms/ollama/]

# weak for unstructured pdf (email threads)
from langchain_community.document_loaders import PyPDFDirectoryLoader    # langchain.document_loader depricated 
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# from langchain_community.embeddings.bedrock import BedrockEmbeddings    # aws embeddings
from langchain_ollama import OllamaEmbeddings    # ollama embeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import argparse
import os
import re
import json
from typing import Optional, Dict, Tuple, List, Set
import difflib
import hashlib

# SymPy for advanced math parsing and canonicalization
try:
    from sympy import sympify, simplify, srepr, latex
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("[WARNING] SymPy not installed. Math canonicalization disabled. Install: pip install sympy")

# Evaluation metrics (terminal-only, not in UI)
try:
    from evaluation_metrics import (
        evaluate_response,
        print_evaluation_report,
        get_evaluation_summary
    )
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    print("[INFO] Evaluation metrics module not found. Evaluation disabled.")

# CONSTANTS
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(SCRIPT_DIR, "chroma_db")
COLLECTION_NAME = "rag_tutorial"
DATA_PATH = os.path.join(SCRIPT_DIR, "pdfs")
HISTORY_FILE = os.path.join(SCRIPT_DIR, "conversation_history.json")

# Conversation history storage (in-memory, loaded from persistent storage)
conversation_history: Dict[str, Dict] = {}

# Feedback storage file
FEEDBACK_FILE = os.path.join(SCRIPT_DIR, "feedback_data.json")

# Reference scripts for ROUGE/BERTScore evaluation
REFERENCE_SCRIPTS_PATH = os.path.join(SCRIPT_DIR, "reference_scripts.json")

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'      # Removed content
    GREEN = '\033[92m'    # Added content
    YELLOW = '\033[93m'   # Changed content
    BLUE = '\033[94m'     # Info
    CYAN = '\033[96m'     # Headers
    GREY = '\033[90m'     # Unchanged/context
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'       # Reset
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_GREY = '\033[100m'


# ===========================
# ADVANCED MATH-AWARE RETRIEVAL SYSTEM
# Using SymPy + Canonicalization + Advanced Matching
# ===========================

# Global equation index for fast lookups
EQUATION_INDEX = {}  # {equation_id: [chunk_ids]}


def parse_and_canonicalize_equation(equation_str: str) -> Dict:
    """
    Parse LaTeX equation with SymPy and canonicalize it.
    
    Returns:
        Dict with canonical form, original, parsing status
    """
    result = {
        'original': equation_str,
        'canonical': None,
        'canonical_str': None,
        'parsed': False,
        'symbols': set(),
        'numeric_value': None
    }
    
    if not SYMPY_AVAILABLE:
        result['canonical_str'] = equation_str.strip()
        return result
    
    try:
        # Try to parse as LaTeX first
        try:
            expr = parse_latex(equation_str)
            result['parsed'] = True
        except:
            # Fallback: try sympify (handles plain math notation)
            expr = sympify(equation_str, evaluate=False)
            result['parsed'] = True
        
        # Canonicalize: simplify and normalize
        canonical_expr = simplify(expr)
        result['canonical'] = canonical_expr
        result['canonical_str'] = srepr(canonical_expr)  # Stable string representation
        
        # Extract symbols
        result['symbols'] = {str(s) for s in canonical_expr.free_symbols}
        
        # Try to get numeric value (if it's a constant expression)
        try:
            result['numeric_value'] = float(canonical_expr.evalf())
        except:
            pass
            
    except Exception as e:
        # Parsing failed, use normalized string as fallback
        result['canonical_str'] = re.sub(r'\s+', '', equation_str.lower())
    
    return result


def create_equation_identifier(equation_str: str, use_canonical: bool = True) -> str:
    """
    Create stable equation identifier using SHA256 of canonical form.
    
    Args:
        equation_str: LaTeX equation string
        use_canonical: Use SymPy canonicalization if available
        
    Returns:
        SHA256 hash (first 16 characters)
    """
    if use_canonical and SYMPY_AVAILABLE:
        parsed = parse_and_canonicalize_equation(equation_str)
        identifier_str = parsed['canonical_str'] if parsed['canonical_str'] else equation_str
    else:
        # Fallback: normalize whitespace and case
        identifier_str = re.sub(r'\s+', '', equation_str.lower())
    
    # SHA256 hash (more robust than MD5)
    equation_id = hashlib.sha256(identifier_str.encode()).hexdigest()[:16]
    
    return equation_id


def check_numeric_equivalence(eq1_str: str, eq2_str: str, tolerance: float = 1e-6) -> bool:
    """
    Check if two equations are numerically equivalent.
    
    Args:
        eq1_str, eq2_str: Equation strings
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if numerically equivalent
    """
    if not SYMPY_AVAILABLE:
        return False
    
    try:
        parsed1 = parse_and_canonicalize_equation(eq1_str)
        parsed2 = parse_and_canonicalize_equation(eq2_str)
        
        # If both have numeric values, compare them
        if parsed1['numeric_value'] is not None and parsed2['numeric_value'] is not None:
            return abs(parsed1['numeric_value'] - parsed2['numeric_value']) < tolerance
        
        # If both have canonical forms, compare structurally
        if parsed1['canonical'] is not None and parsed2['canonical'] is not None:
            diff = simplify(parsed1['canonical'] - parsed2['canonical'])
            return diff == 0
            
    except Exception:
        pass
    
    return False


def compute_jaccard_similarity(set1: Set, set2: Set) -> float:
    """
    Compute Jaccard similarity between two sets.
    
    Args:
        set1, set2: Sets to compare
        
    Returns:
        Jaccard similarity (0-1)
    """
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def extract_equations(text: str) -> List[str]:
    """
    Extract mathematical equations from text.
    Supports both inline and display LaTeX formats.
    
    Args:
        text: Text containing LaTeX equations
        
    Returns:
        List of extracted equations (normalized)
    """
    equations = []
    
    # Pattern 1: Display math with $$ ... $$
    display_math_double = re.findall(r'\$\$(.*?)\$\$', text, re.DOTALL)
    equations.extend(display_math_double)
    
    # Pattern 2: Display math with \[ ... \]
    display_math_bracket = re.findall(r'\\\[(.*?)\\\]', text, re.DOTALL)
    equations.extend(display_math_bracket)
    
    # Pattern 3: Inline math with $ ... $
    inline_math = re.findall(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', text)
    equations.extend(inline_math)
    
    # Pattern 4: Inline math with \( ... \)
    inline_math_paren = re.findall(r'\\\((.*?)\\\)', text)
    equations.extend(inline_math_paren)
    
    # Pattern 5: LaTeX environments (equation, align, etc.)
    env_pattern = r'\\begin\{(equation|align|gather|multline|eqnarray)\*?\}(.*?)\\end\{\1\*?\}'
    env_matches = re.findall(env_pattern, text, re.DOTALL)
    equations.extend([match[1] for match in env_matches])
    
    # Normalize: strip whitespace and remove empty strings
    equations = [eq.strip() for eq in equations if eq.strip()]
    
    return equations


def create_equation_identifier(equation: str) -> str:
    """
    Create a unique identifier for an equation.
    
    Args:
        equation: LaTeX equation string
        
    Returns:
        MD5 hash of normalized equation
    """
    # Normalize: remove whitespace and convert to lowercase
    normalized = re.sub(r'\s+', '', equation.lower())
    
    # Create hash
    equation_id = hashlib.md5(normalized.encode()).hexdigest()[:12]
    
    return equation_id


def extract_math_symbols(equation: str) -> Set[str]:
    """
    Extract mathematical symbols and operators from equation.
    
    Args:
        equation: LaTeX equation string
        
    Returns:
        Set of symbols found in equation
    """
    symbols = set()
    
    # Common LaTeX commands
    latex_commands = re.findall(r'\\[a-zA-Z]+', equation)
    symbols.update(latex_commands)
    
    # Greek letters
    greek_pattern = r'\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)'
    greek_letters = re.findall(greek_pattern, equation, re.IGNORECASE)
    symbols.update([f'\\{g}' for g in greek_letters])
    
    # Operators
    operators = re.findall(r'[\+\-\*/=<>≤≥≠≈∑∏∫∂∇]', equation)
    symbols.update(operators)
    
    # Variables (single letters, possibly with subscripts/superscripts)
    variables = re.findall(r'\b[a-zA-Z]\b', equation)
    symbols.update(variables)
    
    return symbols


def analyze_chunk_math(text: str, chunk_id: str = None) -> Dict:
    """
    Advanced mathematical content analysis using SymPy canonicalization.
    
    Args:
        text: Text chunk to analyze
        chunk_id: Optional chunk identifier for indexing
        
    Returns:
        Dictionary with comprehensive math analysis
    """
    equations = extract_equations(text)
    
    # Parse and canonicalize all equations
    parsed_equations = []
    equation_ids = []
    all_symbols = set()
    canonical_forms = []
    
    for eq in equations:
        # Parse with SymPy
        parsed = parse_and_canonicalize_equation(eq)
        parsed_equations.append(parsed)
        
        # Create stable equation ID from canonical form
        eq_id = create_equation_identifier(eq, use_canonical=True)
        equation_ids.append(eq_id)
        
        # Update global equation index
        if chunk_id and eq_id:
            if eq_id not in EQUATION_INDEX:
                EQUATION_INDEX[eq_id] = []
            if chunk_id not in EQUATION_INDEX[eq_id]:
                EQUATION_INDEX[eq_id].append(chunk_id)
        
        # Collect symbols from canonical form
        if parsed['symbols']:
            all_symbols.update(parsed['symbols'])
        else:
            # Fallback to regex-based extraction
            all_symbols.update(extract_math_symbols(eq))
        
        # Store canonical representation
        if parsed['canonical_str']:
            canonical_forms.append(parsed['canonical_str'])
    
    # Calculate math density (equations per 1000 characters)
    math_density = (len(equations) / max(len(text), 1)) * 1000 if text else 0
    
    return {
        'equations': equations,
        'equation_ids': equation_ids,
        'canonical_forms': canonical_forms,
        'parsed_equations': parsed_equations,
        'math_symbols': list(all_symbols),
        'has_math': len(equations) > 0,
        'math_density': math_density,
        'equation_count': len(equations),
        'sympy_parsed': SYMPY_AVAILABLE and any(p['parsed'] for p in parsed_equations)
    }


def compute_advanced_math_similarity(query_math: Dict, chunk_math: Dict) -> Dict:
    """
    Advanced math similarity using multiple matching strategies.
    
    Args:
        query_math: Math analysis of query
        chunk_math: Math analysis of chunk
        
    Returns:
        Dictionary with detailed similarity scores
    """
    scores = {
        'exact_match': 0.0,      # Exact equation ID matches
        'jaccard_symbols': 0.0,   # Jaccard similarity of symbols
        'numeric_equiv': 0.0,     # Numeric equivalence
        'structural': 0.0,        # Structural similarity
        'density_bonus': 0.0,     # Math density boost
        'total': 0.0
    }
    
    # No math in query
    if not query_math['has_math']:
        if chunk_math['has_math']:
            scores['total'] = 0.3  # Slight bonus for math-containing chunks
        return scores
    
    # Math in query but not in chunk (strong penalty)
    if not chunk_math['has_math']:
        return scores
    
    # Both have math - compute multi-faceted similarity
    
    # 1. Exact equation ID matches (30% weight)
    query_eq_ids = set(query_math['equation_ids'])
    chunk_eq_ids = set(chunk_math['equation_ids'])
    
    if query_eq_ids and chunk_eq_ids:
        exact_matches = len(query_eq_ids & chunk_eq_ids)
        scores['exact_match'] = exact_matches / len(query_eq_ids)
    
    # 2. Jaccard similarity of symbols (25% weight)
    query_symbols = set(query_math['math_symbols'])
    chunk_symbols = set(chunk_math['math_symbols'])
    
    if query_symbols and chunk_symbols:
        scores['jaccard_symbols'] = compute_jaccard_similarity(query_symbols, chunk_symbols)
    
    # 3. Numeric equivalence check (20% weight)
    if SYMPY_AVAILABLE and 'parsed_equations' in query_math and 'parsed_equations' in chunk_math:
        equiv_count = 0
        total_comparisons = 0
        
        for q_parsed in query_math.get('parsed_equations', []):
            for c_parsed in chunk_math.get('parsed_equations', []):
                if q_parsed['original'] and c_parsed['original']:
                    total_comparisons += 1
                    if check_numeric_equivalence(q_parsed['original'], c_parsed['original']):
                        equiv_count += 1
        
        if total_comparisons > 0:
            scores['numeric_equiv'] = equiv_count / total_comparisons
    
    # 4. Structural similarity using canonical forms (15% weight)
    query_canonical = set(query_math.get('canonical_forms', []))
    chunk_canonical = set(chunk_math.get('canonical_forms', []))
    
    if query_canonical and chunk_canonical:
        scores['structural'] = compute_jaccard_similarity(query_canonical, chunk_canonical)
    
    # 5. Math density bonus (10% weight)
    if chunk_math['math_density'] > 0:
        scores['density_bonus'] = min(chunk_math['math_density'] / 10.0, 1.0)
    
    # Combined score with weights
    scores['total'] = (
        scores['exact_match'] * 0.30 +
        scores['jaccard_symbols'] * 0.25 +
        scores['numeric_equiv'] * 0.20 +
        scores['structural'] * 0.15 +
        scores['density_bonus'] * 0.10
    )
    
    return scores


def compute_math_similarity_score(query_math: Dict, chunk_math: Dict) -> float:
    """
    Compute overall math similarity score (wrapper for advanced scoring).
    
    Args:
        query_math: Math analysis of query
        chunk_math: Math analysis of chunk
        
    Returns:
        Similarity score (0-1)
    """
    scores = compute_advanced_math_similarity(query_math, chunk_math)
    return min(scores['total'], 1.0)


def math_aware_retrieval(db, query_txt: str, k: int = 5, math_boost_factor: float = 0.4):
    """
    Advanced math-aware retrieval with SymPy canonicalization and inverted index.
    
    Args:
        db: Chroma database instance
        query_txt: Query text
        k: Number of results to return
        math_boost_factor: Boost factor for math similarity (0-1)
        
    Returns:
        List of (Document, score) tuples, re-ranked with advanced math matching
    """
    # Analyze query for mathematical content
    query_math = analyze_chunk_math(query_txt)
    
    # Fast exact match using inverted equation index
    exact_match_chunk_ids = set()
    if query_math['has_math'] and query_math['equation_ids']:
        for eq_id in query_math['equation_ids']:
            if eq_id in EQUATION_INDEX:
                exact_match_chunk_ids.update(EQUATION_INDEX[eq_id])
        
        if exact_match_chunk_ids:
            print(f"[INFO] Found {len(exact_match_chunk_ids)} chunk(s) with exact equation matches via index")
    
    # Retrieve more candidates than needed
    initial_k = min(k * 3, 25)  # Retrieve 3x more candidates
    results = db.similarity_search_with_score(query_txt, k=initial_k)
    
    if not results:
        return []
    
    # Re-rank results with advanced math matching
    boosted_results = []
    detailed_scores = []
    
    for doc, base_score in results:
        chunk_id = doc.metadata.get('id', '')
        chunk_text = doc.page_content
        
        # Load chunk math metadata
        chunk_math = {
            'has_math': doc.metadata.get('has_math', False),
            'math_density': doc.metadata.get('math_density', 0),
            'equation_count': doc.metadata.get('equation_count', 0),
            'equation_ids': doc.metadata.get('equation_ids', '').split(',') if doc.metadata.get('equation_ids') else [],
            'math_symbols': doc.metadata.get('math_symbols', '').split(',') if doc.metadata.get('math_symbols') else [],
            'canonical_forms': doc.metadata.get('canonical_forms', '').split('|') if doc.metadata.get('canonical_forms') else [],
            'parsed_equations': []  # Not stored in metadata, would need re-parsing
        }
        
        # If metadata not available, analyze on the fly
        if not chunk_math['has_math'] and not doc.metadata.get('equation_count'):
            chunk_math = analyze_chunk_math(chunk_text, chunk_id)
        
        # Compute advanced math similarity
        math_scores = compute_advanced_math_similarity(query_math, chunk_math)
        total_math_score = math_scores['total']
        
        # Extra boost for exact matches from inverted index
        if chunk_id in exact_match_chunk_ids:
            total_math_score = min(total_math_score * 1.2, 1.0)  # 20% boost
        
        # Combine scores: base similarity + math boost
        # Lower base_score is better (distance), so subtract math boost
        final_score = base_score - (total_math_score * math_boost_factor)
        
        boosted_results.append((doc, final_score, total_math_score, math_scores))
        detailed_scores.append(math_scores)
    
    # Sort by final score (lower is better)
    boosted_results.sort(key=lambda x: x[1])
    
    # Log detailed math retrieval info
    if query_math['has_math']:
        print(f"\n{Colors.CYAN}[MATH-AWARE RETRIEVAL]{Colors.END}")
        print(f"Query contains {query_math['equation_count']} equation(s)")
        if SYMPY_AVAILABLE and query_math.get('sympy_parsed'):
            print(f"✓ SymPy canonicalization active")
        
        math_boosted_count = sum(1 for _, _, math_score, _ in boosted_results[:k] if math_score > 0.1)
        print(f"Top {k} results: {math_boosted_count} math-boosted chunk(s)")
        
        # Show detailed scores for top result
        if boosted_results and detailed_scores:
            top_scores = boosted_results[0][3]
            print(f"Top result scores: Exact:{top_scores['exact_match']:.2f} | "
                  f"Jaccard:{top_scores['jaccard_symbols']:.2f} | "
                  f"NumEq:{top_scores['numeric_equiv']:.2f} | "
                  f"Struct:{top_scores['structural']:.2f}")
        print()
    
    # Return top k results in (doc, score) format
    return [(doc, score) for doc, score, _, _ in boosted_results[:k]]


# def load_documents():
#     document_loader = PyPDFDirectoryLoader(DATA_PATH)
#     return document_loader.load()


def load_documents():
    """Load documents from the default pdfs/ directory"""
    # Check if PDF directory exists
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] PDF directory does not exist: {DATA_PATH}")
        return []
    
    # Check if there are any PDF files
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"[ERROR] No PDF files found in: {DATA_PATH}")
        print(f"[INFO] Please add PDF files to the directory: {os.path.abspath(DATA_PATH)}")
        return []
    
    print(f"[INFO] Found {len(pdf_files)} PDF file(s): {', '.join(pdf_files)}")
    
    # documents = []
    # for file in os.listdir(DATA_PATH):
    #     if file.endswith('.pdf'):
    #         loader = UnstructuredPDFLoader(f'{DATA_PATH}/{file}')
    #         file_docs = loader.load()
    #         documents.extend(file_docs)

    try:
        document_loader = PyPDFDirectoryLoader(DATA_PATH)
        documents = document_loader.load()
        
        if documents:
            from collections import Counter
            doc_sources = Counter(doc.metadata.get('source', 'unknown') for doc in documents)
            print(f"[INFO] Loaded {len(documents)} document page(s) from {len(doc_sources)} file(s):")
            for source, count in doc_sources.items():
                print(f"    {os.path.basename(source)} => {count} page(s)")
        else:
            print("[WARNING] PDF loader returned empty documents. PDFs may be corrupted or unreadable.")
        
        return documents
    except Exception as e:
        print(f"[ERROR] Failed to load documents: {e}")
        return []


def load_documents_from_files(file_paths: List[str]) -> List[Document]:
    """
    Load documents from a list of PDF file paths.
    This is used when files are uploaded via Streamlit.
    
    Args:
        file_paths: List of absolute file paths to PDF files
        
    Returns:
        List of Document objects
    """
    if not file_paths:
        print("[ERROR] No file paths provided")
        return []
    
    documents = []
    print(f"[INFO] Loading {len(file_paths)} uploaded PDF file(s)...")
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}")
            continue
            
        if not file_path.endswith('.pdf'):
            print(f"[WARNING] Skipping non-PDF file: {file_path}")
            continue
        
        try:
            loader = PyPDFLoader(file_path)
            file_docs = loader.load()
            documents.extend(file_docs)
            print(f"    ✓ Loaded {os.path.basename(file_path)} => {len(file_docs)} page(s)")
        except Exception as e:
            print(f"    ✗ Failed to load {os.path.basename(file_path)}: {e}")
    
    if documents:
        from collections import Counter
        doc_sources = Counter(doc.metadata.get('source', 'unknown') for doc in documents)
        print(f"[INFO] Successfully loaded {len(documents)} page(s) from {len(doc_sources)} file(s)")
    else:
        print("[WARNING] No documents loaded from uploaded files")
    
    return documents


def load_reference_script(pdf_filename: str) -> Optional[str]:
    """
    Load human-authored reference script for a given PDF file.
    Used for ROUGE/BERTScore evaluation.
    
    Args:
        pdf_filename: Name of the PDF file (e.g., "paper1.pdf")
        
    Returns:
        Reference script text if available, None otherwise
    """
    if not os.path.exists(REFERENCE_SCRIPTS_PATH):
        print(f"[INFO] No reference scripts file found at: {REFERENCE_SCRIPTS_PATH}")
        return None
    
    try:
        with open(REFERENCE_SCRIPTS_PATH, 'r', encoding='utf-8') as f:
            reference_data = json.load(f)
        
        # Skip internal metadata fields
        if pdf_filename.startswith('_'):
            return None
        
        if pdf_filename in reference_data:
            script_info = reference_data[pdf_filename]
            if isinstance(script_info, dict) and 'reference_script' in script_info:
                print(f"[INFO] ✓ Loaded reference script for: {pdf_filename}")
                return script_info['reference_script']
            else:
                print(f"[WARNING] Invalid format for reference script: {pdf_filename}")
                return None
        else:
            print(f"[INFO] No reference script found for: {pdf_filename}")
            return None
            
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse reference scripts JSON: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load reference scripts: {e}")
        return None


def get_reference_script_for_query(uploaded_filenames: List[str]) -> Optional[str]:
    """
    Get the best reference script for the current query based on uploaded files.
    
    If multiple files are uploaded, prioritizes the first one with a reference script.
    
    Args:
        uploaded_filenames: List of uploaded PDF filenames
        
    Returns:
        Reference script text if available, None otherwise
    """
    if not uploaded_filenames:
        return None
    
    for filename in uploaded_filenames:
        # Extract just the filename if it's a full path
        if os.path.sep in filename:
            filename = os.path.basename(filename)
        
        reference = load_reference_script(filename)
        if reference:
            return reference
    
    return None


def split_documents(documents: list[Document]):    # type hinting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    
    # Enhance chunks with advanced math metadata
    print(f"[INFO] Analyzing mathematical content in {len(chunks)} chunks...")
    if SYMPY_AVAILABLE:
        print(f"[INFO] ✓ SymPy available - using canonicalization & advanced matching")
    else:
        print(f"[WARNING] SymPy not available - using regex-only matching")
    
    math_chunk_count = 0
    sympy_parsed_count = 0
    
    for idx, chunk in enumerate(chunks):
        # Get or create chunk ID for indexing
        chunk_id = chunk.metadata.get('id', f'chunk_{idx}')
        
        # Advanced math analysis with SymPy
        math_analysis = analyze_chunk_math(chunk.page_content, chunk_id=chunk_id)
        
        # Add comprehensive math metadata to chunk
        chunk.metadata['has_math'] = math_analysis['has_math']
        chunk.metadata['math_density'] = math_analysis['math_density']
        chunk.metadata['equation_count'] = math_analysis['equation_count']
        chunk.metadata['equation_ids'] = ','.join(math_analysis['equation_ids'])
        chunk.metadata['math_symbols'] = ','.join(math_analysis['math_symbols'][:30])  # Store top 30 symbols
        chunk.metadata['canonical_forms'] = '|'.join(math_analysis.get('canonical_forms', [])[:10])  # Store top 10 canonical forms
        chunk.metadata['sympy_parsed'] = math_analysis.get('sympy_parsed', False)
        
        if math_analysis['has_math']:
            math_chunk_count += 1
        if math_analysis.get('sympy_parsed'):
            sympy_parsed_count += 1
    
    print(f"[INFO] Found {math_chunk_count} chunks with mathematical content")
    if SYMPY_AVAILABLE:
        print(f"[INFO] Successfully parsed {sympy_parsed_count} chunks with SymPy canonicalization")
    print(f"[INFO] Equation index contains {len(EQUATION_INDEX)} unique equations")
    
    return chunks


# pay-as-you-go service
# def get_embedding():
#     embeddings = BedrockEmbeddings(
#         credentials_profile_name="default", region_name="us-east-1"
#     )

#     return embeddings


def get_embedding():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")    # best embedding model avialable in ollama
    return embeddings


# attaching different ids to each chunk, for DB update later on
def calculate_chunk_ids(chunks):
    curr_chunk_idx = 0
    prev_page_id = ""

    for chunk in chunks:
        curr_page_id = f"{chunk.metadata['source']}:{chunk.metadata['page']}"
        
        if curr_page_id == prev_page_id:
            curr_chunk_idx += 1
        else:
            curr_chunk_idx = 0    # reset

        prev_page_id = curr_page_id
        chunk_id = f"{curr_page_id}:{curr_chunk_idx}"

        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_chroma(chunks: list[Document]):
    if not chunks or len(chunks) == 0:
        print("[WARNING] No chunks provided to add to Chroma.")
        return
    
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding(),
        persist_directory=CHROMA_PATH
    )
    
    curr_items = db.get(include=[])    # ids included by default
    curr_ids = set(curr_items["ids"])
    print(f"[INFO] Current Documents in Vector DB: {len(curr_items['ids'])}.")

    # adding docs not in db
    new_chunks = []
    new_chunk_ids = []
    valid_ids = set()

    for chunk in chunks:
        chunk_id = chunk.metadata['id']
        valid_ids.add(chunk_id)

        if not chunk.page_content.strip():
            print(f"[WARNING] Empty content for chunk id: {chunk.metadata['id']}")

        else:
            if chunk_id not in curr_ids:
                new_chunks.append(chunk)
                new_chunk_ids.append(chunk.metadata['id'])

    stale_ids = curr_ids - valid_ids
    if stale_ids:
        db.delete(ids=list(stale_ids))
        print(f"[INFO] Deleted {len(stale_ids)} stale ids.")

    else:
        print("[INFO] No stale chunks detected.")

    if new_chunks:
        try:
            db.add_documents(new_chunks, ids=new_chunk_ids)
            print(f"[INFO] Successfully added {len(new_chunks)} new document chunks to vector store.")
        except Exception as e:
            print(f"[ERROR] Failed to add documents to vector store: {e}")

    else:
        if len(curr_ids) == 0:
            print("[WARNING] No new chunks detected AND vector store is empty!")
            print("[WARNING] This may indicate that:")
            print("  1. Documents were already indexed (try --reset to re-index)")
            print("  2. PDFs failed to load or were empty")
            print("  3. Chunking produced no valid chunks")
        else:
            print(f"[INFO] No new chunks detected. Vector store already contains {len(curr_ids)} chunks.")
    # db.persist() [automatically done in newer versions]


def load_conversation_history() -> Dict[str, Dict]:
    """
    Load conversation history from JSON file.
    
    Returns:
        Dictionary containing conversation history per session
    """
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARNING] Failed to load conversation history: {e}")
            return {}
    return {}


def save_conversation_history(history: Dict[str, Dict]):
    """
    Save conversation history to JSON file.
    
    Args:
        history: Dictionary containing conversation history per session
    """
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"[WARNING] Failed to save conversation history: {e}")


def clear_conversation_history():
    """
    Clear all conversation history from persistent storage.
    """
    global conversation_history
    conversation_history = {}
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
            print(f"[INFO] Conversation history cleared.")
        except IOError as e:
            print(f"[WARNING] Failed to clear conversation history: {e}")
    else:
        print(f"[INFO] No conversation history file found.")


# ============================
# FEEDBACK STORAGE SYSTEM
# ============================

def load_feedback_data() -> Dict:
    """Load feedback data from JSON file."""
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"[WARNING] Failed to load feedback data: {e}")
            return {}
    return {}


def save_feedback_data(feedback_data: Dict):
    """Save feedback data to JSON file."""
    try:
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"[WARNING] Failed to save feedback data: {e}")


def store_bullet_feedback(session_id: str, bullet_id: str, bullet_text: str, 
                          source_chunk: str, source_metadata: Dict,
                          feedback: str, timestamp: str = None):
    """
    Store user feedback for a generated bullet point.
    
    Args:
        session_id: Session identifier
        bullet_id: Unique ID for the bullet
        bullet_text: The generated bullet text
        source_chunk: Source chunk text
        source_metadata: Metadata (file, page, etc.)
        feedback: 'accepted' or 'rejected'
        timestamp: Optional timestamp
    """
    import datetime
    
    feedback_data = load_feedback_data()
    
    if session_id not in feedback_data:
        feedback_data[session_id] = []
    
    feedback_entry = {
        'bullet_id': bullet_id,
        'bullet_text': bullet_text,
        'source_chunk': source_chunk,
        'source_metadata': source_metadata,
        'feedback': feedback,
        'timestamp': timestamp or datetime.datetime.now().isoformat()
    }
    
    feedback_data[session_id].append(feedback_entry)
    save_feedback_data(feedback_data)
    
    print(f"[INFO] Feedback recorded: {feedback} for bullet '{bullet_text[:50]}...'")


def get_feedback_statistics(session_id: str = None) -> Dict:
    """
    Get feedback statistics.
    
    Args:
        session_id: Optional session ID to filter by
        
    Returns:
        Dictionary with acceptance rates and counts
    """
    feedback_data = load_feedback_data()
    
    if session_id:
        entries = feedback_data.get(session_id, [])
    else:
        entries = [item for session in feedback_data.values() for item in session]
    
    if not entries:
        return {'total': 0, 'accepted': 0, 'rejected': 0, 'acceptance_rate': 0.0}
    
    accepted = sum(1 for e in entries if e['feedback'] == 'accepted')
    rejected = sum(1 for e in entries if e['feedback'] == 'rejected')
    
    return {
        'total': len(entries),
        'accepted': accepted,
        'rejected': rejected,
        'acceptance_rate': (accepted / len(entries)) * 100 if entries else 0.0
    }


def clear_feedback_data():
    """Clear all feedback data."""
    if os.path.exists(FEEDBACK_FILE):
        try:
            os.remove(FEEDBACK_FILE)
            print(f"[INFO] Feedback data cleared.")
        except IOError as e:
            print(f"[WARNING] Failed to clear feedback data: {e}")


def detect_refinement_request(query_txt: str, session_id: str = "default", model_name: str = "llama3.2:3b") -> bool:
    """
    Detects if the query is a refinement request using a concrete semantic-based approach.
    Uses keyword scoring + LLM classification for reliable detection.
    
    Args:
        query_txt: The user's query text
        session_id: Session ID to check for conversation history
        model_name: Model name for LLM classification (if needed)
        
    Returns:
        True if the query appears to be a refinement request, False otherwise
    """
    # First, check if there's conversation history - if not, it can't be a refinement
    history = load_conversation_history()
    if session_id not in history or not history[session_id].get("previous_answer"):
        return False
    
    query_lower = query_txt.lower().strip()
    
    # METHOD 1: Keyword-based scoring (concrete and fast)
    # Refinement action keywords (high weight)
    refinement_actions = {
        'make': 3, 'modify': 3, 'change': 3, 'update': 3, 'adjust': 3,
        'refine': 4, 'improve': 3, 'enhance': 3, 'rewrite': 3, 'rephrase': 3,
        'edit': 3, 'revise': 3, 'fix': 2, 'correct': 2, 'clean': 2, 'polish': 2
    }
    
    # Refinement target keywords (medium weight)
    refinement_targets = {
        'it': 2, 'this': 2, 'that': 2, 'answer': 2, 'response': 2,
        'slide': 2, 'section': 2, 'part': 2, 'point': 2
    }
    
    # Refinement quality keywords (medium weight)
    refinement_qualities = {
        'shorter': 2, 'longer': 2, 'concise': 2, 'detailed': 2, 'brief': 2,
        'technical': 2, 'simple': 2, 'simpler': 2, 'clear': 2, 'clearer': 2,
        'better': 2, 'clean': 2, 'cleaner': 2, 'precise': 2, 'accurate': 2,
        'more': 1, 'less': 1, 'too': 1, 'enough': 1
    }
    
    # Question words (negative weight - indicates new question)
    question_words = {
        'what': -3, 'how': -3, 'why': -3, 'when': -3, 'where': -3,
        'who': -3, 'which': -3, 'explain': -2, 'describe': -2, 'tell': -2
    }
    
    # Calculate score
    score = 0
    words = query_lower.split()
    
    for word in words:
        # Remove punctuation for matching
        clean_word = re.sub(r'[^\w]', '', word)
        
        if clean_word in refinement_actions:
            score += refinement_actions[clean_word]
        if clean_word in refinement_targets:
            score += refinement_targets[clean_word]
        if clean_word in refinement_qualities:
            score += refinement_qualities[clean_word]
        if clean_word in question_words:
            score += question_words[clean_word]
    
    # Check for specific refinement patterns (slide numbers, sections, etc.)
    specific_refinement_patterns = [
        r'(slide|section|part|point)\s+\d+',  # "slide 2", "section 3"
        r'refine\s+(only|just|the)',  # "refine only slide 2"
        r'make\s+(slide|section|part)\s+\d+',  # "make slide 2 cleaner"
    ]
    
    for pattern in specific_refinement_patterns:
        if re.search(pattern, query_lower):
            score += 5  # Strong indicator
    
    # METHOD 2: Length and structure analysis
    # Very short queries (< 5 words) with action words are likely refinements
    word_count = len(words)
    if word_count < 5 and score > 0:
        score += 2  # Boost for short refinement requests
    
    # METHOD 3: Decision based on score
    # If score is high enough, it's definitely a refinement
    if score >= 4:
        print(f"[INFO] Detected refinement by keyword scoring (score: {score})")
        return True
    
    # If score is negative (has question words), likely NOT a refinement
    if score <= -2:
        return False
    
    # METHOD 4: For ambiguous cases, use LLM classification
    # This handles edge cases and natural language variations
    if score > 0 or word_count < 8:  # Short or has some refinement indicators
        return classify_with_llm(query_txt, model_name, history[session_id].get("previous_answer", ""))
    
    return False


def classify_with_llm(query_txt: str, model_name: str, previous_answer: str = "") -> bool:
    """
    Uses LLM to classify if the query is a refinement request.
    Provides context about previous answer for better classification.
    
    Args:
        query_txt: The user's query text
        model_name: Model name for LLM
        previous_answer: The previous answer for context
        
    Returns:
        True if LLM classifies it as a refinement request, False otherwise
    """
    # Provide context about previous answer
    answer_preview = previous_answer[:200] + "..." if len(previous_answer) > 200 else previous_answer
    
    classification_prompt = f"""You are analyzing a user's input to determine if they want to REFINE/MODIFY a previous answer or ask a NEW question.

PREVIOUS ANSWER (context):
"{answer_preview}"

USER'S NEW INPUT:
"{query_txt}"

TASK: Determine if the user wants to REFINE the previous answer or ask a NEW question.

REFINEMENT requests include:
- Requests to modify/change/improve the previous answer
- Requests like "make it shorter/longer/technical/simple"
- Requests to refine specific parts: "refine slide 2", "make section 3 cleaner"
- Requests using words: make, modify, change, refine, improve, enhance, rewrite, rephrase
- Requests about quality: shorter, longer, concise, detailed, technical, simple, clear, better

NEW questions include:
- Standalone questions starting with: What, How, Why, When, Where, Who, Which
- Questions asking for explanation: "Explain X", "Describe Y", "Tell me about Z"
- Questions that don't reference the previous answer

IMPORTANT: If the user mentions specific parts (slide, section, part, point) with numbers, it's likely a refinement.
IMPORTANT: If the query is very short (< 10 words) and contains action words (make, change, refine), it's likely a refinement.

Answer with ONLY "YES" (if refinement) or "NO" (if new question):"""
    
    try:
        model = OllamaLLM(model=model_name)
        response = model.invoke(classification_prompt).strip().upper()
        result = response.startswith("YES")
        print(f"[INFO] LLM classification: {'REFINEMENT' if result else 'NEW QUESTION'}")
        return result
    except Exception as e:
        print(f"[WARNING] LLM classification failed: {e}. Defaulting to False.")
        return False


def extract_script_length(query_txt: str) -> Optional[str]:
    """
    Extracts script length from query.
    
    Args:
        query_txt: The user query
        
    Returns:
        Script length: "30s", "90s", "5min", or None
    """
    query_lower = query_txt.lower()
    
    # Pattern matching for various length formats
    length_patterns = [
        (r'30\s*(?:second|sec|s)\b', '30s'),
        (r'90\s*(?:second|sec|s)\b', '90s'),
        (r'5\s*(?:minute|min|m)\b', '5min'),
        (r'five\s*(?:minute|min)\b', '5min'),
        (r'\b30s\b', '30s'),
        (r'\b90s\b', '90s'),
        (r'\b5min\b', '5min'),
        (r'short\b', '30s'),
        (r'brief\b', '30s'),
        (r'medium\b', '90s'),
        (r'long\b', '5min'),
        (r'extended\b', '5min'),
    ]
    
    for pattern, length in length_patterns:
        if re.search(pattern, query_lower):
            print(f"[DEBUG] Detected script length: {length}")
            return length
    
    return None


def extract_script_style(query_txt: str) -> Optional[str]:
    """
    Extracts script style from query.
    
    Args:
        query_txt: The user query
        
    Returns:
        Script style: "technical", "plain-english", "press-release", or None
    """
    query_lower = query_txt.lower()
    
    # Pattern matching for style
    style_patterns = [
        (r'\btechnical\b', 'technical'),
        (r'\bplain[- ]?english\b', 'plain-english'),
        (r'\bsimple\b', 'plain-english'),
        (r'\blayman\b', 'plain-english'),
        (r'\bpress[- ]?release\b', 'press-release'),
        (r'\bmedia\b', 'press-release'),
        (r'\bjournalistic\b', 'press-release'),
        (r'\bacademic\b', 'technical'),
        (r'\bformal\b', 'technical'),
        (r'\bcasual\b', 'plain-english'),
        (r'\bconversational\b', 'plain-english'),
    ]
    
    for pattern, style in style_patterns:
        if re.search(pattern, query_lower):
            print(f"[DEBUG] Detected script style: {style}")
            return style
    
    return None


def extract_refinement_instruction(query_txt: str) -> Tuple[str, Optional[str], Optional[int]]:
    """
    Extracts the specific refinement instruction from the query.
    Handles both general refinements and specific part refinements (e.g., "slide 2").
    
    Args:
        query_txt: The refinement request query
        
    Returns:
        Tuple of (instruction, part_type, part_number)
        - instruction: normalized refinement instruction
        - part_type: "slide", "section", "part", "point", or None
        - part_number: the slide/section number, or None
    """
    query_lower = query_txt.lower().strip()
    
    # First, check for specific part references (slide, section, part, point with numbers)
    specific_part_patterns = [
        r'(slide|section|part|point)\s+#?(\d+)',  # "slide 2", "slide #2"
        r'#(\d+)\s+(slide|section|part|point)',  # "#2 slide", "#3 section"
        r'the\s+#(\d+)\s+(slide|section|part|point)',  # "the #2 slide"
        r'#(\d+)',  # just "#2"
    ]
    
    specific_part = None
    part_type = None
    part_number = None
    
    for pattern in specific_part_patterns:
        match = re.search(pattern, query_lower)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                # Could be (slide, 2) or (2, slide)
                if groups[0].isdigit():
                    # Format: "#2 slide" or "the #2 slide"
                    part_number = int(groups[0])
                    part_type = groups[1]
                elif groups[1].isdigit():
                    # Format: "slide #2" or "slide 2"
                    part_type = groups[0]
                    part_number = int(groups[1])
                specific_part = f"{part_type} {part_number}"
            elif len(groups) == 1:
                # Just a number like "#2", assume it's a slide
                part_type = "slide"
                part_number = int(groups[0])
                specific_part = f"slide {part_number}"
            break
    
    # Extract quality/instruction keywords
    quality_keywords = {
        'technical': 'technical',
        'concise': 'concise',
        'simple': 'simple',
        'simpler': 'simpler',
        'detailed': 'detailed',
        'shorter': 'shorter',
        'longer': 'longer',
        'clear': 'clear',
        'clearer': 'clearer',
        'clean': 'clean',
        'cleaner': 'cleaner',
        'precise': 'precise',
        'brief': 'brief',
        'better': 'better',
    }
    
    # Find quality keywords in query
    found_qualities = []
    for word in query_lower.split():
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in quality_keywords:
            found_qualities.append(quality_keywords[clean_word])
    
    # Build instruction
    instruction_parts = []
    
    # Add specific part if found
    if specific_part:
        instruction_parts.append(f"for {specific_part}")
    
    # Add quality modifiers
    if 'more' in query_lower or 'less' in query_lower:
        if 'more' in query_lower:
            instruction_parts.append("more")
        if 'less' in query_lower:
            instruction_parts.append("less")
    
    # Add quality keywords (if found)
    if found_qualities:
        instruction_parts.extend(found_qualities)
    
    # IMPORTANT: Extract the core request to preserve ALL user intent
    # Remove common prefix words but keep the nuanced descriptors
    cleaned = re.sub(r'^(make|can\s+make|can\s+you\s+make|please\s+make|rewrite|rephrase|refine|improve|change|update|modify)\s+', '', query_lower)
    
    # Remove slide/section references from the instruction
    cleaned = re.sub(r'(in\s+)?(the\s+)?#?\d+\s+(slide|section|part|point)', '', cleaned)
    cleaned = re.sub(r'(slide|section|part|point)\s+#?\d+', '', cleaned)
    
    # Remove trailing filler words but preserve descriptive words
    cleaned = re.sub(r'^\s*(it|this|the\s+answer|the\s+response)\s*', '', cleaned)
    cleaned = re.sub(r'\s+(only|just)$', '', cleaned)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Build the instruction
    if cleaned and len(cleaned) > 3:  # Has meaningful content
        # Preserve the user's full intent
        instruction = cleaned
    elif instruction_parts:
        # Fall back to extracted quality keywords
        instruction = " ".join(instruction_parts)
    else:
        # Last resort: use original query with minimal cleaning
        instruction = re.sub(r'^(make|can\s+make|please\s+make)\s+', '', query_lower)
        instruction = re.sub(r'\s+(it|this)$', '', instruction).strip()
        instruction = instruction if instruction else query_txt
    
    return instruction, part_type, part_number


def get_length_constraint(length: str) -> str:
    """
    Returns length constraint text for LLM prompt.
    
    Args:
        length: Script length ("30s", "90s", "5min")
        
    Returns:
        Constraint text describing the length requirement
    """
    length_constraints = {
        "30s": "approximately 30 seconds when spoken (about 75-90 words total)",
        "90s": "approximately 90 seconds when spoken (about 225-270 words total)",
        "5min": "approximately 5 minutes when spoken (about 750-900 words total)"
    }
    return length_constraints.get(length, "appropriate length")


def get_style_constraint(style: str) -> str:
    """
    Returns style constraint text for LLM prompt.
    
    Args:
        style: Script style ("technical", "plain-english", "press-release")
        
    Returns:
        Constraint text describing the style requirement
    """
    style_constraints = {
        "technical": """Technical style:
- Use domain-specific terminology and jargon
- Include precise definitions and formal language
- Reference technical concepts and academic standards
- Suitable for expert audiences (researchers, engineers, specialists)
- Maintain formal, professional tone""",
        
        "plain-english": """Plain-English style:
- Use simple, everyday language
- Avoid jargon and technical terms (or explain them clearly)
- Use analogies and real-world examples
- Suitable for general audiences and beginners
- Maintain conversational, accessible tone""",
        
        "press-release": """Press-Release style:
- Lead with the most newsworthy information
- Use attention-grabbing headlines and hooks
- Include quotes and human-interest elements
- Emphasize impact, benefits, and real-world applications
- Suitable for media, journalists, and public audiences
- Maintain professional but engaging tone"""
    }
    return style_constraints.get(style, "appropriate style")


def parse_slides_from_answer(answer_text: str) -> Dict[int, str]:
    """
    Parses slides from an answer text and returns a dictionary mapping slide numbers to content.
    
    Args:
        answer_text: The answer text containing slides
        
    Returns:
        Dictionary mapping slide number to slide content (including slide header)
    """
    slides = {}
    
    # Try multiple patterns for slide detection
    slide_patterns = [
        # Pattern 1: "**Slide X:**" format (markdown bold)
        r'(?:^|\n)(\*\*Slide\s+(\d+):.*?)(?=(?:\n\*\*Slide\s+\d+:|$))',
        # Pattern 2: "Slide X:" format
        r'(?:^|\n)(Slide\s+(\d+):.*?)(?=(?:\n(?:Slide\s+\d+:|$)))',
        # Pattern 3: "## Slide X" markdown format
        r'(?:^|\n)(##\s+Slide\s+(\d+).*?)(?=(?:\n##\s+Slide\s+\d+|\n#|$))',
        # Pattern 4: "--- Slide X" format
        r'(?:^|\n)(---\s+Slide\s+(\d+).*?)(?=(?:\n---\s+Slide|\n---$|$))',
    ]
    
    for pattern in slide_patterns:
        matches = list(re.finditer(pattern, answer_text, re.DOTALL | re.IGNORECASE))
        if matches:
            print(f"[DEBUG] Slide parsing: Found {len(matches)} slides using pattern")
            for match in matches:
                slide_content = match.group(1).strip()
                slide_number = int(match.group(2))
                slides[slide_number] = slide_content
                print(f"[DEBUG] Parsed slide {slide_number}, length: {len(slide_content)} chars")
            break  # Use the first pattern that finds slides
    
    if not slides:
        print(f"[WARNING] Could not parse any slides from answer")
        print(f"[DEBUG] Answer preview (first 500 chars): {answer_text[:500]}")
    
    return slides


def reconstruct_answer_from_slides(slides: Dict[int, str], original_answer: str) -> str:
    """
    Reconstructs the full answer from modified slides dictionary.
    
    Args:
        slides: Dictionary mapping slide number to slide content
        original_answer: The original answer text (for format preservation)
        
    Returns:
        Reconstructed answer text
    """
    if not slides:
        return original_answer
    
    # Sort slides by number
    sorted_slide_numbers = sorted(slides.keys())
    
    # Reconstruct with proper spacing
    reconstructed = []
    for slide_num in sorted_slide_numbers:
        slide_content = slides[slide_num]
        reconstructed.append(slide_content)
    
    # Join with double newlines to preserve slide separation
    return "\n\n".join(reconstructed)


def compute_line_diff(old_text: str, new_text: str) -> List[Tuple[str, str]]:
    """
    Computes line-by-line diff between old and new text.
    
    Args:
        old_text: Original text
        new_text: Modified text
        
    Returns:
        List of tuples (status, line) where status is 'removed', 'added', 'unchanged', 'changed'
    """
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    
    # Use difflib to compute differences
    diff = list(difflib.unified_diff(old_lines, new_lines, lineterm='', n=0))
    
    # Parse the unified diff format
    result = []
    i = 0
    old_idx = 0
    new_idx = 0
    
    # Use SequenceMatcher for better line-by-line comparison
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Lines are unchanged
            for line in old_lines[i1:i2]:
                result.append(('unchanged', line))
        elif tag == 'delete':
            # Lines were removed
            for line in old_lines[i1:i2]:
                result.append(('removed', line))
        elif tag == 'insert':
            # Lines were added
            for line in new_lines[j1:j2]:
                result.append(('added', line))
        elif tag == 'replace':
            # Lines were changed
            for line in old_lines[i1:i2]:
                result.append(('removed', line))
            for line in new_lines[j1:j2]:
                result.append(('added', line))
    
    return result


def display_colored_diff(old_text: str, new_text: str, title: str = "DIFF") -> None:
    """
    Displays a color-coded diff in the terminal (GitHub-style).
    
    Args:
        old_text: Original text
        new_text: Modified text
        title: Title for the diff section
    """
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}[{title}]{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 80}{Colors.END}\n")
    
    diff_lines = compute_line_diff(old_text, new_text)
    
    for status, line in diff_lines:
        if status == 'removed':
            print(f"{Colors.BG_RED} - {Colors.END}{Colors.RED}{line}{Colors.END}")
        elif status == 'added':
            print(f"{Colors.BG_GREEN} + {Colors.END}{Colors.GREEN}{line}{Colors.END}")
        elif status == 'unchanged':
            print(f"{Colors.GREY}   {line}{Colors.END}")
        elif status == 'changed':
            print(f"{Colors.YELLOW}{line}{Colors.END}")


def detect_answer_format(previous_answer: str) -> str:
    """
    Detects the format/structure of the previous answer to preserve it in refinement.
    
    Args:
        previous_answer: The previous answer text
        
    Returns:
        A description of the format detected
    """
    answer_lower = previous_answer.lower()
    
    # Detect slide format
    if any(keyword in answer_lower for keyword in ['slide', 'speaker note', 'speaker notes', 'talk', 'presentation']):
        # Check for structured slide format
        if 'slide' in answer_lower and ('speaker note' in answer_lower or 'speaker notes' in answer_lower):
            return "slides_with_speaker_notes"
        elif 'slide' in answer_lower:
            return "slides"
    
    # Detect bullet points
    if previous_answer.count('•') > 2 or previous_answer.count('-') > 5 or previous_answer.count('*') > 5:
        return "bullet_points"
    
    # Detect numbered list
    if re.search(r'^\d+[\.\)]\s', previous_answer, re.MULTILINE):
        return "numbered_list"
    
    # Detect LaTeX/math equations
    if '$' in previous_answer or '\\(' in previous_answer or '\\[' in previous_answer:
        return "with_equations"
    
    # Detect code blocks
    if '```' in previous_answer:
        return "with_code_blocks"
    
    # Default: paragraph format
    return "paragraph"


def parse_refinement_response(response: str, previous_answer: str) -> Dict[str, str]:
    """
    Parses the LLM response to extract refined answer, changes made, and explanation.
    Handles multiple response formats more robustly.
    
    Args:
        response: The raw LLM response
        previous_answer: The previous answer for comparison
        
    Returns:
        Dictionary with 'refined_answer', 'changes_made', and 'why_changed'
    """
    # Try to parse structured response
    refined_answer = ""
    changes_made = ""
    why_changed = ""
    
    # Try multiple parsing strategies
    
    # Strategy 1: Look for explicit section markers (with or without colons)
    patterns = [
        (r'REFINED[_\s]?ANSWER:?\s*\n(.*?)(?=\n\s*CHANGES[_\s]?MADE:|$)', re.DOTALL | re.IGNORECASE),
        (r'CHANGES[_\s]?MADE:?\s*\n(.*?)(?=\n\s*WHY[_\s]?CHANGED:|$)', re.DOTALL | re.IGNORECASE),
        (r'WHY[_\s]?CHANGED:?\s*\n(.*?)$', re.DOTALL | re.IGNORECASE),
    ]
    
    refined_match = re.search(patterns[0][0], response, patterns[0][1])
    changes_match = re.search(patterns[1][0], response, patterns[1][1])
    why_match = re.search(patterns[2][0], response, patterns[2][1])
    
    if refined_match:
        refined_answer = refined_match.group(1).strip()
    if changes_match:
        changes_made = changes_match.group(1).strip()
    if why_match:
        why_changed = why_match.group(1).strip()
    
    # Strategy 2: If Strategy 1 failed, try simpler splitting
    if not refined_answer:
        if "REFINED_ANSWER:" in response or "REFINED ANSWER:" in response:
            parts = re.split(r'REFINED[_\s]?ANSWER:', response, flags=re.IGNORECASE)
            if len(parts) > 1:
                remaining = parts[1]
                if "CHANGES_MADE:" in remaining or "CHANGES MADE:" in remaining:
                    refined_answer = re.split(r'CHANGES[_\s]?MADE:', remaining, flags=re.IGNORECASE)[0].strip()
                    remaining = re.split(r'CHANGES[_\s]?MADE:', remaining, flags=re.IGNORECASE)[1]
                    if "WHY_CHANGED:" in remaining or "WHY CHANGED:" in remaining:
                        changes_made = re.split(r'WHY[_\s]?CHANGED:', remaining, flags=re.IGNORECASE)[0].strip()
                        why_changed = re.split(r'WHY[_\s]?CHANGED:', remaining, flags=re.IGNORECASE)[1].strip()
                    else:
                        changes_made = remaining.strip()
                else:
                    refined_answer = remaining.strip()
    
    # Strategy 3: Fallback - use entire response as refined answer
    if not refined_answer or len(refined_answer) < 10:
        # Try to find slide content in the response
        slide_patterns = [
            r'(Slide\s+\d+:.*?)(?=\n\n|$)',
            r'(##\s+Slide\s+\d+.*?)(?=\n\n|$)',
        ]
        
        for pattern in slide_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                refined_answer = "\n\n".join(matches)
                break
        
        if not refined_answer:
            # Last resort: use full response
            refined_answer = response.strip()
        
        # Generate basic change info
        if not changes_made:
            changes_made = "LLM response did not include explicit change documentation. See comparison below."
        if not why_changed:
            why_changed = "Answer was refined based on user's request."
    
    # Ensure we have something for each field
    if not changes_made:
        changes_made = "Compare the previous and refined answers to see the specific changes."
    if not why_changed:
        why_changed = "Changes were made to address the user's refinement request."
    
    return {
        "refined_answer": refined_answer,
        "changes_made": changes_made,
        "why_changed": why_changed
    }


def build_slide_specific_refinement_prompt(
    original_query: str,
    previous_answer: str,
    refinement_instruction: str,
    context: str,
    slide_number: int,
    slide_content: str,
    user_query: str,
    script_length: Optional[str] = None,
    script_style: Optional[str] = None
) -> str:
    """
    Builds a prompt for refining a SPECIFIC slide only.
    
    Args:
        original_query: The original user query
        previous_answer: The full previous answer
        refinement_instruction: The specific refinement instruction
        context: The relevant context from the vector store
        slide_number: The specific slide number to refine
        slide_content: The content of the specific slide
        user_query: The user's refinement query (for exact parsing)
        
    Returns:
        A formatted prompt string for the LLM
    """
    # Build length/style constraints
    length_constraint_text = ""
    style_constraint_text = ""
    
    if script_length:
        length_constraint_text = f"\nLENGTH REQUIREMENT: The slide content should be {get_length_constraint(script_length)}"
    
    if script_style:
        style_constraint_text = f"\n\nSTYLE REQUIREMENT:\n{get_style_constraint(script_style)}"
    
    SLIDE_REFINEMENT_PROMPT = """You are a helpful assistant that refines SPECIFIC slides based on user feedback.

IMPORTANT: You MUST refine ONLY Slide {slide_number}. Do NOT modify any other slides.

Context from documents (RAG-retrieved):
{context}

---

Original Question: {original_query}

FULL PREVIOUS ANSWER (for context only):
{previous_answer}

---

SLIDE TO REFINE (Slide {slide_number}):
{slide_content}

---

User's Refinement Request: {refinement_instruction}

USER'S EXACT REQUEST (parse this carefully):
"{user_query}"

REFINED INSTRUCTION (extracted):
{refinement_instruction}
{length_constraint_text}
{style_constraint_text}

CRITICAL REQUIREMENTS FOR REFINEMENT:
1. You MUST refine ONLY Slide {slide_number} - do NOT output any other slides
2. Parse the USER'S EXACT REQUEST above and apply ALL aspects of their intent
3. The instruction may contain MULTIPLE qualities (e.g., "longer, elaborative, creative and interesting")
   - You MUST address EVERY quality mentioned, not just the first one
   - Example: if they say "longer, creative and interesting", make it longer AND creative AND interesting
4. If they specify a number (e.g., "keep to two speaker notes"), you MUST have EXACTLY that number
5. Common refinement types:
   - "longer/elaborate": Make points more detailed and comprehensive
   - "shorter/concise": Make points brief and to the point
   - "simpler": Use plain language, avoid jargon
   - "technical": Use domain-specific terminology
   - "creative": Add novel perspectives, analogies, or interesting angles
   - "interesting": Make engaging with examples, stories, or hooks
   - "clearer": Improve clarity and remove ambiguity
6. Maintain the EXACT format of the slide (keep slide header, bullets, speaker notes if present)
7. Preserve any LaTeX equations in their original format
8. Use ONLY the context provided above for factual accuracy
9. Output ONLY the refined Slide {slide_number}, not the entire presentation

Please provide your refined slide in the following format:

REFINED_ANSWER:
[Your refined version of Slide {slide_number} here, maintaining the exact same format]

CHANGES_MADE:
[List the specific changes you made to Slide {slide_number}. Be explicit about what was changed, added, or removed.]

WHY_CHANGED:
[Explain why these changes were made and how they address the user's refinement request: "{refinement_instruction}"]

Provide your response in the exact format above with the three sections clearly labeled."""
    
    prompt = SLIDE_REFINEMENT_PROMPT.format(
        context=context,
        original_query=original_query,
        previous_answer=previous_answer,
        refinement_instruction=refinement_instruction,
        slide_number=slide_number,
        slide_content=slide_content,
        user_query=user_query,
        length_constraint_text=length_constraint_text,
        style_constraint_text=style_constraint_text
    )
    
    return prompt


def build_refinement_prompt(
    original_query: str,
    previous_answer: str,
    refinement_instruction: str,
    context: str,
    slide_number: Optional[int] = None,
    script_length: Optional[str] = None,
    script_style: Optional[str] = None
) -> str:
    """
    Builds a comprehensive prompt for query refinement that integrates:
    - The original query
    - The previous answer
    - The refinement instruction
    - The relevant context
    
    Args:
        original_query: The original user query
        previous_answer: The previous answer that needs to be refined
        refinement_instruction: The specific refinement instruction (e.g., "more technical", "shorter")
        context: The relevant context from the vector store
        slide_number: Optional slide number if refining a specific slide
        
    Returns:
        A formatted prompt string for the LLM
    """
    # Detect the format of the previous answer to preserve it
    answer_format = detect_answer_format(previous_answer)
    format_instruction = get_format_preservation_instruction(answer_format, previous_answer)
    
    # Add slide-specific instruction if applicable
    slide_specific_note = ""
    if slide_number is not None:
        slide_specific_note = f"""
CRITICAL: The user has requested changes to SLIDE {slide_number} ONLY.
You MUST:
1. Keep all other slides EXACTLY as they are
2. ONLY modify Slide {slide_number}
3. Maintain the exact format and structure of all slides
4. Apply the refinement instruction ONLY to Slide {slide_number}
"""
    
    # Build length/style constraints
    length_constraint_text = ""
    style_constraint_text = ""
    
    if script_length:
        length_constraint_text = f"\nLENGTH REQUIREMENT: The response should be {get_length_constraint(script_length)}"
    
    if script_style:
        style_constraint_text = f"\n\nSTYLE REQUIREMENT:\n{get_style_constraint(script_style)}"
    
    REFINEMENT_PROMPT_TEMPLATE = """You are a helpful assistant that refines answers based on user feedback.

IMPORTANT: You MUST base your refined answer ONLY on the context provided below. Do not use any information outside of this context.
{slide_specific_note}

Context from documents (RAG-retrieved):
{context}

---

Original Question: {original_query}

Previous Answer (NOTE: PRESERVE THIS EXACT FORMAT STRUCTURE):
{previous_answer}

---

User's Refinement Request: {refinement_instruction}
{length_constraint_text}
{style_constraint_text}

{format_instruction}

CRITICAL REQUIREMENTS FOR REFINEMENT:
1. You MUST preserve the exact format and structure of the previous answer
2. If the previous answer was in slides format, maintain the slide structure (Slide 1, Slide 2, etc.)
3. If the previous answer had speaker notes, maintain the speaker notes format
4. If the previous answer had equations in LaTeX, preserve the LaTeX formatting
5. If the previous answer had bullet points, maintain bullet point structure
6. The refinement instruction may contain MULTIPLE qualities (e.g., "shorter and more technical")
   - Apply ALL aspects mentioned: "{refinement_instruction}"
   - Don't ignore any part of the instruction
7. Only modify the content (make it less technical, more concise, etc.) but keep the structure identical
8. OUTPUT ALL SLIDES IN THE ANSWER, not just the modified one

Please provide a refined answer in the following format:

REFINED_ANSWER:
[Your refined answer here that:
1. Addresses the original question: "{original_query}"
2. Incorporates the refinement instruction: "{refinement_instruction}"
3. Uses ONLY the context provided above for accuracy (RAG-based)
4. Maintains factual correctness while applying the requested modification
5. Does NOT include information not found in the context
6. Maintains the EXACT same format/structure as the previous answer
7. Preserves all formatting elements (slides, speaker notes, equations, bullets, etc.)
8. Includes ALL slides, not just the modified ones]

CHANGES_MADE:
[Explain what specific changes were made compared to the previous answer. List the key differences, additions, removals, or modifications. Reference specific parts of the context that support your changes.]

WHY_CHANGED:
[Explain why these changes were made - how they address the user's refinement request and improve the answer according to their instruction, while staying grounded in the provided context.]

Provide your response in the exact format above with the three sections clearly labeled."""
    
    prompt_template = ChatPromptTemplate.from_template(REFINEMENT_PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context,
        original_query=original_query,
        previous_answer=previous_answer,
        refinement_instruction=refinement_instruction,
        format_instruction=format_instruction,
        slide_specific_note=slide_specific_note,
        length_constraint_text=length_constraint_text,
        style_constraint_text=style_constraint_text
    )
    
    return prompt


def display_slide_by_slide_delta(
    previous_answer: str, 
    refined_answer: str, 
    part_type: Optional[str] = None, 
    part_number: Optional[int] = None,
    changes_made: str = "",
    why_changed: str = ""
) -> None:
    """
    Displays a granular slide-by-slide comparison of changes with GitHub-style diff colors.
    
    Args:
        previous_answer: The previous answer text
        refined_answer: The refined answer text
        part_type: The type of part being refined (e.g., "slide")
        part_number: The number of the specific part being refined
        changes_made: Description of changes made
        why_changed: Explanation of why changes were made
    """
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}[SLIDE-BY-SLIDE COMPARISON - GitHub Style]{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 80}{Colors.END}")
    
    # Parse slides from both answers
    prev_slides = parse_slides_from_answer(previous_answer)
    refined_slides = parse_slides_from_answer(refined_answer)
    
    if not prev_slides and not refined_slides:
        # No slides detected, show regular diff
        print(f"\n{Colors.YELLOW}[INFO] No slide structure detected. Showing full comparison.{Colors.END}")
        display_colored_diff(previous_answer, refined_answer, "FULL TEXT DIFF")
        return
    
    # Get all slide numbers from both versions
    all_slide_nums = sorted(set(list(prev_slides.keys()) + list(refined_slides.keys())))
    
    if not all_slide_nums:
        print(f"\n{Colors.YELLOW}[INFO] No slides could be parsed. Showing full comparison.{Colors.END}")
        return
    
    # Display each slide comparison
    for slide_num in all_slide_nums:
        prev_slide = prev_slides.get(slide_num, "")
        refined_slide = refined_slides.get(slide_num, "")
        
        # Check if this slide was modified
        is_modified = prev_slide != refined_slide
        is_target_slide = (part_number is not None and slide_num == part_number)
        
        if is_modified:
            # Modified slide header
            if is_target_slide:
                print(f"\n{Colors.BG_YELLOW}{Colors.BOLD} 🎯 SLIDE {slide_num} - MODIFIED (TARGET) 🎯 {Colors.END}")
            else:
                print(f"\n{Colors.BG_YELLOW}{Colors.BOLD} ⚠️  SLIDE {slide_num} - MODIFIED ⚠️  {Colors.END}")
            
            # Display GitHub-style diff
            display_colored_diff(
                prev_slide if prev_slide else "[Slide not present]",
                refined_slide if refined_slide else "[Slide removed]",
                f"SLIDE {slide_num} DIFF"
            )
        else:
            # Unchanged slide
            print(f"\n{Colors.GREY}✅ Slide {slide_num}: UNCHANGED{Colors.END}")
    
    # Summary box
    modified_count = sum(1 for num in all_slide_nums if prev_slides.get(num, "") != refined_slides.get(num, ""))
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}[SUMMARY]{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 80}{Colors.END}")
    print(f"{Colors.GREEN}✓ Modified: {modified_count} slide(s){Colors.END}")
    print(f"{Colors.GREY}✓ Unchanged: {len(all_slide_nums) - modified_count} slide(s){Colors.END}")
    print(f"{Colors.BOLD}✓ Total: {len(all_slide_nums)} slide(s){Colors.END}")
    
    if part_number is not None:
        if part_number in prev_slides:
            print(f"{Colors.YELLOW}🎯 Target: {part_type} {part_number} modification requested{Colors.END}")
        else:
            print(f"{Colors.RED}⚠️  Warning: Requested {part_type} {part_number} not found in original answer{Colors.END}")
    
    # Display "Why Changed" box
    if why_changed:
        print(f"\n{Colors.BLUE}{Colors.BOLD}╔{'═' * 78}╗{Colors.END}")
        print(f"{Colors.BLUE}{Colors.BOLD}║{Colors.END} {Colors.BOLD}WHY IT CHANGED{Colors.END}{' ' * 63}{Colors.BLUE}{Colors.BOLD}║{Colors.END}")
        print(f"{Colors.BLUE}{Colors.BOLD}╠{'═' * 78}╣{Colors.END}")
        
        # Word wrap the explanation
        import textwrap
        wrapped_lines = textwrap.wrap(why_changed, width=76)
        for line in wrapped_lines:
            print(f"{Colors.BLUE}{Colors.BOLD}║{Colors.END} {line:<76} {Colors.BLUE}{Colors.BOLD}║{Colors.END}")
        
        print(f"{Colors.BLUE}{Colors.BOLD}╚{'═' * 78}╝{Colors.END}")
    
    # Display "Changes Made" box
    if changes_made:
        print(f"\n{Colors.GREEN}{Colors.BOLD}╔{'═' * 78}╗{Colors.END}")
        print(f"{Colors.GREEN}{Colors.BOLD}║{Colors.END} {Colors.BOLD}CHANGES MADE{Colors.END}{' ' * 64}{Colors.GREEN}{Colors.BOLD}║{Colors.END}")
        print(f"{Colors.GREEN}{Colors.BOLD}╠{'═' * 78}╣{Colors.END}")
        
        # Word wrap the changes
        import textwrap
        wrapped_lines = textwrap.wrap(changes_made, width=76)
        for line in wrapped_lines:
            print(f"{Colors.GREEN}{Colors.BOLD}║{Colors.END} {line:<76} {Colors.GREEN}{Colors.BOLD}║{Colors.END}")
        
        print(f"{Colors.GREEN}{Colors.BOLD}╚{'═' * 78}╝{Colors.END}")
    
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 80}{Colors.END}\n")


def get_format_preservation_instruction(format_type: str, previous_answer: str) -> str:
    """
    Generates specific format preservation instructions based on detected format.
    
    Args:
        format_type: The detected format type
        previous_answer: The previous answer for reference
        
    Returns:
        Format-specific preservation instructions
    """
    if format_type == "slides_with_speaker_notes":
        return """FORMAT PRESERVATION INSTRUCTIONS:
- The previous answer is in SLIDE PRESENTATION format with SPEAKER NOTES
- You MUST maintain the slide structure (Slide 1, Slide 2, etc.)
- You MUST preserve the speaker notes format (Speaker Notes: or similar markers)
- Keep the same number of slides or adjust only if the refinement requires it
- Preserve any LaTeX equations in their original format
- Maintain the slide-level bullets and speaker note structure
- Only modify the content to apply the refinement instruction, NOT the format structure"""
    
    elif format_type == "slides":
        return """FORMAT PRESERVATION INSTRUCTIONS:
- The previous answer is in SLIDE PRESENTATION format
- You MUST maintain the slide structure (Slide 1, Slide 2, etc.)
- Keep the same number of slides or adjust only if the refinement requires it
- Preserve any LaTeX equations in their original format
- Maintain the slide-level bullets structure
- Only modify the content to apply the refinement instruction, NOT the format structure"""
    
    elif format_type == "bullet_points":
        return """FORMAT PRESERVATION INSTRUCTIONS:
- The previous answer uses BULLET POINTS
- You MUST maintain the bullet point structure
- Keep the same hierarchical structure (main bullets, sub-bullets)
- Only modify the content to apply the refinement instruction, NOT the bullet format"""
    
    elif format_type == "numbered_list":
        return """FORMAT PRESERVATION INSTRUCTIONS:
- The previous answer uses a NUMBERED LIST
- You MUST maintain the numbered list structure
- Keep the same numbering format
- Only modify the content to apply the refinement instruction, NOT the numbering format"""
    
    elif format_type == "with_equations":
        return """FORMAT PRESERVATION INSTRUCTIONS:
- The previous answer contains MATHEMATICAL EQUATIONS in LaTeX format
- You MUST preserve all LaTeX equations exactly as they appear (using $, \\(, \\[, etc.)
- Do not convert equations to plain text
- Maintain equation formatting and structure
- Only modify the surrounding text to apply the refinement instruction"""
    
    elif format_type == "with_code_blocks":
        return """FORMAT PRESERVATION INSTRUCTIONS:
- The previous answer contains CODE BLOCKS
- You MUST preserve code blocks with proper formatting (``` markers)
- Maintain code syntax and structure
- Only modify comments or explanations, NOT the code itself"""
    
    else:
        return """FORMAT PRESERVATION INSTRUCTIONS:
- Maintain the same overall structure and formatting style as the previous answer
- Preserve any special formatting elements (headings, sections, etc.)
- Only modify the content to apply the refinement instruction"""


def query_rag(query_txt: str, model_name: str, session_id: str = "default", uploaded_filenames: List[str] = None):
    # Load conversation history from persistent storage
    global conversation_history
    conversation_history = load_conversation_history()
    
    # Load reference script for ROUGE/BERTScore evaluation (if available)
    reference_script = None
    if uploaded_filenames:
        reference_script = get_reference_script_for_query(uploaded_filenames)
        if reference_script:
            print(f"[INFO] 📊 Reference script loaded for quality evaluation (ROUGE/BERTScore)")
        else:
            print(f"[INFO] No reference script available. Skipping ROUGE/BERTScore.")
    
    # Debug: Print session history status
    if session_id in conversation_history:
        print(f"[DEBUG] Session '{session_id}' found in history")
        print(f"[DEBUG] Has previous_answer: {bool(conversation_history[session_id].get('previous_answer'))}")
    else:
        print(f"[DEBUG] Session '{session_id}' NOT found in history")
        print(f"[DEBUG] Available sessions: {list(conversation_history.keys())}")

    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding(),
        persist_directory=CHROMA_PATH
    )

    # Check if this is a refinement request
    is_refinement = detect_refinement_request(query_txt, session_id, model_name)
    print(f"[DEBUG] Is refinement request: {is_refinement}")
    
    if is_refinement:
        # Handle refinement request
        if session_id not in conversation_history:
            print("[WARNING] No previous conversation found for refinement. Treating as new query.")
            is_refinement = False
        else:
            history = conversation_history[session_id]
            original_query = history.get("original_query", "")
            previous_answer = history.get("previous_answer", "")
            
            if not original_query or not previous_answer:
                print("[WARNING] Incomplete conversation history. Treating as new query.")
                is_refinement = False
            else:
                print(f"[INFO] Detected refinement request. Original query: {original_query}")
                refinement_instruction, part_type, part_number = extract_refinement_instruction(query_txt)
                print(f"[INFO] Refinement instruction: {refinement_instruction}")
                
                # Extract length/style from refinement query (may override previous)
                new_length = extract_script_length(query_txt)
                new_style = extract_script_style(query_txt)
                
                # Get current length/style from history
                current_length = history.get("script_length")
                current_style = history.get("script_style")
                
                # Use new if specified, otherwise keep current
                script_length = new_length if new_length else current_length
                script_style = new_style if new_style else current_style
                
                # Log changes
                if new_length and new_length != current_length:
                    print(f"[INFO] 📏 Length changed: {current_length} → {new_length}")
                if new_style and new_style != current_style:
                    print(f"[INFO] 🎨 Style changed: {current_style} → {new_style}")
                
                if script_length:
                    print(f"[INFO] Current script length: {script_length}")
                if script_style:
                    print(f"[INFO] Current script style: {script_style}")
                
                # Check if this is a slide-specific refinement
                is_slide_specific = part_type is not None and part_number is not None
                if is_slide_specific:
                    print(f"[INFO] Slide-specific refinement detected: {part_type} {part_number}")
                
                # Detect and log format preservation
                detected_format = detect_answer_format(previous_answer)
                format_names = {
                    "slides_with_speaker_notes": "Slides with Speaker Notes",
                    "slides": "Slides",
                    "bullet_points": "Bullet Points",
                    "numbered_list": "Numbered List",
                    "with_equations": "With Equations",
                    "with_code_blocks": "With Code Blocks",
                    "paragraph": "Paragraph"
                }
                print(f"[INFO] Detected answer format: {format_names.get(detected_format, detected_format)} - will preserve this format")
                
                # Retrieve relevant context (use original query for better retrieval)
                # Use math-aware retrieval for better equation matching
                results = math_aware_retrieval(db, original_query, k=5)
                
                # Validate that we retrieved context
                if not results or len(results) == 0:
                    print("\n[ERROR] No documents retrieved from vector store. Cannot perform RAG refinement.")
                    print("[ERROR] The vector database appears to be empty or no relevant documents found.")
                    print("[SOLUTION] Please ensure:")
                    print("  1. PDF files exist in the ./pdfs directory")
                    print("  2. Documents are being loaded and indexed (check the INFO messages above)")
                    print("  3. Try running with --reset flag to force re-indexing")
                    print("\n[INFO] Refinement requires indexed documents to work correctly.")
                    return None, [], []
                
                context_txt = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
                
                # Validate context is not empty
                if not context_txt or not context_txt.strip():
                    print("\n[ERROR] Retrieved context is empty. Cannot perform RAG refinement.")
                    print("[ERROR] Please ensure documents are properly indexed.")
                    return None, [], []
                
                print(f"[INFO] Retrieved {len(results)} document chunks from vector store for RAG refinement.")
                
                # Initialize reconstruction flag
                needs_reconstruction = False
                original_slides = None
                
                # Handle slide-specific refinement differently
                if is_slide_specific and (detected_format == "slides" or detected_format == "slides_with_speaker_notes"):
                    # Parse slides from previous answer
                    slides = parse_slides_from_answer(previous_answer)
                    
                    if part_number in slides:
                        print(f"[INFO] Using slide-specific refinement for {part_type} {part_number}")
                        slide_content = slides[part_number]
                        
                        # Build slide-specific refinement prompt
                        prompt = build_slide_specific_refinement_prompt(
                            original_query=original_query,
                            previous_answer=previous_answer,
                            refinement_instruction=refinement_instruction,
                            context=context_txt,
                            slide_number=part_number,
                            slide_content=slide_content,
                            user_query=query_txt,
                            script_length=script_length,
                            script_style=script_style
                        )
                        
                        # Mark that we need to reconstruct after refinement
                        needs_reconstruction = True
                        original_slides = slides.copy()
                    else:
                        print(f"[WARNING] Slide {part_number} not found in previous answer. Using general refinement.")
                        print(f"[DEBUG] Available slides: {list(slides.keys())}")
                        # Fallback to general refinement with slide number hint
                        prompt = build_refinement_prompt(
                            original_query=original_query,
                            previous_answer=previous_answer,
                            refinement_instruction=refinement_instruction,
                            context=context_txt,
                            slide_number=part_number,
                            script_length=script_length,
                            script_style=script_style
                        )
                        needs_reconstruction = False  # Can't reconstruct if slide not found
                else:
                    # Build general refinement prompt
                    prompt = build_refinement_prompt(
                        original_query=original_query,
                        previous_answer=previous_answer,
                        refinement_instruction=refinement_instruction,
                        context=context_txt,
                        slide_number=part_number if is_slide_specific else None,
                        script_length=script_length,
                        script_style=script_style
                    )
                    needs_reconstruction = False  # General refinement doesn't need reconstruction
                
                model = OllamaLLM(model=model_name)
                response_txt = model.invoke(prompt)
                
                # Parse the response to extract refined answer, changes, and explanation
                parsed_response = parse_refinement_response(response_txt, previous_answer)
                refined_answer = parsed_response["refined_answer"]
                changes_made = parsed_response["changes_made"]
                why_changed = parsed_response["why_changed"]
                
                # CRITICAL: Reconstruct full answer if this was a slide-specific refinement
                if needs_reconstruction and original_slides is not None:
                    print(f"[INFO] Reconstructing full answer with modified {part_type} {part_number}")
                    
                    # Parse the refined slide from the LLM response
                    refined_slide_content = refined_answer.strip()
                    
                    # Validate: Check if LLM returned only the target slide or all slides
                    refined_slides_check = parse_slides_from_answer(refined_slide_content)
                    
                    if len(refined_slides_check) > 1:
                        # LLM returned multiple slides - extract only the target one
                        print(f"[WARNING] LLM returned {len(refined_slides_check)} slides instead of 1. Extracting only slide {part_number}")
                        if part_number in refined_slides_check:
                            refined_slide_content = refined_slides_check[part_number]
                        else:
                            print(f"[ERROR] Target slide {part_number} not found in LLM response. Using full response.")
                    elif len(refined_slides_check) == 1:
                        # LLM correctly returned only one slide
                        target_slide_num = list(refined_slides_check.keys())[0]
                        if target_slide_num != part_number:
                            print(f"[WARNING] LLM returned slide {target_slide_num} instead of {part_number}. Adjusting...")
                        refined_slide_content = list(refined_slides_check.values())[0]
                    else:
                        # Couldn't parse as slide format, assume it's just the content
                        print(f"[INFO] Using raw LLM response as refined slide content")
                    
                    # Update only the target slide in the original slides dictionary
                    original_slides[part_number] = refined_slide_content
                    
                    # Reconstruct the full answer with all slides
                    refined_answer = reconstruct_answer_from_slides(original_slides, previous_answer)
                    
                    print(f"[INFO] ✅ Reconstructed complete answer: {len(original_slides)} total slides")
                    print(f"[INFO] ✅ Modified: {part_type} {part_number}")
                    print(f"[INFO] ✅ Unchanged: {len(original_slides) - 1} slides")
                
                # Display the results with delta comparison
                print("\n" + "="*80)
                print("[REFINED ANSWER]")
                print("="*80)
                print(refined_answer)
                
                # Enhanced delta display
                if is_slide_specific and (detected_format == "slides" or detected_format == "slides_with_speaker_notes"):
                    # Use slide-by-slide comparison for slide-specific refinements
                    display_slide_by_slide_delta(previous_answer, refined_answer, part_type, part_number, changes_made, why_changed)
                else:
                    # Standard delta display for non-slide refinements
                    print("\n" + "="*80)
                    print("[DELTA: OLD vs NEW]")
                    print("="*80)
                    print("\n[PREVIOUS ANSWER]")
                    print("-" * 80)
                    print(previous_answer)
                    print("\n[REFINED ANSWER]")
                    print("-" * 80)
                    print(refined_answer)
                
                print("\n" + "="*80)
                print("[CHANGES MADE]")
                print("="*80)
                print(changes_made)
                print("\n" + "="*80)
                print("[WHY IT CHANGED]")
                print("="*80)
                print(why_changed)
                print("="*80 + "\n")
                
                # Display length/style info if present
                if script_length or script_style:
                    print(f"\n{Colors.CYAN}{'=' * 80}{Colors.END}")
                    print(f"{Colors.CYAN}{Colors.BOLD}[CURRENT SCRIPT PARAMETERS]{Colors.END}")
                    print(f"{Colors.CYAN}{'=' * 80}{Colors.END}")
                    if script_length:
                        print(f"{Colors.GREEN}📏 Length: {script_length}{Colors.END}")
                    if script_style:
                        print(f"{Colors.BLUE}🎨 Style: {script_style}{Colors.END}")
                    print(f"{Colors.CYAN}{'=' * 80}{Colors.END}\n")
                
                # Update conversation history with new length/style
                conversation_history[session_id] = {
                    "original_query": original_query,
                    "previous_answer": refined_answer,
                    "refinement_count": history.get("refinement_count", 0) + 1,
                    "script_length": script_length,
                    "script_style": script_style
                }
                
                # Save conversation history to persistent storage
                save_conversation_history(conversation_history)
                
                sources = [doc.metadata.get("id", None) for doc, _ in results]
                print(f"[SOURCES] {sources}")
                
                # Return response, source IDs, and full retrieved chunks for feedback UI
                retrieved_chunks = [{
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                } for doc, score in results]
                
                # AUTOMATIC EVALUATION (Terminal Only)
                if EVALUATION_AVAILABLE:
                    try:
                        evaluation = evaluate_response(response_txt, retrieved_chunks, reference_text=reference_script)
                        print_evaluation_report(evaluation, show_detailed=False)
                    except Exception as e:
                        print(f"[WARNING] Evaluation failed: {e}")
                
                return response_txt, sources, retrieved_chunks
    
    # Standard query handling
    if not is_refinement:
        # Extract length and style from query
        script_length = extract_script_length(query_txt)
        script_style = extract_script_style(query_txt)
        
        # Build length/style constraints
        length_constraint = ""
        style_constraint = ""
        
        if script_length:
            length_constraint = f"\nLENGTH REQUIREMENT: The response should be {get_length_constraint(script_length)}"
            print(f"[INFO] Script length constraint: {script_length}")
        
        if script_style:
            style_constraint = f"\n\nSTYLE REQUIREMENT:\n{get_style_constraint(script_style)}"
            print(f"[INFO] Script style constraint: {script_style}")
        
        PROMPT_TEMPLATE = """
IMPORTANT: You MUST answer the question based ONLY on the context provided below. Do not use any information outside of this context. If the context does not contain enough information, say so explicitly.

Context from documents (RAG-retrieved):
{context}

---

Question: {question}
{length_constraint}
{style_constraint}

Instructions:
- Answer the question in detail based ONLY on the context provided above
- Do not make up information not found in the context
- If the context doesn't contain enough information, state that clearly
- Provide a comprehensive answer using the retrieved context
- STRICTLY follow any length and style requirements specified above

Answer:"""

        # retrieve most relevant chunks to our question
        # Use math-aware retrieval for better equation matching
        results = math_aware_retrieval(db, query_txt, k=5)
        
        # Validate that we retrieved context
        if not results or len(results) == 0:
            print("\n[ERROR] No documents retrieved from vector store. Cannot perform RAG.")
            print("[ERROR] The vector database appears to be empty.")
            print("[SOLUTION] Please ensure:")
            print("  1. PDF files exist in the ./pdfs directory")
            print("  2. Documents are being loaded and indexed (check the INFO messages above)")
            print("  3. If documents were already indexed, they may have been cleared")
            print("  4. Try running with --reset flag to force re-indexing")
            print("\n[INFO] The system requires indexed documents to perform RAG-based answering.")
            return None, [], []
        
        context_txt = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        
        # Validate context is not empty
        if not context_txt or not context_txt.strip():
            print("\n[ERROR] Retrieved context is empty. Cannot perform RAG.")
            print("[ERROR] Please ensure documents are properly indexed.")
            return None, [], []
        
        print(f"[INFO] Retrieved {len(results)} document chunks from vector store for RAG.")
        
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            context=context_txt, 
            question=query_txt,
            length_constraint=length_constraint,
            style_constraint=style_constraint
        )
        # print(prompt)

        model = OllamaLLM(model=model_name)
        response_txt = model.invoke(prompt)
        
        # Display length/style info if present
        if script_length or script_style:
            print(f"\n{Colors.CYAN}{'=' * 80}{Colors.END}")
            print(f"{Colors.CYAN}{Colors.BOLD}[SCRIPT PARAMETERS]{Colors.END}")
            print(f"{Colors.CYAN}{'=' * 80}{Colors.END}")
            if script_length:
                print(f"{Colors.GREEN}📏 Length: {script_length}{Colors.END}")
            if script_style:
                print(f"{Colors.BLUE}🎨 Style: {script_style}{Colors.END}")
            print(f"{Colors.CYAN}{'=' * 80}{Colors.END}\n")
        
        print(f"[RESULT] {response_txt}")

        # Store conversation history with length/style
        conversation_history[session_id] = {
            "original_query": query_txt,
            "previous_answer": response_txt,
            "refinement_count": 0,
            "script_length": script_length,
            "script_style": script_style
        }
        
        # Save conversation history to persistent storage
        save_conversation_history(conversation_history)

        sources = [doc.metadata.get("id", None) for doc, _ in results]
        print(f"[SOURCES] {sources}")
        
        # Return response, source IDs, and full retrieved chunks for feedback UI
        retrieved_chunks = [{
            'content': doc.page_content,
            'metadata': doc.metadata,
            'score': score
        } for doc, score in results]
        
        # AUTOMATIC EVALUATION (Terminal Only)
        if EVALUATION_AVAILABLE:
            try:
                evaluation = evaluate_response(response_txt, retrieved_chunks, reference_text=reference_script)
                print_evaluation_report(evaluation, show_detailed=False)
            except Exception as e:
                print(f"[WARNING] Evaluation failed: {e}")
        
        return response_txt, sources, retrieved_chunks


def reset_collection():
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding(),
        persist_directory=CHROMA_PATH
    )
    # before_existing = db.get(include=[])
    # print(f"[DEBUG] Document IDs (Before Reset): {before_existing['ids']}")

    db.delete_collection()
    print(f"[INFO] Chroma Collection '{COLLECTION_NAME}' has been reset.")

    # db = Chroma(
    #     collection_name=COLLECTION_NAME,
    #     embedding_function=get_embedding(),
    #     persist_directory=CHROMA_PATH
    # )

    # after_existing = db.get(include=[])
    # print(f"[DEBUG] Document IDs (After Reset): {after_existing['ids']}")


def main():
    # defining arguments
    parser = argparse.ArgumentParser(description="Local RAG Pdf(s) QnA")
    parser.add_argument("--query", type=str, required=True, help="Query you want to ask")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Ollama Model ID for Response Generation")
    parser.add_argument("--reset", action="store_true", help="Reset Chroma Collection before indexing")
    parser.add_argument("--session-id", type=str, default="default", help="Session ID for conversation history (for refinement requests)")
    parser.add_argument("--clear-history", action="store_true", help="Clear conversation history before processing query")
    # store_true: if argument is present, store true else false

    args = parser.parse_args()

    if args.clear_history:
        clear_conversation_history()

    if args.reset:
        reset_collection()

    documents = load_documents()
    
    if not documents or len(documents) == 0:
        print("[ERROR] No documents loaded from PDF directory.")
        print(f"[ERROR] Please ensure PDF files exist in: {DATA_PATH}")
        print("[ERROR] Cannot proceed without documents to index.")
        return
    
    print(f"[INFO] Loaded {len(documents)} document(s) from PDF directory.")

    chunks = split_documents(documents)
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    if not chunks_with_ids or len(chunks_with_ids) == 0:
        print("[ERROR] No chunks created from documents.")
        print("[ERROR] Cannot proceed without document chunks.")
        return
    
    print(f"[INFO] Created {len(chunks_with_ids)} document chunks.")

    # print("[DEBUG] All chunk IDs and sources:")
    # for chunk in chunks_with_ids:
    #     print(f"  - {chunk.metadata['id']} | {chunk.metadata['source']} | len: {len(chunk.page_content)}")


    add_to_chroma(chunks_with_ids)

    # query_txt = """
    #     Are there any clauses which defines that I cannot open another work for myself
    #     while working at TCS Research.
    # """
    query_rag(query_txt=args.query, model_name=args.model, session_id=args.session_id)
    # pprint.pp(model_response)


# ensures that script executes only when run directly
if __name__ == "__main__":
    main()
