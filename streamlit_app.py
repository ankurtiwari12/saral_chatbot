import streamlit as st
import os
import json
import re
import tempfile
import shutil
from playground import (
    load_documents,
    load_documents_from_files,
    split_documents,
    calculate_chunk_ids,
    add_to_chroma,
    query_rag,
    reset_collection,
    clear_conversation_history,
    load_conversation_history,
    extract_refinement_instruction,
    store_bullet_feedback,
    get_feedback_statistics,
    clear_feedback_data,
    extract_equations,
    SYMPY_AVAILABLE,
    HISTORY_FILE,
    DATA_PATH,
    CHROMA_PATH
)

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        transform: scale(1.02);
    }
    .query-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    .result-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    .info-box {
        background: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background: #667eea;
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background: #f1f3f5;
        color: #333;
        margin-right: 20%;
    }
    .bullet-container {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .bullet-text {
        font-size: 1.05rem;
        color: #333;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    .source-highlight {
        background: #fff9c4;
        padding: 0.75rem;
        border-radius: 6px;
        border-left: 3px solid #fbc02d;
        font-size: 0.9rem;
        color: #555;
        margin-top: 0.5rem;
        font-family: monospace;
    }
    .feedback-buttons {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    .accepted-bullet {
        border-left: 4px solid #4caf50;
        background: #f1f8f4;
    }
    .rejected-bullet {
        border-left: 4px solid #f44336;
        background: #fef2f2;
        opacity: 0.7;
    }
    .math-badge {
        display: inline-block;
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .feedback-stats {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = "default"
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = {}
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'model_name' not in st.session_state:
    st.session_state.model_name = "llama3.2:3b"
if 'show_full_history' not in st.session_state:
    st.session_state.show_full_history = False
if 'uploaded_files_info' not in st.session_state:
    st.session_state.uploaded_files_info = []
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None
if 'bullet_feedback' not in st.session_state:
    st.session_state.bullet_feedback = {}  # {bullet_id: 'accepted'/'rejected'}
if 'last_retrieved_chunks' not in st.session_state:
    st.session_state.last_retrieved_chunks = []
if 'show_feedback_ui' not in st.session_state:
    st.session_state.show_feedback_ui = False
if 'pending_feedback_refinement' not in st.session_state:
    st.session_state.pending_feedback_refinement = None
if 'auto_submit_query' not in st.session_state:
    st.session_state.auto_submit_query = None

def load_and_index_documents():
    """Load and index documents from pdfs/ folder (legacy)"""
    try:
        with st.spinner("📚 Loading and indexing documents..."):
            documents = load_documents()
            if not documents or len(documents) == 0:
                return False, "No documents found. Please add PDF files to the pdfs directory."
            
            chunks = split_documents(documents)
            chunks_with_ids = calculate_chunk_ids(chunks)
            add_to_chroma(chunks_with_ids)
            st.session_state.documents_loaded = True
            return True, f"Successfully indexed {len(chunks)} chunks from {len(documents)} documents!"
    except Exception as e:
        return False, f"Error loading documents: {str(e)}"


def load_and_index_uploaded_documents(uploaded_files):
    """Load and index documents from uploaded files"""
    try:
        if not uploaded_files:
            return False, "No files uploaded"
        
        with st.spinner(f"📚 Loading and indexing {len(uploaded_files)} uploaded file(s)..."):
            # Create temporary directory for uploaded files
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            
            # Save uploaded files to temp directory
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
            
            # Load documents from file paths
            documents = load_documents_from_files(file_paths)
            
            if not documents or len(documents) == 0:
                # Clean up temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False, "Failed to load documents. Please check if PDFs are valid."
            
            # Process and index documents
            chunks = split_documents(documents)
            chunks_with_ids = calculate_chunk_ids(chunks)
            add_to_chroma(chunks_with_ids)
            
            # Store uploaded file info in session state
            st.session_state.documents_loaded = True
            st.session_state.uploaded_files_info = [
                {"name": f.name, "size": f.size} for f in uploaded_files
            ]
            st.session_state.temp_dir = temp_dir
            
            return True, f"✅ Successfully indexed {len(chunks)} chunks from {len(documents)} page(s)!"
    except Exception as e:
        return False, f"❌ Error loading documents: {str(e)}"


def display_refinement_info(query: str, session_id: str):
    """Display information about refinement requests"""
    history = load_conversation_history()
    
    # Check if it's a refinement request
    if session_id in history and history[session_id].get("previous_answer"):
        try:
            # Extract refinement instruction
            instruction, part_type, part_number = extract_refinement_instruction(query)
            
            if part_type and part_number:
                st.info(f"🎯 **Slide-Specific Refinement Detected**\n\n"
                       f"Target: {part_type.capitalize()} {part_number}\n\n"
                       f"Instruction: {instruction}")
            else:
                st.info(f"✨ **General Refinement Detected**\n\n"
                       f"Instruction: {instruction}")
        except Exception as e:
            pass


# Note: Change tracking with GitHub-style diffs is displayed in the terminal output
# Users can view detailed color-coded diffs by checking the console/terminal


# ========================================
# BULLET-LEVEL FEEDBACK UI FUNCTIONS
# ========================================

def parse_bullets_from_response(response_text: str) -> list:
    """
    Parse individual bullet points/statements from LLM response.
    
    Returns:
        List of tuples: (bullet_id, bullet_text, slide_num)
    """
    bullets = []
    bullet_id = 0
    
    # Try to extract slides first
    slide_pattern = r'(?:^|\n)(?:\*\*)?Slide\s+(\d+)(?:\*\*)?:?\s*(.*?)(?=(?:\n(?:\*\*)?Slide\s+\d+|$))'
    slides = re.findall(slide_pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    if slides:
        # Parse bullets within each slide
        for slide_num, slide_content in slides:
            # Pattern 1: Bullet points with •, -, *
            bullet_matches = re.findall(r'[•\-*]\s+(.+?)(?=\n[•\-*\n]|$)', slide_content, re.DOTALL)
            if bullet_matches:
                for text in bullet_matches:
                    text = text.strip()
                    # Clean up: remove extra whitespace and newlines
                    text = ' '.join(text.split())
                    if text and len(text) > 10:
                        bullets.append((f"bullet_{bullet_id}", text, int(slide_num)))
                        bullet_id += 1
            else:
                # Pattern 2: Try key-value format (e.g., "Title: ...", "Definition: ...")
                kv_matches = re.findall(r'([A-Z][a-zA-Z\s]+):\s*[""""]?([^""""\n]+)[""""]?', slide_content)
                if kv_matches:
                    for key, value in kv_matches:
                        if key.strip() and value.strip() and len(value.strip()) > 10:
                            bullets.append((f"bullet_{bullet_id}", f"{key.strip()}: {value.strip()}", int(slide_num)))
                            bullet_id += 1
    else:
        # No slides detected, parse bullets from full text
        # Try bullet points
        bullet_matches = re.findall(r'(?:^|\n)[•\-*]\s+(.+?)(?=\n[•\-*]|$)', response_text, re.MULTILINE)
        if bullet_matches:
            for text in bullet_matches:
                text = text.strip()
                text = ' '.join(text.split())
                if text and len(text) > 10:
                    bullets.append((f"bullet_{bullet_id}", text, None))
                    bullet_id += 1
        else:
            # Try key-value pairs
            kv_matches = re.findall(r'([A-Z][a-zA-Z\s]+):\s*[""""]?([^""""\n]+)[""""]?', response_text)
            if kv_matches:
                for key, value in kv_matches:
                    if key.strip() and value.strip() and len(value.strip()) > 10:
                        bullets.append((f"bullet_{bullet_id}", f"{key.strip()}: {value.strip()}", None))
                        bullet_id += 1
    
    return bullets


def find_source_for_bullet(bullet_text: str, retrieved_chunks: list) -> dict:
    """
    Find the most relevant source chunk for a bullet point.
    
    Returns:
        Dictionary with source information
    """
    if not retrieved_chunks:
        return None
    
    # Simple keyword matching - count overlapping words
    bullet_words = set(re.findall(r'\b\w+\b', bullet_text.lower()))
    
    best_match = None
    best_score = 0
    
    for chunk in retrieved_chunks:
        chunk_text = chunk.get('content', '')
        chunk_words = set(re.findall(r'\b\w+\b', chunk_text.lower()))
        
        # Calculate overlap
        overlap = len(bullet_words & chunk_words)
        score = overlap / len(bullet_words) if bullet_words else 0
        
        if score > best_score:
            best_score = score
            best_match = chunk
    
    return best_match if best_score > 0.2 else retrieved_chunks[0]  # Fallback to first chunk


def highlight_text_in_source(source_text: str, bullet_text: str, max_length: int = 300) -> str:
    """
    Extract and highlight relevant portion of source text.
    
    Returns:
        Highlighted excerpt from source
    """
    # Extract keywords from bullet
    keywords = re.findall(r'\b\w{4,}\b', bullet_text.lower())
    keywords = list(set(keywords))[:5]  # Top 5 unique keywords
    
    if not keywords:
        return source_text[:max_length] + "..." if len(source_text) > max_length else source_text
    
    # Find best matching segment
    best_pos = 0
    best_match_count = 0
    
    # Sliding window approach
    words = source_text.split()
    window_size = 50
    
    for i in range(0, len(words), 10):
        window = ' '.join(words[i:i+window_size]).lower()
        match_count = sum(1 for kw in keywords if kw in window)
        if match_count > best_match_count:
            best_match_count = match_count
            best_pos = i
    
    # Extract segment
    segment_words = words[max(0, best_pos-10):best_pos+window_size+10]
    segment = ' '.join(segment_words)
    
    # Truncate if too long
    if len(segment) > max_length:
        segment = segment[:max_length] + "..."
    
    return segment


def display_bullet_feedback_ui(response_text: str, retrieved_chunks: list, session_id: str):
    """
    Display interactive bullet-level feedback UI with accept/reject buttons.
    """
    st.markdown("---")
    st.markdown("### 📝 **Review & Provide Feedback**")
    st.markdown("Review each generated statement and accept or reject it to help improve the system.")
    
    bullets = parse_bullets_from_response(response_text)
    
    if not bullets:
        st.warning("⚠️ No individual bullets detected in this response.")
        with st.expander("🔍 Debug Info", expanded=False):
            st.text(f"Response length: {len(response_text)} characters")
            st.text(f"Response preview (first 500 chars):")
            st.code(response_text[:500])
        st.info("💡 **Tip**: For best results, ask for presentations with explicit bullet points or numbered speaker notes.")
        return
    
    st.markdown(f"**Found {len(bullets)} statement(s) to review:**")
    
    for bullet_id, bullet_text, slide_num in bullets:
        # Check if feedback already given
        feedback_status = st.session_state.bullet_feedback.get(bullet_id, None)
        
        # Determine container style
        container_class = "bullet-container"
        if feedback_status == 'accepted':
            container_class += " accepted-bullet"
        elif feedback_status == 'rejected':
            container_class += " rejected-bullet"
        
        # Find source for this bullet
        source = find_source_for_bullet(bullet_text, retrieved_chunks)
        
        # Check if bullet contains math
        has_math = bool(extract_equations(bullet_text))
        
        # Display bullet in container
        st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
        
        # Slide indicator
        if slide_num:
            st.markdown(f"**Slide {slide_num}**")
        
        # Bullet text
        bullet_html = f'<div class="bullet-text">• {bullet_text}'
        if has_math:
            bullet_html += '<span class="math-badge">📐 Contains Math</span>'
        bullet_html += '</div>'
        st.markdown(bullet_html, unsafe_allow_html=True)
        
        # Source highlight
        if source:
            source_text = source.get('content', '')
            highlighted = highlight_text_in_source(source_text, bullet_text)
            metadata = source.get('metadata', {})
            
            source_file = os.path.basename(metadata.get('source', 'Unknown'))
            source_page = metadata.get('page', 'N/A')
            
            with st.expander(f"📖 Source: {source_file} (Page {source_page})", expanded=False):
                st.markdown(f'<div class="source-highlight">{highlighted}</div>', unsafe_allow_html=True)
                
                # Show math if present in source
                if has_math:
                    equations = extract_equations(source_text)
                    if equations:
                        st.markdown("**Equations in source:**")
                        for eq in equations[:2]:  # Show first 2 equations
                            st.latex(eq)
        
        # Feedback buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col2:
            if st.button("✅ Accept", key=f"accept_{bullet_id}", disabled=feedback_status=='accepted'):
                st.session_state.bullet_feedback[bullet_id] = 'accepted'
                
                # Store feedback
                store_bullet_feedback(
                    session_id=session_id,
                    bullet_id=bullet_id,
                    bullet_text=bullet_text,
                    source_chunk=source.get('content', '') if source else '',
                    source_metadata=source.get('metadata', {}) if source else {},
                    feedback='accepted'
                )
                st.rerun()
        
        with col3:
            if st.button("❌ Reject", key=f"reject_{bullet_id}", disabled=feedback_status=='rejected'):
                st.session_state.bullet_feedback[bullet_id] = 'rejected'
                
                # Store feedback
                store_bullet_feedback(
                    session_id=session_id,
                    bullet_id=bullet_id,
                    bullet_text=bullet_text,
                    source_chunk=source.get('content', '') if source else '',
                    source_metadata=source.get('metadata', {}) if source else {},
                    feedback='rejected'
                )
                st.rerun()
        
        # Feedback status
        if feedback_status:
            status_emoji = "✅" if feedback_status == 'accepted' else "❌"
            status_text = "Accepted" if feedback_status == 'accepted' else "Rejected"
            with col1:
                st.markdown(f"**{status_emoji} {status_text}**")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show feedback statistics and regenerate option
    st.markdown("---")
    stats = get_feedback_statistics(session_id)
    
    if stats['total'] > 0:
        st.markdown(f"""
        <div class="feedback-stats">
            <h4>📊 Feedback Summary</h4>
            <p><strong>Total Reviewed:</strong> {stats['total']}</p>
            <p><strong>Accepted:</strong> {stats['accepted']} ({stats['acceptance_rate']:.1f}%)</p>
            <p><strong>Rejected:</strong> {stats['rejected']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate refined answer based on feedback
        st.markdown("---")
        if st.button("✨ Generate Refined Answer Based on Feedback", type="primary", use_container_width=True):
            # Collect accepted and rejected bullets
            accepted_bullets = []
            rejected_bullets = []
            
            for bullet_id, bullet_text, slide_num in bullets:
                feedback = st.session_state.bullet_feedback.get(bullet_id)
                if feedback == 'accepted':
                    accepted_bullets.append(bullet_text)
                elif feedback == 'rejected':
                    rejected_bullets.append(bullet_text)
            
            if accepted_bullets or rejected_bullets:
                # Create refinement instruction
                refinement_instruction = "Based on user feedback, please refine the answer:\n\n"
                
                if accepted_bullets:
                    refinement_instruction += f"✅ **KEEP these points (user accepted {len(accepted_bullets)} statement(s)):**\n"
                    for i, bullet in enumerate(accepted_bullets[:3], 1):  # Show first 3
                        refinement_instruction += f"{i}. {bullet[:100]}...\n"
                    if len(accepted_bullets) > 3:
                        refinement_instruction += f"... and {len(accepted_bullets) - 3} more accepted statements.\n"
                    refinement_instruction += "\n"
                
                if rejected_bullets:
                    refinement_instruction += f"❌ **REMOVE/FIX these points (user rejected {len(rejected_bullets)} statement(s)):**\n"
                    for i, bullet in enumerate(rejected_bullets[:3], 1):  # Show first 3
                        refinement_instruction += f"{i}. {bullet[:100]}...\n"
                    if len(rejected_bullets) > 3:
                        refinement_instruction += f"... and {len(rejected_bullets) - 3} more rejected statements.\n"
                    refinement_instruction += "\n"
                
                refinement_instruction += "Please generate a refined answer that:\n"
                refinement_instruction += "- Keeps all accepted statements\n"
                refinement_instruction += "- Removes or corrects rejected statements\n"
                refinement_instruction += "- Maintains the same format and structure\n"
                refinement_instruction += "- Preserves equations and technical accuracy\n"
                
                # Store instruction and trigger refinement
                st.session_state['pending_feedback_refinement'] = refinement_instruction
                st.info("🔄 Refinement instruction created. Scroll down to submit it as a new query.")
                
                # Show the instruction
                with st.expander("📝 View Refinement Instruction", expanded=True):
                    st.markdown(refinement_instruction)
                    if st.button("📤 Submit This Refinement", key="submit_refinement"):
                        st.session_state['auto_submit_query'] = refinement_instruction
                        st.rerun()
            else:
                st.warning("⚠️ No feedback provided yet. Please accept or reject some bullets first.")


def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG Q&A System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your PDF documents using Retrieval Augmented Generation</p>', unsafe_allow_html=True)
    
    # Feature Showcase Expander
    with st.expander("✨ **Feature Showcase - Example Prompts**", expanded=False):
        st.markdown("""
        ### 🎯 Core Features
        
        #### 1️⃣ **Generate Presentations with Length & Style**
        ```
        "Create a 30s technical presentation on neural networks with 3 speaker notes per slide"
        "Make a 5min plain-english talk on quantum computing focusing on methodology"
        "Generate a 90s press-release about blockchain technology"
        ```
        
        #### 2️⃣ **Refine Specific Slides**
        ```
        "make slide 2 less technical"
        "make #3 more creative and engaging"
        "in the #2 slide, keep only 2 speaker notes and elaborate them"
        ```
        
        #### 3️⃣ **Change Length & Style**
        ```
        "make it 30s"  (changes from current length to 30s)
        "make it plain-english"  (changes from current style)
        "make slide 3 press-release style"  (changes specific slide style)
        ```
        
        #### 4️⃣ **Multiple Quality Refinements**
        ```
        "make it longer, creative, and interesting"
        "make slide 2 shorter but keep it technical"
        "refine it to be more engaging with examples"
        ```
        
        #### 5️⃣ **Math-Aware Queries**
        ```
        "Explain the equations in the paper"
        "Create slides focusing on the mathematical formulations"
        "Show the key algorithms with their equations"
        ```
        *Math-aware retrieval automatically boosts chunks containing relevant equations!*
        
        #### 6️⃣ **Interactive Feedback with Regeneration**
        ```
        1. Enable "Interactive Feedback" in sidebar
        2. Generate a presentation
        3. Review each bullet point
        4. Accept ✅ or Reject ❌ individual statements
        5. See sources and provenance for each bullet
        6. Click "Generate Refined Answer Based on Feedback"
        7. System regenerates keeping accepted, removing rejected!
        ```
        *Your feedback directly improves the output!*
        
        ---
        
        ### 📏 **Supported Lengths**
        - **30s** (75-90 words) - Quick pitch
        - **90s** (225-270 words) - Summary
        - **5min** (750-900 words) - Full presentation
        
        ### 🎨 **Supported Styles**
        - **technical** - Domain-specific, formal (for experts)
        - **plain-english** - Simple, accessible (for general audience)
        - **press-release** - Newsworthy, engaging (for media)
        
        ---
        
        ### 💡 **Pro Tips**
        - 📤 **Upload PDFs via sidebar** - No need to copy files to pdfs/ folder!
        - Use session IDs to manage multiple conversations
        - Check terminal output for detailed change tracking with color-coded diffs
        - All outputs are grounded in your uploaded PDF documents (RAG-based)
        - Reset database before uploading new document sets for clean context
        """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # SymPy Status Indicator
        if SYMPY_AVAILABLE:
            st.success("🔬 **Advanced Math Retrieval**: ✓ Active", icon="✅")
            with st.expander("ℹ️ Math Features"):
                st.markdown("""
                - ✓ Equation canonicalization
                - ✓ Handles equivalent forms (x+x = 2x)
                - ✓ Numeric equivalence (1/2 = 0.5)
                - ✓ Structural similarity matching
                - ✓ 5-component scoring system
                """)
        else:
            st.warning("⚠️ **SymPy not installed** - Using basic math matching", icon="⚠️")
            with st.expander("📦 Install for Better Results"):
                st.code("pip install sympy", language="bash")
                st.caption("Math retrieval will still work, but without canonicalization.")
        
        st.divider()
        
        # Session ID
        st.subheader("Session Management")
        new_session_id = st.text_input(
            "Session ID",
            value=st.session_state.session_id,
            help="Use different session IDs to keep separate conversation histories"
        )
        if new_session_id != st.session_state.session_id:
            st.session_state.session_id = new_session_id
            st.session_state.messages = []
            st.rerun()
        
        # Model selection
        st.subheader("Model Settings")
        st.session_state.model_name = st.selectbox(
            "Ollama Model",
            ["llama3.2:3b", "llama3.2:1b", "llama3.1:8b", "mistral", "phi3"],
            index=0,
            help="Select the Ollama model to use for generating responses"
        )
        
        # Document management
        st.subheader("📄 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to use as context for the RAG system"
        )
        
        # Display uploaded files
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) selected")
            with st.expander("📋 View Selected Files", expanded=True):
                for file in uploaded_files:
                    file_size_mb = file.size / (1024 * 1024)
                    st.text(f"📄 {file.name} ({file_size_mb:.2f} MB)")
            
            # Load and index button
            if st.button("🚀 Load & Index Uploaded Documents", use_container_width=True, type="primary"):
                success, message = load_and_index_uploaded_documents(uploaded_files)
                if success:
                    st.success(message)
                    st.balloons()
                else:
                    st.error(message)
        else:
            st.info("👆 Upload PDF files above to get started")
        
        # Show currently loaded documents
        if st.session_state.uploaded_files_info:
            st.markdown("---")
            st.markdown("**📚 Currently Indexed Documents:**")
            with st.expander(f"✅ {len(st.session_state.uploaded_files_info)} document(s) loaded", expanded=False):
                for file_info in st.session_state.uploaded_files_info:
                    file_size_mb = file_info['size'] / (1024 * 1024)
                    st.text(f"✓ {file_info['name']} ({file_size_mb:.2f} MB)")
        
        # Legacy: Load from pdfs/ folder (optional)
        st.markdown("---")
        st.markdown("**🗂️ Or load from pdfs/ folder:**")
        pdf_files = []
        if os.path.exists(DATA_PATH):
            pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
        
        if pdf_files:
            with st.expander(f"📁 {len(pdf_files)} file(s) in pdfs/ folder"):
                for pdf in pdf_files:
                    st.text(f"📄 {pdf}")
            
            if st.button("🔄 Load from pdfs/ folder", use_container_width=True):
                success, message = load_and_index_documents()
                if success:
                    st.success(message)
                else:
                    st.error(message)
        else:
            st.caption("(pdfs/ folder is empty)")
        
        # Reset database
        st.subheader("🗑️ Database Actions")
        if st.button("🗑️ Reset Vector Database", use_container_width=True):
            with st.spinner("Resetting database..."):
                try:
                    reset_collection()
                    st.session_state.documents_loaded = False
                    st.session_state.uploaded_files_info = []
                    
                    # Clean up temp directory if exists
                    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
                        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
                    st.session_state.temp_dir = None
                    
                    st.success("✅ Database reset successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Clear conversation history
        if st.button("🗑️ Clear Conversation History", use_container_width=True):
            try:
                clear_conversation_history()
                st.session_state.messages = []
                st.session_state.conversation_history = {}
                st.success("✅ Conversation history cleared!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # View conversation history
        st.subheader("📜 History")
        history = load_conversation_history()
        if history:
            st.info(f"📝 {len(history)} session(s) with history")
            
            # Full conversation history viewer
            with st.expander("📖 View Full Conversation History", expanded=False):
                st.markdown("### Complete Conversation History")
                
                for session_id, session_data in history.items():
                    st.markdown(f"---")
                    st.markdown(f"#### 🆔 Session: `{session_id}`")
                    
                    # Original Query
                    original_query = session_data.get('original_query', 'N/A')
                    st.markdown(f"**📝 Original Query:**")
                    st.info(original_query)
                    
                    # Previous Answer
                    previous_answer = session_data.get('previous_answer', 'N/A')
                    if previous_answer and previous_answer != 'N/A':
                        st.markdown(f"**💬 Answer:**")
                        # Show answer in expandable section if long
                        if len(previous_answer) > 500:
                            with st.expander(f"View Answer ({len(previous_answer)} characters)", expanded=False):
                                st.markdown(previous_answer)
                        else:
                            st.markdown(previous_answer)
                    
                    # Refinement Info
                    refinement_count = session_data.get('refinement_count', 0)
                    if refinement_count > 0:
                        st.success(f"✨ Refined {refinement_count} time(s)")
                    else:
                        st.text("🆕 Original answer (not refined)")
                    
                    # Copy button for session
                    st.code(f"Session ID: {session_id}", language=None)
            
            # Quick session list
            with st.expander("📋 Quick Session List"):
                for sid, data in history.items():
                    query_preview = data.get('original_query', 'N/A')[:60]
                    refinements = data.get('refinement_count', 0)
                    st.text(f"• {sid}: \"{query_preview}...\" ({refinements} refinements)")
        else:
            st.info("No conversation history yet")
        
        # Feedback UI Settings
        st.subheader("📝 Feedback Settings")
        st.session_state.show_feedback_ui = st.checkbox(
            "Enable Interactive Feedback",
            value=st.session_state.show_feedback_ui,
            help="Show bullet-level review UI with accept/reject buttons"
        )
        
        if st.session_state.show_feedback_ui:
            # Show feedback statistics
            stats = get_feedback_statistics(st.session_state.session_id)
            if stats['total'] > 0:
                st.metric("Acceptance Rate", f"{stats['acceptance_rate']:.1f}%")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Accepted", stats['accepted'])
                with col_b:
                    st.metric("Rejected", stats['rejected'])
            
            # Clear feedback button
            if st.button("🗑️ Clear Feedback Data", use_container_width=True):
                clear_feedback_data()
                st.session_state.bullet_feedback = {}
                st.success("✅ Feedback data cleared!")
        
        # About section
        st.subheader("ℹ️ About")
        math_status = "✓ Active" if SYMPY_AVAILABLE else "Basic Mode"
        st.markdown(f"""
        This RAG (Retrieval Augmented Generation) system:
        - 📤 Upload PDFs directly via UI
        - 🗂️ Indexes them in a vector database
        - 💬 Answers questions using document context
        - ✨ Supports answer refinement & change tracking
        - 📏 Multiple script lengths (30s, 90s, 5min)
        - 🎨 Multiple styles (technical, plain-english, press-release)
        - 🔬 **Advanced Math Retrieval** ({math_status})
          - Equation canonicalization (x+x = 2x)
          - Numeric equivalence (1/2 = 0.5)
          - 5-component similarity scoring
        - 📝 Interactive bullet feedback (accept/reject)
        - 📖 Source provenance with highlights
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        # Display bullet-level feedback UI (after chat messages, outside submission block)
        if (st.session_state.show_feedback_ui and 
            st.session_state.last_retrieved_chunks and 
            len(st.session_state.messages) > 0 and 
            st.session_state.messages[-1]["role"] == "assistant"):
            
            last_response = st.session_state.messages[-1]["content"]
            display_bullet_feedback_ui(
                response_text=last_response,
                retrieved_chunks=st.session_state.last_retrieved_chunks,
                session_id=st.session_state.session_id
            )
        
        # Query input (with auto-fill from feedback refinement)
        st.markdown('<div class="query-box">', unsafe_allow_html=True)
        
        # Check if there's an auto-submit query from feedback
        default_query = ""
        if 'auto_submit_query' in st.session_state:
            default_query = st.session_state['auto_submit_query']
            del st.session_state['auto_submit_query']  # Clear it
        
        query = st.text_area(
            "Enter your question:",
            value=default_query,
            height=100,
            placeholder="Ask a question about your documents...\n\nExamples:\n- What is machine learning?\n- Explain the main concepts\n- make it shorter (to refine previous answer)",
            key="query_input"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show hint if pending feedback refinement
        if 'pending_feedback_refinement' in st.session_state and query:
            st.success("✅ Feedback-based refinement ready! Click 'Submit Query' below to generate the refined answer.")
        
        # Query buttons
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        with col_btn1:
            submit_query = st.button("🚀 Submit Query", use_container_width=True, type="primary")
        with col_btn2:
            clear_chat = st.button("🗑️ Clear Chat", use_container_width=True)
        with col_btn3:
            auto_index = st.button("📚 Auto-Index", use_container_width=True)
        
        if clear_chat:
            st.session_state.messages = []
            st.rerun()
        
        if auto_index:
            success, message = load_and_index_documents()
            if success:
                st.success(message)
            else:
                st.error(message)
        
        # Process query
        if submit_query and query:
            # Check if this is a feedback-based refinement
            is_feedback_refinement = 'pending_feedback_refinement' in st.session_state and st.session_state.pending_feedback_refinement
            
            # Add user message to chat
            if is_feedback_refinement:
                st.session_state.messages.append({
                    "role": "user", 
                    "content": f"🔄 [Feedback-based refinement]\n\n{query}"
                })
                # Clear the pending refinement
                st.session_state.pending_feedback_refinement = None
            else:
                st.session_state.messages.append({"role": "user", "content": query})
            
            # Check if documents are loaded
            if not st.session_state.documents_loaded:
                with st.spinner("📚 Auto-loading documents..."):
                    success, message = load_and_index_documents()
                    if not success:
                        st.error(message)
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {message}"})
                        st.rerun()
            
            # Check if this might be a refinement request
            history = load_conversation_history()
            is_likely_refinement = any(word in query.lower() for word in ['make', 'shorter', 'longer', 'technical', 'concise', 'simpler', 'refine', 'improve', 'slide', 'section'])
            has_history = st.session_state.session_id in history and history[st.session_state.session_id].get("previous_answer")
            
            # Display refinement detection info
            if is_likely_refinement and has_history:
                display_refinement_info(query, st.session_state.session_id)
            elif is_likely_refinement and not has_history:
                st.warning(f"⚠️ This looks like a refinement request, but session '{st.session_state.session_id}' has no previous answer. It will be treated as a new query.")
            
            # Store previous answer for comparison
            previous_answer = None
            if has_history:
                previous_answer = history[st.session_state.session_id].get("previous_answer")
            
            # Process query
            with st.spinner("🤔 Thinking..."):
                try:
                    # Extract uploaded filenames for reference script loading
                    uploaded_filenames = [f['name'] for f in st.session_state.uploaded_files_info] if st.session_state.uploaded_files_info else None
                    
                    response, sources, retrieved_chunks = query_rag(
                        query_txt=query,
                        model_name=st.session_state.model_name,
                        session_id=st.session_state.session_id,
                        uploaded_filenames=uploaded_filenames
                    )
                    
                    # Store retrieved chunks for feedback UI
                    st.session_state.last_retrieved_chunks = retrieved_chunks
                    
                    if response:
                        # Add assistant response to chat (with feedback indicator if applicable)
                        if is_feedback_refinement:
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"✨ **[Refined based on your feedback]**\n\n{response}"
                            })
                            # Clear previous feedback for fresh review
                            st.session_state.bullet_feedback = {}
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Display response in result box
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        if is_feedback_refinement:
                            st.markdown("### ✨ Refined Answer (Based on Your Feedback)")
                            st.success("🎉 Answer regenerated! Accepted points are preserved, rejected points have been removed/corrected. Review the new answer below and provide fresh feedback if needed.")
                        else:
                            st.markdown("### 📝 Answer")
                        st.markdown(response)
                        
                        # Show if this was a refinement
                        current_history = load_conversation_history()
                        if st.session_state.session_id in current_history:
                            session_data = current_history[st.session_state.session_id]
                            refinement_count = session_data.get("refinement_count", 0)
                            
                            if refinement_count > 0:
                                st.info(f"✨ This answer has been refined {refinement_count} time(s)")
                                st.info(f"💡 **Tip**: Check the terminal output for detailed change tracking with GitHub-style color-coded diffs!")
                        
                        if sources:
                            with st.expander("📚 Sources"):
                                for i, source in enumerate(sources, 1):
                                    st.text(f"{i}. {source}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Note: Feedback UI now displayed outside submission block (after rerun)
                    else:
                        error_msg = "No response generated. Please check if documents are properly indexed."
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        st.error(error_msg)
                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.error(error_msg)
            
            st.rerun()
    
    with col2:
        st.header("📊 Status")
        
        # System status
        st.subheader("System Status")
        
        # Check if Chroma DB exists
        db_exists = os.path.exists(CHROMA_PATH) and os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3"))
        if db_exists:
            st.success("✅ Vector Database: Active")
        else:
            st.warning("⚠️ Vector Database: Not initialized")
        
        # Check PDF files
        if pdf_files:
            st.success(f"✅ PDF Files: {len(pdf_files)} found")
        else:
            st.error("❌ PDF Files: None found")
        
        # Documents loaded status
        if st.session_state.documents_loaded:
            st.success("✅ Documents: Indexed")
        else:
            st.warning("⚠️ Documents: Not indexed")
        
        # Current session info
        st.subheader("Current Session")
        st.info(f"**Session ID:** {st.session_state.session_id}")
        st.info(f"**Model:** {st.session_state.model_name}")
        st.info(f"**Messages:** {len(st.session_state.messages)}")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
        
        # Conversation history for current session
        history = load_conversation_history()
        if st.session_state.session_id in history:
            session_data = history[st.session_state.session_id]
            st.subheader("📜 Current Session History")
            st.success(f"✅ Session has history")
            
            # Original Query
            original_query = session_data.get('original_query', 'N/A')
            st.markdown(f"**📝 Original Query:**")
            st.text_area("Original Query", value=original_query, height=80, key=f"query_{st.session_state.session_id}", disabled=True, label_visibility="collapsed")
            
            # Answer
            previous_answer = session_data.get('previous_answer', '')
            if previous_answer:
                st.markdown(f"**💬 Current Answer:**")
                with st.expander(f"View Answer ({len(previous_answer)} chars)", expanded=False):
                    st.markdown(previous_answer)
            
            # Stats
            refinement_count = session_data.get('refinement_count', 0)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Refinements", refinement_count)
            with col2:
                has_previous = bool(previous_answer)
                if has_previous:
                    st.success("✅ Can refine")
                else:
                    st.warning("⚠️ No answer")
            
            # Full JSON view
            with st.expander("🔍 View Raw JSON Data"):
                st.json(session_data)
        else:
            st.info(f"ℹ️ No history for session '{st.session_state.session_id}'")
            st.text("Ask a question first to create history")
        
        # Full conversation history button
        st.subheader("📚 All Conversations")
        if st.button("📖 View All Conversation History", use_container_width=True):
            st.session_state.show_full_history = True
        
        if st.session_state.get('show_full_history', False):
            with st.expander("📖 Complete Conversation History", expanded=True):
                history = load_conversation_history()
                if history:
                    for session_id, session_data in history.items():
                        st.markdown(f"### 🆔 Session: `{session_id}`")
                        
                        # Query
                        st.markdown("**📝 Query:**")
                        st.info(session_data.get('original_query', 'N/A'))
                        
                        # Answer
                        answer = session_data.get('previous_answer', '')
                        if answer:
                            st.markdown("**💬 Answer:**")
                            if len(answer) > 1000:
                                with st.expander(f"View Answer ({len(answer)} characters)"):
                                    st.markdown(answer)
                            else:
                                st.markdown(answer)
                        
                        # Metadata
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text(f"Refinements: {session_data.get('refinement_count', 0)}")
                        with col2:
                            st.text(f"Answer Length: {len(answer)} chars")
                        
                        st.markdown("---")
                else:
                    st.info("No conversation history found")
                
                if st.button("❌ Close History Viewer"):
                    st.session_state.show_full_history = False
                    st.rerun()

if __name__ == "__main__":
    main()

