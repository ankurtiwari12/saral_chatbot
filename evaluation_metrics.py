"""
Evaluation Metrics for RAG System

Provides:
1. Factuality Proxy - overlap between generated claims and source sentences
2. Citation Coverage - percentage of content with provenance
3. ROUGE / BERTScore for quality assessment

Logs to terminal during runtime (not shown in UI).
"""

import re
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import numpy as np

# Try to import ROUGE and BERTScore
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("[WARNING] rouge-score not installed. Install: pip install rouge-score")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("[WARNING] bert-score not installed. Install: pip install bert-score")


class Colors:
    """ANSI color codes for terminal output"""
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    END = '\033[0m'
    BOLD = '\033[1m'


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple heuristics.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting (handles basic cases)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def split_into_claims(generated_text: str) -> List[str]:
    """
    Split generated text into individual claims/statements.
    
    Args:
        generated_text: Generated answer
        
    Returns:
        List of claims
    """
    claims = []
    
    # Split by slides if present
    slides = re.split(r'\*\*Slide\s+\d+:', generated_text)
    
    for slide in slides:
        # Extract speaker notes (main content source)
        speaker_notes = re.findall(r'Speaker Notes?:\s*(.+?)(?=\n\n|$)', slide, re.DOTALL | re.IGNORECASE)
        
        # Extract bullet points
        bullets = re.findall(r'(?:^|\n)\s*[-•*]\s*(.+?)(?=\n|$)', slide)
        
        # Extract numbered points
        numbered = re.findall(r'(?:^|\n)\s*\d+\.\s*(.+?)(?=\n|$)', slide)
        
        # Add all found claims
        claims.extend(speaker_notes)
        claims.extend(bullets)
        claims.extend(numbered)
    
    # If no structured content, fall back to sentences
    if not claims:
        claims = split_into_sentences(generated_text)
    
    return [c.strip() for c in claims if c.strip()]


def compute_exact_overlap(claim: str, source_sentence: str) -> float:
    """
    Compute exact word overlap between claim and source.
    
    Args:
        claim: Generated claim
        source_sentence: Source sentence from retrieved chunks
        
    Returns:
        Overlap ratio (0-1)
    """
    # Normalize: lowercase, remove punctuation
    claim_words = set(re.findall(r'\w+', claim.lower()))
    source_words = set(re.findall(r'\w+', source_sentence.lower()))
    
    if not claim_words:
        return 0.0
    
    overlap = len(claim_words & source_words)
    return overlap / len(claim_words)


def compute_sequence_similarity(claim: str, source_sentence: str) -> float:
    """
    Compute sequence similarity using SequenceMatcher.
    
    Args:
        claim: Generated claim
        source_sentence: Source sentence
        
    Returns:
        Similarity ratio (0-1)
    """
    return SequenceMatcher(None, claim.lower(), source_sentence.lower()).ratio()


def compute_factuality_score(
    generated_text: str,
    retrieved_chunks: List[Dict],
    threshold: float = 0.3
) -> Dict:
    """
    Compute factuality proxy by measuring overlap with source sentences.
    
    Args:
        generated_text: Generated answer
        retrieved_chunks: List of retrieved chunk dicts with 'content' key
        threshold: Minimum overlap to consider claim grounded
        
    Returns:
        Dictionary with factuality metrics
    """
    # Extract claims from generated text
    claims = split_into_claims(generated_text)
    
    # Extract sentences from all retrieved chunks
    source_sentences = []
    for chunk in retrieved_chunks:
        content = chunk.get('content', '')
        source_sentences.extend(split_into_sentences(content))
    
    # Match each claim to best source sentence
    grounded_claims = 0
    claim_scores = []
    
    for claim in claims:
        best_overlap = 0.0
        best_similarity = 0.0
        
        for source_sent in source_sentences:
            overlap = compute_exact_overlap(claim, source_sent)
            similarity = compute_sequence_similarity(claim, source_sent)
            
            best_overlap = max(best_overlap, overlap)
            best_similarity = max(best_similarity, similarity)
        
        # Use max of overlap and similarity
        claim_score = max(best_overlap, best_similarity)
        claim_scores.append(claim_score)
        
        if claim_score >= threshold:
            grounded_claims += 1
    
    # Compute metrics
    total_claims = len(claims)
    factuality_score = grounded_claims / total_claims if total_claims > 0 else 0.0
    avg_overlap = np.mean(claim_scores) if claim_scores else 0.0
    
    return {
        'total_claims': total_claims,
        'grounded_claims': grounded_claims,
        'ungrounded_claims': total_claims - grounded_claims,
        'factuality_score': factuality_score,  # Percentage grounded
        'avg_overlap': avg_overlap,  # Average overlap score
        'claim_scores': claim_scores,
        'threshold': threshold
    }


def compute_citation_coverage(
    generated_text: str,
    retrieved_chunks: List[Dict],
    threshold: float = 0.25
) -> Dict:
    """
    Compute what percentage of content has retrieved provenance.
    
    This checks if each generated claim can be traced back to at least
    one sentence in the retrieved source chunks.
    
    Args:
        generated_text: Generated answer
        retrieved_chunks: Retrieved chunks
        threshold: Minimum similarity to consider provenance (default: 0.25)
        
    Returns:
        Dictionary with citation coverage metrics
    """
    claims = split_into_claims(generated_text)
    
    # Extract all sentences from all retrieved chunks
    source_sentences = []
    for chunk in retrieved_chunks:
        content = chunk.get('content', '')
        source_sentences.extend(split_into_sentences(content))
    
    # Check which claims have source provenance
    claims_with_provenance = 0
    claim_provenance_scores = []
    
    for claim in claims:
        # Check if claim matches ANY sentence in ANY source chunk
        has_provenance = False
        best_match_score = 0.0
        
        for source_sent in source_sentences:
            # Use both exact overlap and sequence similarity
            overlap = compute_exact_overlap(claim, source_sent)
            similarity = compute_sequence_similarity(claim, source_sent)
            match_score = max(overlap, similarity)
            best_match_score = max(best_match_score, match_score)
            
            if match_score > threshold:
                has_provenance = True
                break  # Found provenance, no need to check more
        
        claim_provenance_scores.append(best_match_score)
        
        if has_provenance:
            claims_with_provenance += 1
    
    total_claims = len(claims)
    coverage = claims_with_provenance / total_claims if total_claims > 0 else 0.0
    avg_provenance_score = np.mean(claim_provenance_scores) if claim_provenance_scores else 0.0
    
    return {
        'total_claims': total_claims,
        'claims_with_provenance': claims_with_provenance,
        'claims_without_provenance': total_claims - claims_with_provenance,
        'citation_coverage': coverage,  # Percentage with provenance
        'source_chunks_count': len(retrieved_chunks),
        'avg_provenance_score': avg_provenance_score,
        'threshold': threshold
    }


def compute_rouge_scores(generated_text: str, reference_text: str) -> Dict:
    """
    Compute ROUGE scores against reference text.
    
    Args:
        generated_text: Generated answer
        reference_text: Human-authored reference script
        
    Returns:
        Dictionary with ROUGE scores
    """
    if not ROUGE_AVAILABLE:
        return {
            'rouge1': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
            'rouge2': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
            'rougeL': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
            'available': False
        }
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    
    return {
        'rouge1': {
            'precision': scores['rouge1'].precision,
            'recall': scores['rouge1'].recall,
            'fmeasure': scores['rouge1'].fmeasure
        },
        'rouge2': {
            'precision': scores['rouge2'].precision,
            'recall': scores['rouge2'].recall,
            'fmeasure': scores['rouge2'].fmeasure
        },
        'rougeL': {
            'precision': scores['rougeL'].precision,
            'recall': scores['rougeL'].recall,
            'fmeasure': scores['rougeL'].fmeasure
        },
        'available': True
    }


def compute_bertscore(generated_text: str, reference_text: str) -> Dict:
    """
    Compute BERTScore against reference text.
    
    Args:
        generated_text: Generated answer
        reference_text: Human-authored reference script
        
    Returns:
        Dictionary with BERTScore metrics
    """
    if not BERTSCORE_AVAILABLE:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'available': False
        }
    
    # Compute BERTScore
    P, R, F1 = bert_score([generated_text], [reference_text], lang='en', verbose=False)
    
    return {
        'precision': float(P[0]),
        'recall': float(R[0]),
        'f1': float(F1[0]),
        'available': True
    }


def evaluate_response(
    generated_text: str,
    retrieved_chunks: List[Dict],
    reference_text: str = None
) -> Dict:
    """
    Comprehensive evaluation of generated response.
    
    Args:
        generated_text: Generated answer
        retrieved_chunks: Retrieved source chunks
        reference_text: Optional human-authored reference (for ROUGE/BERTScore)
        
    Returns:
        Dictionary with all evaluation metrics
    """
    evaluation = {}
    
    # 1. Factuality Proxy
    factuality = compute_factuality_score(generated_text, retrieved_chunks)
    evaluation['factuality'] = factuality
    
    # 2. Citation Coverage
    coverage = compute_citation_coverage(generated_text, retrieved_chunks)
    evaluation['citation_coverage'] = coverage
    
    # 3. ROUGE Scores (if reference provided)
    if reference_text:
        rouge = compute_rouge_scores(generated_text, reference_text)
        evaluation['rouge'] = rouge
        
        # 4. BERTScore (if reference provided)
        bertscore = compute_bertscore(generated_text, reference_text)
        evaluation['bertscore'] = bertscore
    else:
        evaluation['rouge'] = None
        evaluation['bertscore'] = None
    
    return evaluation


def print_evaluation_report(evaluation: Dict, show_detailed: bool = False):
    """
    Print evaluation metrics to terminal.
    
    Args:
        evaluation: Evaluation dictionary from evaluate_response()
        show_detailed: Show detailed claim-by-claim scores
    """
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}EVALUATION METRICS{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    # 1. Factuality Proxy
    fact = evaluation['factuality']
    fact_color = Colors.GREEN if fact['factuality_score'] >= 0.7 else (Colors.YELLOW if fact['factuality_score'] >= 0.5 else Colors.RED)
    
    print(f"{Colors.BOLD}1. FACTUALITY PROXY{Colors.END}")
    print(f"   Total Claims: {fact['total_claims']}")
    print(f"   Grounded Claims: {Colors.GREEN}{fact['grounded_claims']}{Colors.END}")
    print(f"   Ungrounded Claims: {Colors.RED}{fact['ungrounded_claims']}{Colors.END}")
    print(f"   {fact_color}Factuality Score: {fact['factuality_score']:.2%}{Colors.END} (threshold: {fact['threshold']})")
    print(f"   Average Overlap: {fact['avg_overlap']:.3f}")
    
    if show_detailed and fact['claim_scores']:
        print(f"   Claim-by-Claim Scores:")
        for i, score in enumerate(fact['claim_scores'][:10], 1):  # Show first 10
            score_color = Colors.GREEN if score >= fact['threshold'] else Colors.RED
            print(f"     Claim {i}: {score_color}{score:.3f}{Colors.END}")
        if len(fact['claim_scores']) > 10:
            print(f"     ... ({len(fact['claim_scores']) - 10} more claims)")
    
    print()
    
    # 2. Citation Coverage
    cov = evaluation['citation_coverage']
    cov_color = Colors.GREEN if cov['citation_coverage'] >= 0.8 else (Colors.YELLOW if cov['citation_coverage'] >= 0.6 else Colors.RED)
    
    print(f"{Colors.BOLD}2. CITATION COVERAGE{Colors.END}")
    print(f"   Total Claims: {cov['total_claims']}")
    print(f"   With Provenance: {Colors.GREEN}{cov['claims_with_provenance']}{Colors.END}")
    print(f"   Without Provenance: {Colors.RED}{cov['claims_without_provenance']}{Colors.END}")
    print(f"   {cov_color}Coverage Score: {cov['citation_coverage']:.2%}{Colors.END} (threshold: {cov['threshold']})")
    print(f"   Average Provenance: {cov['avg_provenance_score']:.3f}")
    print(f"   Source Chunks Used: {cov['source_chunks_count']}")
    print()
    
    # 3. ROUGE Scores
    if evaluation['rouge'] and evaluation['rouge']['available']:
        rouge = evaluation['rouge']
        print(f"{Colors.BOLD}3. ROUGE SCORES{Colors.END}")
        print(f"   ROUGE-1:")
        print(f"     Precision: {rouge['rouge1']['precision']:.4f}")
        print(f"     Recall:    {rouge['rouge1']['recall']:.4f}")
        print(f"     F-Measure: {Colors.GREEN}{rouge['rouge1']['fmeasure']:.4f}{Colors.END}")
        print(f"   ROUGE-2:")
        print(f"     Precision: {rouge['rouge2']['precision']:.4f}")
        print(f"     Recall:    {rouge['rouge2']['recall']:.4f}")
        print(f"     F-Measure: {Colors.GREEN}{rouge['rouge2']['fmeasure']:.4f}{Colors.END}")
        print(f"   ROUGE-L:")
        print(f"     Precision: {rouge['rougeL']['precision']:.4f}")
        print(f"     Recall:    {rouge['rougeL']['recall']:.4f}")
        print(f"     F-Measure: {Colors.GREEN}{rouge['rougeL']['fmeasure']:.4f}{Colors.END}")
        print()
    elif evaluation['rouge'] is not None:
        print(f"{Colors.BOLD}3. ROUGE SCORES{Colors.END}")
        print(f"   {Colors.YELLOW}Not available (rouge-score not installed){Colors.END}")
        print(f"   Install: pip install rouge-score")
        print()
    
    # 4. BERTScore
    if evaluation['bertscore'] and evaluation['bertscore']['available']:
        bert = evaluation['bertscore']
        print(f"{Colors.BOLD}4. BERTSCORE{Colors.END}")
        print(f"   Precision: {bert['precision']:.4f}")
        print(f"   Recall:    {bert['recall']:.4f}")
        print(f"   F1:        {Colors.GREEN}{bert['f1']:.4f}{Colors.END}")
        print()
    elif evaluation['bertscore'] is not None:
        print(f"{Colors.BOLD}4. BERTSCORE{Colors.END}")
        print(f"   {Colors.YELLOW}Not available (bert-score not installed){Colors.END}")
        print(f"   Install: pip install bert-score")
        print()
    
    print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")


def get_evaluation_summary(evaluation: Dict) -> str:
    """
    Get one-line summary of evaluation metrics.
    
    Args:
        evaluation: Evaluation dictionary
        
    Returns:
        Summary string
    """
    fact_score = evaluation['factuality']['factuality_score']
    cov_score = evaluation['citation_coverage']['citation_coverage']
    
    summary = f"Factuality: {fact_score:.1%} | Coverage: {cov_score:.1%}"
    
    if evaluation['rouge'] and evaluation['rouge']['available']:
        rouge_f1 = evaluation['rouge']['rouge1']['fmeasure']
        summary += f" | ROUGE-1: {rouge_f1:.3f}"
    
    if evaluation['bertscore'] and evaluation['bertscore']['available']:
        bert_f1 = evaluation['bertscore']['f1']
        summary += f" | BERTScore: {bert_f1:.3f}"
    
    return summary

