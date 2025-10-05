import streamlit as st
import pandas as pd
import nltk
import spacy
import time
import logging
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Set page configuration
st.set_page_config(
    page_title="Financial Article Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'article_submitted' not in st.session_state:
    st.session_state.article_submitted = False
if 'article_text' not in st.session_state:
    st.session_state.article_text = ""
if 'ambiguity_results' not in st.session_state:
    st.session_state.ambiguity_results = None

# Load models
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("Please download spaCy model: python -m spacy download en_core_web_sm")
        return None

# Load Loughran-McDonald Dictionary
@st.cache_resource
def load_lm_dictionary():
    """Load Loughran-McDonald dictionary"""
    try:
        lm_dict = pd.ExcelFile(r"C:\Users\saman\Downloads\LoughranMcDonald_SentimentWordLists_2018.xlsx")
        
        sheet_map = {
            "Positive": "positive",
            "Negative": "negative", 
            "Uncertainty": "uncertainty",
            "StrongModal": "strong_modal",
            "WeakModal": "weak_modal",
            "Constraining": "constraining",
            "Litigious": "litigious"
        }

        LM_DICT = {}
        for sheet, key in sheet_map.items():
            df = lm_dict.parse(sheet, header=None)
            words = set(df.iloc[:, 0].dropna().astype(str).str.lower())
            LM_DICT[key] = words
        
        st.success("✅ LM Dictionary loaded successfully!")
        return LM_DICT
    except Exception as e:
        st.error(f"Error loading LM dictionary: {e}")
        # Return empty dict as fallback
        return {key: set() for key in sheet_map.values()}

# Hedging words
HEDGING_WORDS = {
    "may", "might", "could", "possibly", "likely", "tends", 
    "appears", "seems", "perhaps", "approximately", "relatively", "generally", "often"
}

# CLAUSE_TYPES for syntactic analysis
CLAUSE_TYPES = {"prep", "ccomp", "advcl", "relcl"}

# Your Pragmatic Ambiguity Function
def quantify_pragmatic_ambiguity(article_text, LM_DICT, max_sentences=30):
    """
    Quantify pragmatic ambiguity for a single article using core LM dictionary rules
    + hedging words.

    Args:
        article_text (str): Article text.
        LM_DICT (dict): Loughran-McDonald dictionary (category -> set of words).
        max_sentences (int): Max sentences to process per article.

    Returns:
        dict: Count, percentage, flagged sentences, total sentences, and detailed analysis.
    """
    try:
        sentences = nltk.sent_tokenize(article_text)
        total_sentences = len(sentences)
        sentences = sentences[:max_sentences]

        if not sentences:
            return {
                'count': 0,
                'percentage': 0.0,
                'flagged_sentences': [],
                'total_sentences': total_sentences,
                'detailed_analysis': [],
                'ambiguity_score': 0.0
            }

        ambiguous_count = 0
        flagged_sentences = []
        detailed_analysis = []

        for i, sent in enumerate(sentences):
            words = set(token.lower().strip(",.!?;:") for token in sent.split())

            # Check presence of LM categories
            has_positive = bool(words & LM_DICT.get("positive", set()))
            has_negative = bool(words & LM_DICT.get("negative", set()))
            has_uncertainty = bool(words & LM_DICT.get("uncertainty", set()))
            has_strong_modal = bool(words & LM_DICT.get("strong_modal", set()))
            has_weak_modal = bool(words & LM_DICT.get("weak_modal", set()))
            is_rhetorical = sent.lower().startswith(('what a ', 'how '))
            has_hedge = bool(words & HEDGING_WORDS)

            # Core 4 rules + hedging
            rule_1 = (has_positive and has_negative)                      # Positive + Negative
            rule_2 = (has_positive and has_uncertainty)                  # Positive + Uncertainty
            rule_3 = (has_strong_modal and (has_uncertainty or has_weak_modal or has_negative))  # Strong modal conflicts
            rule_4 = (is_rhetorical and has_negative)                    # Rhetorical question + Negative
            rule_5 = ((has_positive or has_negative) and has_hedge)        # Hedging + sentiment

            is_ambiguous = rule_1 or rule_2 or rule_3 or rule_4 or rule_5

            if is_ambiguous:
                ambiguous_count += 1
                flagged_sentences.append(sent)
                
                # Store detailed analysis for this sentence
                analysis = {
                    'sentence_number': i + 1,
                    'sentence': sent,
                    'rules_triggered': [],
                    'features_found': []
                }
                
                if rule_1: analysis['rules_triggered'].append("Positive + Negative conflict")
                if rule_2: analysis['rules_triggered'].append("Positive + Uncertainty")
                if rule_3: analysis['rules_triggered'].append("Strong modal conflict")
                if rule_4: analysis['rules_triggered'].append("Rhetorical + Negative")
                if rule_5: analysis['rules_triggered'].append("Sentiment + Hedging")
                
                if has_positive: analysis['features_found'].append("Positive words")
                if has_negative: analysis['features_found'].append("Negative words")
                if has_uncertainty: analysis['features_found'].append("Uncertainty words")
                if has_strong_modal: analysis['features_found'].append("Strong modal words")
                if has_weak_modal: analysis['features_found'].append("Weak modal words")
                if is_rhetorical: analysis['features_found'].append("Rhetorical structure")
                if has_hedge: analysis['features_found'].append("Hedging words")
                
                detailed_analysis.append(analysis)

        percentage = (ambiguous_count / total_sentences) * 100 if total_sentences else 0

        return {
            'count': ambiguous_count,
            'percentage': percentage,
            'flagged_sentences': flagged_sentences,
            'total_sentences': total_sentences,
            'detailed_analysis': detailed_analysis,
            'ambiguity_score': percentage / 100  # Normalized score between 0-1
        }

    except Exception as e:
        st.error(f"Error in pragmatic ambiguity analysis: {e}")
        return {
            'count': 0,
            'percentage': 0.0,
            'flagged_sentences': [],
            'total_sentences': 0,
            'detailed_analysis': [],
            'ambiguity_score': 0.0
        }

# Your Syntactic Ambiguity Function
def quantify_syntactic_ambiguity(article_text, max_sentences=30, threshold=3):
    """
    Quantify syntactic ambiguity in a single article without weights.
    """
    nlp_model = load_spacy_model()
    if nlp_model is None:
        return {
            "count": 0,
            "percentage": 0.0,
            "flagged_sentences": [],
            "total_sentences": 0,
            "detailed_analysis": [],
            "ambiguity_score": 0.0
        }
    
    try:
        doc = nlp_model(article_text)
        sentences = list(doc.sents)[:max_sentences]
        flagged_sentences = []
        detailed_analysis = []

        for i, sent in enumerate(sentences):
            # Count complex clauses
            clause_count = sum(1 for token in sent if token.dep_ in CLAUSE_TYPES)
            
            # Detailed analysis for each sentence
            sentence_analysis = {
                'sentence_number': i + 1,
                'sentence': sent.text,
                'clause_count': clause_count,
                'threshold': threshold,
                'is_ambiguous': clause_count >= threshold,
                'clause_types_found': []
            }
            
            # Identify specific clause types found
            for token in sent:
                if token.dep_ in CLAUSE_TYPES:
                    sentence_analysis['clause_types_found'].append(f"{token.dep_}: '{token.text}'")
            
            if clause_count >= threshold:
                flagged_sentences.append(sent.text)
                detailed_analysis.append(sentence_analysis)

        total_sentences = len(sentences)
        count = len(flagged_sentences)
        percentage = (count / total_sentences) * 100 if total_sentences else 0

        return {
            "count": count,
            "percentage": percentage,
            "flagged_sentences": flagged_sentences,
            "total_sentences": total_sentences,
            "detailed_analysis": detailed_analysis,
            "ambiguity_score": percentage / 100
        }

    except Exception as e:
        logging.warning(f"Error processing article: {e}")
        return {
            "count": 0,
            "percentage": 0.0,
            "flagged_sentences": [],
            "total_sentences": 0,
            "detailed_analysis": [],
            "ambiguity_score": 0.0
        }

# Your Semantic Ambiguity Function
def quantify_semantic_ambiguity(article_text, max_sentences=30, min_meanings=2):
    """
    Counts sentences in the text that contain semantically ambiguous words.
    
    Args:
        article_text (str): Article text.
        max_sentences (int): Maximum sentences to process.
        min_meanings (int): Minimum number of WordNet senses to consider a word ambiguous.
    
    Returns:
        dict: Count, percentage, flagged sentences, total sentences, and detailed analysis.
    """
    try:
        sentences = sent_tokenize(article_text)
        total_sentences = len(sentences)
        sentences = sentences[:max_sentences]
        
        flagged_sentences = []
        detailed_analysis = []

        for i, sent in enumerate(sentences):
            words = word_tokenize(sent)
            ambiguous_words_found = []
            
            for word in words:
                # Clean the word and check if it's alphabetic
                clean_word = word.lower().strip(",.!?;:\"'()[]")
                if clean_word and clean_word.isalpha():
                    synsets = wn.synsets(clean_word)
                    if len(synsets) >= min_meanings:
                        ambiguous_words_found.append({
                            'word': clean_word,
                            'meanings_count': len(synsets),
                            'meanings': [syn.definition() for syn in synsets[:3]]  # First 3 meanings
                        })
            
            if ambiguous_words_found:
                flagged_sentences.append(sent)
                
                # Store detailed analysis for this sentence
                analysis = {
                    'sentence_number': i + 1,
                    'sentence': sent,
                    'ambiguous_words': ambiguous_words_found,
                    'total_ambiguous_words': len(ambiguous_words_found)
                }
                detailed_analysis.append(analysis)
                
        count = len(flagged_sentences)
        percentage = (count / total_sentences) * 100 if total_sentences else 0

        return {
            "count": count,
            "percentage": percentage,
            "flagged_sentences": flagged_sentences,
            "total_sentences": total_sentences,
            "detailed_analysis": detailed_analysis,
            "ambiguity_score": percentage / 100
        }

    except Exception as e:
        st.error(f"Error in semantic ambiguity analysis: {e}")
        return {
            "count": 0,
            "percentage": 0.0,
            "flagged_sentences": [],
            "total_sentences": 0,
            "detailed_analysis": [],
            "ambiguity_score": 0.0
        }

# Main App
st.title("📊 Financial Article Ambiguity Analyzer")
st.markdown("Analyze different types of ambiguity in financial articles")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    st.markdown("---")
    
    st.subheader("⚙️ Analysis Settings")
    
    st.write("**Syntactic Analysis:**")
    syntactic_threshold = st.slider(
        "Clause Count Threshold",
        min_value=1,
        max_value=10,
        value=3,
        help="Higher values = more strict (fewer sentences flagged as ambiguous)"
    )
    
    st.write("**Semantic Analysis:**")
    semantic_threshold = st.slider(
        "Minimum Word Meanings",
        min_value=1,
        max_value=10,
        value=2,
        help="Minimum number of WordNet senses to consider a word ambiguous"
    )
    
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
    **Ambiguity Types:**
    - **Pragmatic**: Conflicting sentiment signals, hedging, uncertainty
    - **Syntactic**: Complex sentence structures, nested clauses
    - **Semantic**: Words with multiple meanings
    """)

# Main content area - BALANCED COLUMNS
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📄 Input Article")
    article_text = st.text_area(
        "Paste your financial article:",
        height=300,
        placeholder="Enter financial news, earnings reports, or market analysis...",
        key="article_input",
        label_visibility="collapsed"
    )
    
    submit_col1, submit_col2 = st.columns(2)
    with submit_col1:
        if st.button("📥 Submit Article", use_container_width=True, type="primary"):
            if article_text.strip():
                st.session_state.article_text = article_text
                st.session_state.article_submitted = True
                st.session_state.ambiguity_results = None
                st.success("✅ Article submitted successfully!")
                st.rerun()
            else:
                st.error("❌ Please enter article text.")

with col2:
    st.subheader("📋 Article Status")
    
    if st.session_state.article_submitted:
        # Show clear submission status
        st.success("✅ Article Submitted")
        
        # Quick stats
        word_count = len(st.session_state.article_text.split())
        sent_count = len(nltk.sent_tokenize(st.session_state.article_text))
        
        # Display metrics in a more compact way
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.metric("Word Count", word_count)
            st.metric("Sentence Count", sent_count)
        with stat_col2:
            st.metric("Syntactic Threshold", syntactic_threshold)
            st.metric("Semantic Threshold", semantic_threshold)
        
        # Analysis button
        st.markdown("---")
        if st.button("🔍 Analyze Ambiguity", use_container_width=True, type="secondary"):
            st.session_state.ambiguity_results = "loading"
            st.rerun()
    else:
        st.info("📝 Waiting for article submission...")
        st.markdown("---")
        st.write("**Next Steps:**")
        st.write("1. Paste article in the left panel")
        st.write("2. Click 'Submit Article'")
        st.write("3. Analyze ambiguity scores")

# Show article preview after submission
if st.session_state.article_submitted and not st.session_state.ambiguity_results:
    st.markdown("---")
    st.subheader("📋 Submitted Article Preview")
    
    # Show first 200 characters as preview
    preview_text = st.session_state.article_text[:200] + "..." if len(st.session_state.article_text) > 200 else st.session_state.article_text
    st.text_area("Article Preview", preview_text, height=100, key="preview", disabled=True)
    
    st.info("🎯 Ready for ambiguity analysis! Click 'Analyze Ambiguity' to proceed.")

# Ambiguity Analysis Section
if st.session_state.ambiguity_results == "loading":
    st.markdown("---")
    st.subheader("🔍 Analyzing Ambiguity")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load LM dictionary
    LM_DICT = load_lm_dictionary()
    
    # Simulate analysis progress
    results = {}
    ambiguities = [
        ("Pragmatic", lambda: quantify_pragmatic_ambiguity(st.session_state.article_text, LM_DICT)),
        ("Syntactic", lambda: quantify_syntactic_ambiguity(st.session_state.article_text, threshold=syntactic_threshold)),
        ("Semantic", lambda: quantify_semantic_ambiguity(st.session_state.article_text, min_meanings=semantic_threshold))
    ]
    
    for i, (name, func) in enumerate(ambiguities):
        status_text.text(f"Analyzing {name} Ambiguity...")
        results[name.lower()] = func()
        progress_bar.progress((i + 1) / len(ambiguities))
        time.sleep(0.5)  # Simulate processing time
    
    st.session_state.ambiguity_results = results
    status_text.text("✅ Analysis Complete!")
    time.sleep(1)
    st.rerun()

# Display Ambiguity Results
if st.session_state.ambiguity_results and st.session_state.ambiguity_results != "loading":
    results = st.session_state.ambiguity_results
    
    st.markdown("---")
    st.subheader("📊 Ambiguity Analysis Results")
    
    # Progress bars for ambiguity scores
    st.write("**Ambiguity Scores:**")
    
    # Pragmatic Ambiguity
    pragmatic_data = results.get('pragmatic', {})
    pragmatic_score = pragmatic_data.get('percentage', 0)
    st.write(f"**Pragmatic Ambiguity:** {pragmatic_score:.1f}%")
    st.progress(pragmatic_score / 100)
    st.caption(f"{pragmatic_data.get('count', 0)} out of {pragmatic_data.get('total_sentences', 0)} sentences")
    
    # Syntactic Ambiguity
    syntactic_data = results.get('syntactic', {})
    syntactic_score = syntactic_data.get('percentage', 0)
    st.write(f"**Syntactic Ambiguity:** {syntactic_score:.1f}%")
    st.progress(syntactic_score / 100)
    st.caption(f"{syntactic_data.get('count', 0)} out of {syntactic_data.get('total_sentences', 0)} sentences")
    
    # Semantic Ambiguity
    semantic_data = results.get('semantic', {})
    semantic_score = semantic_data.get('percentage', 0)
    st.write(f"**Semantic Ambiguity:** {semantic_score:.1f}%")
    st.progress(semantic_score / 100)
    st.caption(f"{semantic_data.get('count', 0)} out of {semantic_data.get('total_sentences', 0)} sentences")
    
    # Detailed Analysis in Expandable Sections
    st.markdown("---")
    st.subheader("📋 Detailed Analysis")
    
    for ambiguity_type, data in results.items():
        with st.expander(f"🔎 {ambiguity_type.title()} Ambiguity Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(f"Score", f"{data.get('percentage', 0):.1f}%")
                st.metric("Ambiguous Sentences", f"{data.get('count', 0)} / {data.get('total_sentences', 0)}")
            
            with col2:
                if data.get('flagged_sentences'):
                    st.write("**Flagged Sentences:**")
                    for i, sent in enumerate(data.get('flagged_sentences', [])[:3]):
                        st.write(f"{i+1}. {sent}")
                
                if len(data.get('flagged_sentences', [])) > 3:
                    st.info(f"... and {len(data.get('flagged_sentences', [])) - 3} more sentences")
            
            # Show detailed analysis for pragmatic ambiguity
            if ambiguity_type == 'pragmatic' and data.get('detailed_analysis'):
                st.markdown("---")
                st.write("**Detailed Rule Analysis:**")
                for analysis in data.get('detailed_analysis', [])[:2]:
                    with st.expander(f"Sentence {analysis['sentence_number']}: {analysis['sentence'][:80]}..."):
                        st.write(f"**Rules Triggered:** {', '.join(analysis['rules_triggered'])}")
                        st.write(f"**Features Found:** {', '.join(analysis['features_found'])}")
            
            # Show detailed analysis for syntactic ambiguity
            if ambiguity_type == 'syntactic' and data.get('detailed_analysis'):
                st.markdown("---")
                st.write("**Detailed Clause Analysis:**")
                for analysis in data.get('detailed_analysis', [])[:2]:
                    with st.expander(f"Sentence {analysis['sentence_number']}: {analysis['sentence'][:80]}..."):
                        st.write(f"**Clause Count:** {analysis['clause_count']} (Threshold: {analysis['threshold']})")
                        if analysis['clause_types_found']:
                            st.write("**Clause Types Found:**")
                            for clause in analysis['clause_types_found'][:5]:
                                st.write(f"• {clause}")
            
            # Show detailed analysis for semantic ambiguity
            if ambiguity_type == 'semantic' and data.get('detailed_analysis'):
                st.markdown("---")
                st.write("**Detailed Word Analysis:**")
                for analysis in data.get('detailed_analysis', [])[:2]:
                    with st.expander(f"Sentence {analysis['sentence_number']}: {analysis['sentence'][:80]}..."):
                        st.write(f"**Ambiguous Words Found:** {analysis['total_ambiguous_words']}")
                        for word_info in analysis['ambiguous_words'][:3]:  # Show first 3 ambiguous words
                            st.write(f"• **{word_info['word']}** - {word_info['meanings_count']} meanings")
                            st.caption(f"Sample meanings: {', '.join(word_info['meanings'][:2])}")

# Footer
st.markdown("---")
st.caption("Financial Article Ambiguity Analyzer • Focus on Ambiguity Quantification")