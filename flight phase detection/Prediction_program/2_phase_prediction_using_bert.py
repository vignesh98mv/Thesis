import os
import sys

# Disable TensorFlow completely
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_TENSORFLOW'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*TensorFlow.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*tensorflow.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Force transformers to use PyTorch
os.environ['USE_TORCH'] = '1'

# Import numpy first to avoid conflicts
import numpy as np

print(f"Using NumPy version: {np.__version__}")

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import re
import os
from pathlib import Path

# =============================
# Configuration
# =============================
MIN_CONFIDENCE_THRESHOLD = 0.30  # Adjustable confidence threshold

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "Outputs" / "1-bert-prediction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase keywords for rule-based matching
PHASE_KEYWORDS = {
    "Preflight": [
    ],
    "Taxi": [
    ],
    "Takeoff": [
        "rotate" # v1 rotate", "V1.. Rotates", "V1.. Rotate"
    ],
    "Climb": [
    ],
    "Cruise": [
    ],
    "Descent": [
    ],
    "Approach": [
    ],
    "Landing": [
        "spoilers", "reverse green", "retard", "50, 40, 30,", "20, 10"
    ],
    "Parking": [
    ]
}

# =============================
# Load the model and tokenizer
# =============================
model_path = BASE_DIR.parent / "bert-model" / "flight_phase_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# =============================
# Load the label encoder
# =============================
label_encoder_path = os.path.join(model_path, "label_encoder.pkl")
label_encoder = joblib.load(label_encoder_path)

# =============================
# Rule-based phase detection
# =============================
def detect_phase_by_keywords(text):
    """
    Detect flight phase based on keywords with confidence 1.0
    Returns: (phase, confidence) or (None, 0) if no match
    """
    text_lower = text.lower()
    
    for phase, keywords in PHASE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                print(f"Keyword matched: '{keyword}' → {phase}")
                return phase, 1.0
    
    return None, 0

# =============================
# Enhanced prediction function with keyword matching and confidence threshold
# =============================
def predict_phase_with_confidence(text):
    """
    Predict flight phase for a single text with confidence score
    Returns: (predicted_phase, confidence_score)
    """
    # First, check for keyword matches
    keyword_phase, keyword_confidence = detect_phase_by_keywords(text)
    if keyword_confidence == 1.0:
        return keyword_phase, keyword_confidence
    
    # If no keyword match, use BERT model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]
        
        # Get predicted class and confidence
        predicted_class_id = logits.argmax().item()
        confidence = probabilities[predicted_class_id].item()
        
        # Apply confidence threshold
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            return "Not identifiable", confidence
        
        # Convert to label
        predicted_phase = label_encoder.inverse_transform([predicted_class_id])[0]
        
        # Map taxi-in and taxi-out to just "taxi"
        if predicted_phase in ["taxi-in", "taxi-out"]:
            predicted_phase = "Taxi"
        
        return predicted_phase, confidence

# =============================
# Improved function to read and parse conversation file
# =============================
def read_conversation_file(filename):
    conversations = []
    line_count = 0
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                    
                # Skip the first two lines (header lines)
                if line_num <= 2:
                    print(f"Skipping header line {line_num}: {line}")
                    continue
                
                # Only process lines that match the timestamp format: [MM:SS → MM:SS] or [HH:MM:SS → HH:MM:SS]
                timestamp_match = None
                
                # Format 1: [HH:MM:SS → HH:MM:SS]
                match = re.match(r'\[(\d{1,2}:\d{2}:\d{2})\s*→\s*(\d{1,2}:\d{2}:\d{2})\]\s*(.+)', line)
                if match:
                    timestamp_match = match
                
                # Format 2: [MM:SS → MM:SS]  
                if not timestamp_match:
                    match = re.match(r'\[(\d{1,2}:\d{2})\s*→\s*(\d{1,2}:\d{2})\]\s*(.+)', line)
                    if match:
                        timestamp_match = match
                
                # If line doesn't match the expected format, skip it
                if not timestamp_match:
                    print(f"Warning: Skipping line {line_num} (wrong format): {line[:60]}...")
                    continue
                
                # Process valid timestamp lines
                start_time = timestamp_match.group(1)
                end_time = timestamp_match.group(2)
                conversation_text = timestamp_match.group(3)
                conversations.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'original_line': line,
                    'text': conversation_text,
                    'line_number': line_num
                })
                line_count += 1
        
        print(f"Successfully parsed {line_count} valid conversation lines from file")
        return conversations
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

# =============================
# Combine consecutive duplicate sentences
# =============================
def combine_duplicate_sentences(conversations):
    """
    Combine consecutive lines with identical text
    Returns: List of conversations with duplicates combined
    """
    if not conversations:
        return []
    
    combined = []
    i = 0
    
    while i < len(conversations):
        current = conversations[i]
        j = i + 1
        
        # Find consecutive duplicates
        while j < len(conversations) and conversations[j]['text'] == current['text']:
            j += 1
        
        if j > i + 1:
            # Found duplicates, combine them
            combined_conversation = current.copy()
            combined_conversation['end_time'] = conversations[j-1]['end_time']
            combined_conversation['original_line'] = f"{current['original_line']} ... {conversations[j-1]['original_line']}"
            combined_conversation['line_numbers'] = list(range(i+1, j+1))
            combined.append(combined_conversation)
            print(f"Combined {j-i} duplicate lines: '{current['text'][:50]}...'")
        else:
            # No duplicates, keep as is
            current['line_numbers'] = [i+1]
            combined.append(current)
        
        i = j
    
    print(f"Combined {len(conversations)} lines into {len(combined)} unique conversations")
    return combined

# =============================
# Combine short sentences to meet minimum word requirement
# =============================
def combine_short_sentences(conversations, min_words=3):
    """
    Combine consecutive short sentences to meet minimum word requirement
    Returns: List of conversations with short sentences combined
    """
    if not conversations:
        return []
    
    combined = []
    i = 0
    
    while i < len(conversations):
        current = conversations[i]
        current_words = len(current['text'].split())
        
        # If current sentence meets minimum word requirement, keep it as is
        if current_words >= min_words:
            combined.append(current)
            i += 1
            continue
        
        # Current sentence is too short, try to combine with following sentences
        combined_text = current['text']
        combined_start = current['start_time']
        combined_end = current['end_time']
        combined_line_numbers = current.get('line_numbers', [current['line_number']])
        j = i + 1
        
        # Keep combining until we meet minimum word requirement or run out of sentences
        while j < len(conversations) and len(combined_text.split()) < min_words:
            next_conv = conversations[j]
            combined_text += ". " + next_conv['text']
            combined_end = next_conv['end_time']
            next_line_numbers = next_conv.get('line_numbers', [next_conv['line_number']])
            combined_line_numbers.extend(next_line_numbers)
            j += 1
        
        # Create combined conversation
        combined_conversation = current.copy()
        combined_conversation['text'] = combined_text
        combined_conversation['start_time'] = combined_start
        combined_conversation['end_time'] = combined_end
        combined_conversation['line_numbers'] = combined_line_numbers
        combined_conversation['original_line'] = f"Combined lines {combined_line_numbers}"
        
        combined.append(combined_conversation)
        
        # Skip the sentences we just combined
        i = j
    
    print(f"Combined short sentences: {len(conversations)} → {len(combined)}")
    return combined

# =============================
# Process file with preprocessing steps
# =============================
def process_conversation_file_enhanced(filename):
    """
    Enhanced processing with duplicate combination and short sentence handling
    """
    # Read original conversations
    conversations = read_conversation_file(filename)
    
    if not conversations:
        print("No conversations found or error reading file.")
        return []
    
    print(f"\nPreprocessing {len(conversations)} conversation lines...")
    
    # Step 1: Combine duplicate sentences
    conversations = combine_duplicate_sentences(conversations)
    
    # Step 2: Combine short sentences (less than 3 words)
    conversations = combine_short_sentences(conversations, min_words=3)
    
    print(f"After preprocessing: {len(conversations)} lines to process\n")
    
    results = []
    
    for conv in conversations:
        try:
            predicted_phase, confidence = predict_phase_with_confidence(conv['text'])
            
            result = {
                'line_numbers': conv.get('line_numbers', [conv['line_number']]),
                'start_time': conv['start_time'],
                'end_time': conv['end_time'],
                'text': conv['text'][:100] + "..." if len(conv['text']) > 100 else conv['text'],
                'full_text': conv['text'],
                'predicted_phase': predicted_phase,
                'confidence': confidence,
                'word_count': len(conv['text'].split())
            }
            results.append(result)
            
            # Print each result as it's processed
            confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            time_range = f"[{conv['start_time']} → {conv['end_time']}]"
            line_info = f"Lines {conv.get('line_numbers', [conv['line_number']])}"
            
            print(f"{line_info:15s} {time_range:20s}: "
                  f"{predicted_phase:15s} {confidence_color} {confidence:.3f} "
                  f"(words: {result['word_count']:2d}) - {conv['text'][:50]}...")
                  
        except Exception as e:
            line_info = conv.get('line_numbers', [conv['line_number']])
            print(f"Error processing lines {line_info}: {e}")
            continue
    
    return results

# =============================
# Save results in the required format
# =============================
def save_results_simple_format(results, input_filename):
    """
    Save results in simple format: "[start time -> end time] \t phase \t confidence \t sentence"
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Create output filename based on input filename
        base_name = os.path.splitext(os.path.basename(input_filename))[0]
        output_filename = os.path.join(OUTPUT_DIR, f"{base_name}_bert_prediction.txt")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            for result in results:
                time_range = f"[{result['start_time']} → {result['end_time']}]"
                phase = result['predicted_phase']
                confidence = result['confidence']
                sentence = result['full_text']
                
                # Write in the required format
                f.write(f'{time_range}\t{phase}\t\t{confidence:.3f}\t{sentence}\n')
        
        print(f"\nResults saved to: {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error saving results: {e}")
        return None

# =============================
# Display results in different formats
# =============================
def display_results_summary(results):
    """Display summary of predictions"""
    if not results:
        return
    
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    
    # Count predictions by phase
    phase_counts = {}
    total_confidence = 0
    
    for result in results:
        phase = result['predicted_phase']
        if phase not in phase_counts:
            phase_counts[phase] = {'count': 0, 'total_confidence': 0}
        phase_counts[phase]['count'] += 1
        phase_counts[phase]['total_confidence'] += result['confidence']
        total_confidence += result['confidence']
    
    # Print phase distribution
    print("\nPhase Distribution:")
    print("-" * 50)
    for phase, stats in sorted(phase_counts.items()):
        avg_confidence = stats['total_confidence'] / stats['count'] if stats['count'] > 0 else 0
        confidence_level = "High" if avg_confidence > 0.7 else "Medium" if avg_confidence > 0.5 else "Low"
        print(f"{phase:15s}: {stats['count']:3d} lines | Avg Confidence: {avg_confidence:.3f} ({confidence_level})")
    
    # Overall statistics
    avg_overall_confidence = total_confidence / len(results) if results else 0
    overall_level = "High" if avg_overall_confidence > 0.7 else "Medium" if avg_overall_confidence > 0.5 else "Low"
    print(f"\nOverall Statistics:")
    print(f"Total lines processed: {len(results)}")
    print(f"Average confidence: {avg_overall_confidence:.3f} ({overall_level})")
    print(f"Confidence threshold: {MIN_CONFIDENCE_THRESHOLD}")

def display_detailed_results(results):
    """Display detailed results in a formatted table"""
    if not results:
        return
    
    print("\n" + "="*120)
    print("DETAILED PREDICTION RESULTS")
    print("="*120)
    print(f"{'Lines':<15} {'Time Range':<25} {'Phase':<15} {'Conf':<6} {'Words':<5} Text Preview")
    print("-" * 120)
    
    for result in results:
        confidence_color = "🟢" if result['confidence'] > 0.8 else "🟡" if result['confidence'] > 0.6 else "🔴"
        time_range = f"[{result['start_time']} → {result['end_time']}]"
        line_info = str(result['line_numbers']) if len(result['line_numbers']) > 1 else f"[{result['line_numbers'][0]}]"
        print(f"{line_info:<15} {time_range:<25} {result['predicted_phase']:<15} "
              f"{confidence_color} {result['confidence']:.3f} {result['word_count']:5d} {result['text']}")

# =============================
# Test with single sentences
# =============================
def test_single_sentences():
    """Test the model with some example sentences"""
    test_sentences = [
        "Parking brake set",
        "Cleared for takeoff",
        "Request taxi to runway",
        "Maintain flight level 350",
        "Cleared to land",
        "Engine start approved",
        "Hello how are you",
    ]
    
    print("SINGLE SENTENCE TESTS")
    print("=" * 60)
    
    for sentence in test_sentences:
        phase, confidence = predict_phase_with_confidence(sentence)
        confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
        print(f"'{sentence}'")
        print(f"→ {phase} {confidence_color} (Confidence: {confidence:.3f})\n")

# =============================
# Main execution
# =============================
if __name__ == "__main__":
    import sys
    import io

    # --- Fix encoding for Windows consoles (UTF-8 emojis) ---
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print(f"Using confidence threshold: {MIN_CONFIDENCE_THRESHOLD}")
    print(f"Output directory: {OUTPUT_DIR}")

    # ---------------------------------------------
    # Handle command-line argument (conversation file)
    # ---------------------------------------------
    if len(sys.argv) > 1:
        conversation_file = sys.argv[1]
    else:
        conversation_file = r"C:\CRV\bert-offline-install\flight phase detection\Outputs\0-whipser-output\Whisper_OP_A320_düs_pra.txt"

    # ---------------------------------------------
    # Run predictions
    # ---------------------------------------------
    results = process_conversation_file_enhanced(conversation_file)

    if results:
        display_detailed_results(results)
        display_results_summary(results)
        output_file = save_results_simple_format(results, conversation_file)
        print(f"\nProcessing complete! Processed {len(results)} lines after preprocessing.")
        print(f"Results saved to: {output_file}")
        # Important: print this path as last line so the caller can read it
        print(output_file)
    else:
        print("No results to display.")
