import re
from collections import defaultdict
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "Outputs" / "2-merged-predcition"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_line(line):
    """Parse a single line and extract components"""
    line = line.strip()
    if not line:
        return None
    
    # Pattern to match the format: [time] phase confidence text
    pattern = r'^\[([^\]]+)\s*→\s*([^\]]+)\]\s+([^\t]+)\t+([\d.]+)\t+(.*)$'
    match = re.match(pattern, line)
    
    if match:
        start_time, end_time, phase, confidence, sentence = match.groups()
        phase = phase.strip()
        
        original_phase = phase
        normalized_phase = phase
        
        # Handle special cases
        if phase.upper() in ['NOT IDENTIFIABLE', 'NOT_IDENTIFIABLE']:
            normalized_phase = 'NOT_IDENTIFIABLE'
        elif phase.upper() in ['CONVERSATION']:
            normalized_phase = 'CONVERSATION'
        elif phase.upper() in ['TAXI-IN', 'TAXI-OUT', 'TAXI']:
            normalized_phase = 'TAXI'
        else:
           
            phase_lower = phase.lower()
            if phase_lower == 'preflight':
                normalized_phase = 'Preflight'
            elif phase_lower == 'takeoff':
                normalized_phase = 'Takeoff'
            elif phase_lower == 'climb':
                normalized_phase = 'Climb'
            elif phase_lower == 'cruise':
                normalized_phase = 'Cruise'
            elif phase_lower == 'descent':
                normalized_phase = 'Descent'
            elif phase_lower == 'approach':
                normalized_phase = 'Approach'
            elif phase_lower == 'landing':
                normalized_phase = 'Landing'
            elif phase_lower == 'parking':
                normalized_phase = 'Parking'
            else:
                
                normalized_phase = phase
        
        # Convert to HH:MM:SS format
        start_hms = normalize_time_to_hms(start_time.strip())
        end_hms = normalize_time_to_hms(end_time.strip())
        
        return {
            'start_time': start_hms,
            'end_time': end_hms,
            'phase': normalized_phase,
            'confidence': float(confidence),
            'sentence': sentence.strip(),
            'total_minutes': get_total_minutes_from_hms(start_hms)
        }
    
    return None

def normalize_time_to_hms(time_str):
    """Convert time to HH:MM:SS format, handling both MM:SS and HH:MM:SS"""
    # Remove milliseconds if present
    time_str = time_str.split('.')[0]
    
    parts = time_str.split(':')
    
    if len(parts) == 3: 
        h, m, s = map(int, parts)
        if h > 23:
            return f"00:{h:02d}:{m:02d}" 
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    elif len(parts) == 2:
        first, second = map(int, parts)
        
        if first > 59:
            return f"{first:02d}:{second:02d}:00"
        else:
            return f"00:{first:02d}:{second:02d}"
    
    elif len(parts) == 1: 
        value = int(parts[0])
        if value > 59:
            return f"{value:02d}:00:00"  
        else:
            return f"00:00:{value:02d}"  
    
    return "00:00:00"

def get_total_minutes_from_hms(hms_time):
    """Get total minutes from HH:MM:SS format (hours*60 + minutes)"""
    h, m, s = map(int, hms_time.split(':'))
    return h * 60 + m

def time_to_seconds(hms_time):
    """Convert HH:MM:SS to total seconds"""
    h, m, s = map(int, hms_time.split(':'))
    return h * 3600 + m * 60 + s

def create_time_windows(parsed_data):
    """Create 1-minute windows based on start time total minutes"""
    if not parsed_data:
        return []
    
    # Sort by start time to ensure chronological order
    parsed_data.sort(key=lambda x: time_to_seconds(x['start_time']))
    
    windows = []
    current_window = []
    window_start_total_minutes = None
    
    for entry in parsed_data:
        if not current_window:
            # Start new window with first entry
            current_window.append(entry)
            window_start_total_minutes = entry['total_minutes']
        else:
            current_total_minutes = entry['total_minutes']
            
            if current_total_minutes <= window_start_total_minutes:
                current_window.append(entry)
            elif current_total_minutes == window_start_total_minutes + 1:
                windows.append(current_window)
                current_window = [entry]
                window_start_total_minutes = current_total_minutes
            else:
                windows.append(current_window)
                current_window = [entry]
                window_start_total_minutes = current_total_minutes
    
    # Add the last window if it exists
    if current_window:
        windows.append(current_window)
    
    return windows

def process_window_entries(entries):
    """Process a window of entries to determine dominant phase and confidence"""
    if not entries:
        return None, 0.0
    
    # Check for small window (≤3 entries)
    if len(entries) <= 3:
        return process_small_window(entries)
    
    # First, check for Preflight special rule
    total_entries = len(entries)
    
    # Count CONVERSATION + NOT_IDENTIFIABLE for percentage calculation
    non_flight_count = 0
    conversation_confidences = []
    
    # Count Preflight separately
    preflight_count = 0
    preflight_confidences = []
    
    for entry in entries:
        if entry['phase'] in ['CONVERSATION', 'NOT_IDENTIFIABLE']:
            non_flight_count += 1
            if entry['phase'] == 'CONVERSATION':
                conversation_confidences.append(entry['confidence'])
        elif entry['phase'] == 'Preflight':
            preflight_count += 1
            preflight_confidences.append(entry['confidence'])
    
    if preflight_count >= 2:
        flight_phase_counts = defaultdict(int)
        flight_phase_confidences = defaultdict(list)
        
        for entry in entries:
            phase = entry['phase']
            if phase not in ['CONVERSATION', 'NOT_IDENTIFIABLE']:
                flight_phase_counts[phase] += 1
                flight_phase_confidences[phase].append(entry['confidence'])
        
        # Find all flight phases that appear at least twice
        valid_flight_phases = []
        for phase, count in flight_phase_counts.items():
            if count >= 2:
                avg_confidence = sum(flight_phase_confidences[phase]) / len(flight_phase_confidences[phase])
                valid_flight_phases.append((phase, count, avg_confidence))
        
        if valid_flight_phases:
            valid_flight_phases.sort(key=lambda x: (x[1], x[2]), reverse=True)
            return valid_flight_phases[0][0], valid_flight_phases[0][2]
    
    if non_flight_count / total_entries >= 0.77:  # Changed from 0.8 to 0.77
        if conversation_confidences:
            avg_confidence = sum(conversation_confidences) / len(conversation_confidences)
        else:
            avg_confidence = 0.0
        return 'CONVERSATION', avg_confidence
    
    # Count flight phases that appear at least twice (excluding CONVERSATION/NOT_IDENTIFIABLE)
    flight_phase_counts = defaultdict(int)
    flight_phase_confidences = defaultdict(list)
    
    for entry in entries:
        phase = entry['phase']
        if phase not in ['CONVERSATION', 'NOT_IDENTIFIABLE']:
            flight_phase_counts[phase] += 1
            flight_phase_confidences[phase].append(entry['confidence'])
    
    # Find flight phases that appear at least twice
    valid_flight_phases = []
    for phase, count in flight_phase_counts.items():
        if count >= 2:
            avg_confidence = sum(flight_phase_confidences[phase]) / len(flight_phase_confidences[phase])
            valid_flight_phases.append((phase, count, avg_confidence))
    
    if valid_flight_phases:
        # Sort by count (highest first), then by average confidence (highest first)
        valid_flight_phases.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return valid_flight_phases[0][0], valid_flight_phases[0][2]
    
    # If no flight phase appears at least twice, default to CONVERSATION
    conv_confidences = [e['confidence'] for e in entries if e['phase'] == 'CONVERSATION']
    if conv_confidences:
        avg_confidence = sum(conv_confidences) / len(conv_confidences)
        return 'CONVERSATION', avg_confidence
    else:
        # If no CONVERSATION entries at all, use overall average
        avg_confidence = sum(e['confidence'] for e in entries) / len(entries)
        return 'CONVERSATION', avg_confidence

def process_small_window(entries):
    """Process windows with 3 or fewer entries"""
    if not entries:
        return None, 0.0
    
    if len(entries) == 1:
        return entries[0]['phase'], entries[0]['confidence']
    
    # Count phase occurrences
    phase_counts = defaultdict(int)
    phase_confidences = defaultdict(list)
    
    for entry in entries:
        phase = entry['phase']
        phase_counts[phase] += 1
        phase_confidences[phase].append(entry['confidence'])
    
    # Check if any phase appears at least twice
    for phase, count in phase_counts.items():
        if count >= 2:
            avg_confidence = sum(phase_confidences[phase]) / len(phase_confidences[phase])
            return phase, avg_confidence
    
    # All phases are different, take highest confidence
    highest_conf_entry = max(entries, key=lambda x: x['confidence'])
    return highest_conf_entry['phase'], highest_conf_entry['confidence']

def format_duration(start_time, end_time):
    """Calculate and format duration between start and end times"""
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)
    
    duration_seconds = end_seconds - start_seconds
    if duration_seconds < 0:
        return "[00:00]"
    
    if duration_seconds < 3600:
        minutes = duration_seconds // 60
        seconds = duration_seconds % 60
        return f"[{minutes:02d}:{seconds:02d}]"
    else:
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        seconds = duration_seconds % 60
        return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"

def format_time_display(start_time, end_time):
    """Format time for display in output"""
    return f"[{start_time} → {end_time}]"

def calculate_confidence_seconds(start_time, end_time, confidence):
    """Calculate duration in seconds multiplied by confidence"""
    duration_seconds = time_to_seconds(end_time) - time_to_seconds(start_time)
    return duration_seconds * confidence

def merge_consecutive_windows(window_results):
    """Merge consecutive windows with the same phase"""
    if not window_results:
        return []
    
    merged_results = []
    current_phase = window_results[0]['phase']
    current_start = window_results[0]['start_time']
    current_end = window_results[0]['end_time']
    current_confidence = window_results[0]['confidence']
    
    for i in range(1, len(window_results)):
        window = window_results[i]
        
        if window['phase'] == current_phase:
            current_end = window['end_time']
        else:
            merged_results.append({
                'start_time': current_start,
                'end_time': current_end,
                'phase': current_phase,
                'confidence': current_confidence
            })
            
            current_phase = window['phase']
            current_start = window['start_time']
            current_end = window['end_time']
            current_confidence = window['confidence']
    
    merged_results.append({
        'start_time': current_start,
        'end_time': current_end,
        'phase': current_phase,
        'confidence': current_confidence
    })
    
    return merged_results

def merge_short_conversations(results):
    """Merge CONVERSATION phases that are less than 1 minute into adjacent segments"""
    if not results:
        return []
    
    merged = []
    i = 0
    
    while i < len(results):
        current = results[i]
        
        # Check if current is a short CONVERSATION (less than 1 minute)
        if current['phase'] == 'CONVERSATION':
            duration_seconds = time_to_seconds(current['end_time']) - time_to_seconds(current['start_time'])
            
            if duration_seconds < 60:  # Less than 1 minute
                # Case 1: Has previous segment (not first segment)
                if i > 0:
                    prev = merged[-1] if merged else results[i-1]
                    
                    # Case 1a: Has next segment and next segment has same phase as previous
                    if i + 1 < len(results) and results[i+1]['phase'] == prev['phase']:
                        next_seg = results[i+1]
                        
                        # Calculate weighted average confidence for all three segments
                        prev_duration = time_to_seconds(prev['end_time']) - time_to_seconds(prev['start_time'])
                        current_duration = duration_seconds
                        next_duration = time_to_seconds(next_seg['end_time']) - time_to_seconds(next_seg['start_time'])
                        total_duration = prev_duration + current_duration + next_duration
                        
                        weighted_confidence = (
                            (prev['confidence'] * prev_duration) +
                            (current['confidence'] * current_duration) +
                            (next_seg['confidence'] * next_duration)
                        ) / total_duration
                        
                        # Create merged segment
                        merged_segment = {
                            'start_time': prev['start_time'],
                            'end_time': next_seg['end_time'],
                            'phase': prev['phase'],
                            'confidence': weighted_confidence
                        }
                        
                        # Replace last segment in merged list or append
                        if merged and merged[-1] == prev:
                            merged[-1] = merged_segment
                        else:
                            merged.append(merged_segment)
                        
                        # Skip next segment since we merged it
                        i += 2
                        continue
                    
                    # Case 1b: Merge with previous segment only
                    else:
                        # Calculate weighted average confidence
                        prev_duration = time_to_seconds(prev['end_time']) - time_to_seconds(prev['start_time'])
                        current_duration = duration_seconds
                        total_duration = prev_duration + current_duration
                        
                        weighted_confidence = (
                            (prev['confidence'] * prev_duration) +
                            (current['confidence'] * current_duration)
                        ) / total_duration
                        
                        # Create merged segment
                        merged_segment = {
                            'start_time': prev['start_time'],
                            'end_time': current['end_time'],
                            'phase': prev['phase'],
                            'confidence': weighted_confidence
                        }
                        
                        # Replace last segment in merged list
                        if merged and merged[-1] == prev:
                            merged[-1] = merged_segment
                        else:
                            merged.append(merged_segment)
                        
                        i += 1
                        continue
                
                # Case 2: No previous segment (first segment is short CONVERSATION)
                # Merge with next segment if it exists
                elif i + 1 < len(results):
                    next_seg = results[i+1]
                    
                    # Calculate weighted average confidence
                    current_duration = duration_seconds
                    next_duration = time_to_seconds(next_seg['end_time']) - time_to_seconds(next_seg['start_time'])
                    total_duration = current_duration + next_duration
                    
                    weighted_confidence = (
                        (current['confidence'] * current_duration) +
                        (next_seg['confidence'] * next_duration)
                    ) / total_duration
                    
                    # Create merged segment
                    merged_segment = {
                        'start_time': current['start_time'],
                        'end_time': next_seg['end_time'],
                        'phase': next_seg['phase'],
                        'confidence': weighted_confidence
                    }
                    
                    merged.append(merged_segment)
                    i += 2
                    continue
        
        # Not a short CONVERSATION, keep as is
        if not merged or results[i] != merged[-1]:
            merged.append(results[i])
        i += 1
    
    return merged


def process_phase_file(input_filename, output_filename):
    """Main function to process the file with 1-minute time windows"""
    
    # Read and parse all lines
    parsed_data = []
    with open(input_filename, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            parsed_line = parse_line(line)
            if parsed_line:
                parsed_data.append(parsed_line)
    
    print(f"Total lines parsed: {len(parsed_data)}")
    
    if len(parsed_data) == 0:
        print("No data found in the file!")
        return []
    
    # Create 1-minute windows
    windows = create_time_windows(parsed_data)
    print(f"Number of 1-minute windows: {len(windows)}")
    
    # Process each window
    window_results = []
    
    for i, window_entries in enumerate(windows):
        if not window_entries:
            continue
        
        # Determine window time range
        window_start = window_entries[0]['start_time']
        window_end = window_entries[-1]['end_time']
        
        # Process window to get dominant phase and confidence
        dominant_phase, avg_confidence = process_window_entries(window_entries)
        
        if dominant_phase:
            window_results.append({
                'start_time': window_start,
                'end_time': window_end,
                'phase': dominant_phase,
                'confidence': avg_confidence
            })
            print(f"Window {i+1}: {window_start} to {window_end}, Phase: {dominant_phase}, Confidence: {avg_confidence:.3f}, Entries: {len(window_entries)}")
    
    # Merge consecutive windows with same phase
    merged_results = merge_consecutive_windows(window_results)
    
    # NEW: Merge short CONVERSATION segments (less than 1 minute)
    merged_results = merge_short_conversations(merged_results)
    
    # Write results to file
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.write("1-Minute Time Window Analysis Results\n")
        outfile.write("=" * 80 + "\n")
        outfile.write("Window boundary: when start total minutes increases by 1\n")
        outfile.write("No overlapping windows\n")
        outfile.write("=" * 80 + "\n\n")
        
        for result in merged_results:
            time_range = format_time_display(result['start_time'], result['end_time'])
            duration = format_duration(result['start_time'], result['end_time'])
            confidence_seconds = calculate_confidence_seconds(result['start_time'], result['end_time'], result['confidence'])
            
            result_line = f"{time_range} : {duration} : [{confidence_seconds:.3f}] : {result['phase']}\t(Average Confidence: {result['confidence']:.3f})\n"
            outfile.write(result_line)
    
    return merged_results

def main():
    import sys
    import io

    # Fix encoding for Windows
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Handle command-line arguments
    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
    else:
        input_filename = (
            r"D:\Coburg\Thesis\flight phase detection\Outputs\1-bert-prediction\Whisper_OP_A320_düs_pra_bert_prediction.txt"
        )
    
    base_name = os.path.splitext(os.path.basename(input_filename))[0].replace("_bert_prediction", "")
    output_filename = os.path.join(OUTPUT_DIR, f"{base_name}_merged_window.txt")

    try:
        print("Starting 1-minute time window analysis...")
        print("Window boundary: when start total minutes increases by 1")
        print("=" * 60)

        results = process_phase_file(input_filename, output_filename)

        print(f"\nAnalysis complete!")
        print(f"Results saved to: {output_filename}")
        print(f"Total segments after merging: {len(results)}")

        # Show final results
        print("\nFinal Results (Merged):")
        print("-" * 70)
        for result in results:
            time_range = format_time_display(result['start_time'], result['end_time'])
            duration = format_duration(result['start_time'], result['end_time'])
            confidence_seconds = calculate_confidence_seconds(result['start_time'], result['end_time'], result['confidence'])
            
            print(f"{time_range} : {duration} : [{confidence_seconds:.3f}] : {result['phase']}\t(Average Confidence: {result['confidence']:.3f})")
        
        # Print output filename for caller
        print(output_filename)

    except FileNotFoundError:
        print(f"Error: File '{input_filename}' not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()