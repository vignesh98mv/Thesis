import re
from datetime import timedelta
import sys
import io, os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "Outputs" / "3-final-phase"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LABEL_PATH = BASE_DIR.parent / "Outputs" / "3-final-phase" / "Labels"
LABEL_PATH.mkdir(parents=True, exist_ok=True)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------

def parse_time(t):
    """Parse time in mm:ss or hh:mm:ss format → timedelta"""
    parts = t.strip().split(':')
    if len(parts) == 2:
        m, s = map(int, parts)
        return timedelta(minutes=m, seconds=s)
    elif len(parts) == 3:
        h, m, s = map(int, parts)
        return timedelta(hours=h, minutes=m, seconds=s)
    return timedelta(0)


def parse_line(line):
    """Parse a line from the log file (handles new format with additional numeric field)."""
    pattern = (
        r"\[(.*?)\s*→\s*(.*?)\]\s*:\s*\[(.*?)\]\s*:\s*\[([\d.]+)\]\s*:\s*([A-Za-z]+)"
        r"\s*\(Average Confidence:\s*([0-9.]+)\)"
    )
    match = re.search(pattern, line)
    if not match:
        return None

    start, end, duration, numeric, phase, conf = match.groups()
    start_time = parse_time(start)
    end_time = parse_time(end)
    
    # Calculate duration as end - start instead of using the duration field
    calculated_duration = end_time - start_time

    confidence = float(conf) if conf is not None else None

    return {
        "start": start_time,
        "end": end_time,
        "duration": calculated_duration,
        "numeric": float(numeric),
        "phase": phase.strip(),
        "confidence": confidence
    }

def format_timedelta(td):
    """Format timedelta as HH:MM:SS or MM:SS if short"""
    total_seconds = int(td.total_seconds())
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02}:{m:02}:{s:02}"
    else:
        return f"{m:02}:{s:02}"


def time_to_seconds(t):
    parts = t.split(':')
    parts = [int(p) for p in parts]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        raise ValueError(f"Invalid time format: {t}")
    return h * 3600 + m * 60 + s

# -----------------------------------------------------------
# Main Phase Processing Logic
# -----------------------------------------------------------

def process_flight_phases(input_file, output_file):
    
    # Define phase order and limits
    phase_order = ["Preflight", "TAXI", "Takeoff", "Climb", "Cruise",
                   "Descent", "Approach", "Landing", "TAXI", "Parking"]

    min_limits = [
        "05:00", "00:30", "00:45", "10:00", "08:00",
        "05:00", "03:00", "00:30", "02:00", "02:00"
    ]
    max_limits = [
        "22:10", "30:50", "03:45", "40:00", "02:20:00",
        "40:00", "30:00", "03:30", "20:00", "20:00"
    ]

    min_limits = [parse_time(t) for t in min_limits]
    max_limits = [parse_time(t) for t in max_limits]

    with open(input_file, 'r', encoding="utf-8") as f:
        raw_lines = f.readlines()
        # Parse each line, ignoring lines that don't match
        lines = [parse_line(line) for line in raw_lines if parse_line(line)]

    if not lines:
        print("No valid data found in input file.")
        return "", []  # Return empty results

    phases, durations, start_times, end_times = [], [], [], []

    # Initialize with first line
    initialization = lines[0]
    phase = initialization["phase"]
    dur = initialization["duration"]
    start = initialization["start"]
    end = initialization["end"]
    
    current_phase = None
    current_phase_start_time = initialization["start"]
    current_phase_end_time = initialization["end"]
    current_phase_duration = current_phase_end_time - current_phase_start_time

    # -------------------------------------------------------
    # Loop through parsed entries
    # -------------------------------------------------------
    i = 0
    iteration = 0
    while i < len(lines):
        entry = lines[i]
        phase = entry["phase"]
        dur = entry["duration"]
        start = entry["start"]
        end = entry["end"]
        
        total_duration = end - current_phase_start_time

        if current_phase is None:
            # Code for finding the initial phase
            if phase == "CONVERSATION":
                # Extend conversation phase
                current_phase_end_time = end
                current_phase_duration = total_duration
                # i += 1
            else:
                if dur > timedelta(seconds=28):
                    if phase in phase_order:
                        idx = phase_order.index(phase)
                        
                        # Check if accumulated time is within allowed limits for this phase
                        if total_duration <= max_limits[idx]:
                            # Use accumulated time for the new phase
                            current_phase = phase
                            current_phase_end_time = end
                            current_phase_duration = total_duration
                            print(f"Initial phase found: {phase} at {format_timedelta(current_phase_start_time)} (using accumulated time: {format_timedelta(total_duration)})")
                        else:
                            # If accumulated time exceeds max limit, save as CONVERSATION and start fresh
                            phases.append("CONVERSATION")
                            durations.append(total_duration)
                            start_times.append(current_phase_start_time)
                            end_times.append(current_phase_end_time)
                            print(f"Saved CONVERSATION phase (exceeded limit): {format_timedelta(total_duration)}")
                            
                            # Start fresh with the new phase
                            current_phase = phase
                            current_phase_start_time = start
                            current_phase_end_time = end
                            current_phase_duration = dur
                            print(f"Initial phase found: {phase} at {format_timedelta(start)} (exceeded limit, starting fresh)")
                    else:
                        # Phase not in phase_order, extend as CONVERSATION
                        current_phase_end_time = end
                        current_phase_duration = total_duration
                        i += 1
                else:
                    # Duration too short, extend as CONVERSATION
                    current_phase_end_time = end
                    current_phase_duration = total_duration
                    i += 1
        else:
            # Code after finding the initial phase
            if current_phase == "Preflight":
                j = i
                while(j < i + 5 and j < len(lines)):
                    candidate = lines[j]
                    if candidate["phase"] == "Preflight":
                        candidate_duration = candidate["duration"]
                        phase_idx = phase_order.index(current_phase)
                        preflight_total_duration = candidate["end"] - current_phase_start_time
                        current_phase_end_time = candidate["end"]
                        current_phase_duration = preflight_total_duration
                        total_duration = preflight_total_duration
                        iteration = j
                        i = j
                    j += 1
                    
                i = iteration + 1
                entry = lines[i]
                phase = entry["phase"]
                dur = entry["duration"]
                start = entry["start"]
                end = entry["end"]
                
                phases.append(current_phase)
                durations.append(current_phase_duration)
                start_times.append(current_phase_start_time)
                end_times.append(current_phase_end_time)
                current_phase = "TAXI"
                current_phase_start_time = start
                current_phase_end_time = end
                current_phase_duration = dur

            if phase == current_phase:
                current_phase_end_time = end
                current_phase_duration = total_duration
            elif phase == "CONVERSATION":
                if dur < timedelta(minutes=4):
                    # Extend phase through short conversation
                    current_phase_end_time = end
                    current_phase_duration = total_duration
                else:
                    if(i + 1) == len(lines):
                        nxt_entry = lines[i]
                        nxt_phase = nxt_entry["phase"]
                    else:
                        nxt_entry = lines[i + 1]
                        nxt_phase = nxt_entry["phase"]

                    if current_phase == "Climb" and nxt_phase != "Climb":
                        # Append current phase and change to Cruise
                        phases.append(current_phase)
                        durations.append(current_phase_duration)
                        start_times.append(current_phase_start_time)
                        end_times.append(current_phase_end_time)
                        
                        # Change of phase
                        current_phase = "Cruise"
                        current_phase_start_time = start
                        current_phase_end_time = end
                        current_phase_duration = dur
                    elif current_phase in ["Descent", "Cruise", "Climb"]:
                        # Extend phase
                        current_phase_end_time = end
                        current_phase_duration = total_duration
                    else:
                        # Very rare case
                        print("Not able to detect the phases properly - failed because there is huge amount of conversation.")
                        break
            else:
                # Check if this is the next expected phase
                if current_phase in phase_order:
                    # Determine next expected phase
                    idx = phase_order.index(current_phase)
                    
                    # Special handling for TAXI phase
                    if current_phase == "TAXI":
                        # Check if this is the first TAXI (after Preflight) or second TAXI (after Landing)
                        if phases and phases[-1] == "Landing":
                            # This is the second TAXI, next should be Parking
                            next_expected = "Parking"
                        else:
                            # This is the first TAXI, next should be Takeoff
                            next_expected = phase_order[idx + 1] if idx + 1 < len(phase_order) else None
                    else:
                        # For other phases, use normal order
                        next_expected = phase_order[idx + 1] if idx + 1 < len(phase_order) else None
                    
                    if phase == next_expected and (next_expected in ["Takeoff", "Landing"] or dur >= timedelta(seconds=40)):
                        if next_expected in ["Landing", "Takeoff"]:
                            # Special handling for Landing/Takeoff
                            candidate_phases = []
                            
                            # Look at next 6 phases
                            for j in range(i, min(i + 6, len(lines))):
                                candidate = lines[j]
                                if candidate["phase"] == next_expected:
                                    candidate_duration = candidate["duration"]
                                    
                                    # Check if duration meets minimum requirement
                                    phase_idx = phase_order.index(phase)
                                    if candidate_duration > min_limits[phase_idx]:
                                        # Calculate how close it is to max limit
                                        time_diff = abs(candidate_duration.total_seconds() - max_limits[phase_idx].total_seconds())
                                        candidate_phases.append({
                                            "index": j,
                                            "phase": candidate["phase"],
                                            "duration": candidate_duration,
                                            "start": candidate["start"],
                                            "end": candidate["end"],
                                            "time_diff": time_diff
                                        })
                                        print(f"phases candidate found: {candidate['phase']} with duration {format_timedelta(candidate_duration)}")
                            
                            if candidate_phases:
                                # Find the candidate closest to max limit
                                best_candidate = min(candidate_phases, key=lambda x: x["time_diff"])
                                
                                # Update current phase end to the start of best candidate
                                current_phase_end_time = best_candidate["start"]
                                current_phase_duration = current_phase_end_time - current_phase_start_time
                                
                                # Save current phase
                                phases.append(current_phase)
                                durations.append(current_phase_duration)
                                start_times.append(current_phase_start_time)
                                end_times.append(current_phase_end_time)
                                
                                # Set the best candidate as new current phase
                                current_phase = best_candidate["phase"]
                                current_phase_start_time = best_candidate["start"]
                                current_phase_end_time = best_candidate["end"]
                                current_phase_duration = best_candidate["duration"]
                                
                                # Skip ahead to the chosen candidate
                                i = best_candidate["index"]
                            else:
                                # No suitable candidate found, use normal transition
                                phases.append(current_phase)
                                durations.append(current_phase_duration)
                                start_times.append(current_phase_start_time)
                                end_times.append(current_phase_end_time)
                                
                                current_phase = phase
                                current_phase_start_time = start
                                current_phase_end_time = end
                                current_phase_duration = dur
                        else:
                            # Normal phase transition for other phases
                            phases.append(current_phase)
                            durations.append(current_phase_duration)
                            start_times.append(current_phase_start_time)
                            end_times.append(current_phase_end_time)
                            
                            current_phase = phase
                            current_phase_start_time = start
                            current_phase_end_time = end
                            current_phase_duration = dur
                    else:
                        # Extend current phase
                        current_phase_end_time = end
                        current_phase_duration = total_duration
                else:
                    # Current phase not in phase_order, extend
                    current_phase_end_time = end
                    current_phase_duration = total_duration
        
        i += 1

    # Append last phase
    if current_phase and current_phase_duration > timedelta(0):
        phases.append(current_phase)
        durations.append(current_phase_duration)
        start_times.append(current_phase_start_time)
        end_times.append(current_phase_end_time)

    # -------------------------------------------------------
    # Generate output
    # -------------------------------------------------------
    output_lines = []
    phase_results = []  # To store formatted phase results
    
    # Build the flight phase analysis results
    separator = "=" * 80
    phase_results.append(separator)
    phase_results.append("FLIGHT PHASE ANALYSIS RESULTS")
    phase_results.append(separator)
    
    for i in range(len(phases)):
        line = (
            f"[{format_timedelta(start_times[i])} → {format_timedelta(end_times[i])}] : "
            f"[{format_timedelta(durations[i])}] : {phases[i]:<10}"
        )
        phase_results.append(line)
        output_lines.append(line)

    with open(output_file, 'w', encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    # For creating the Audacity labels
    Label_Line = []
    label_results = []
    label_results.append(f"Audacity labels:\n")

    for j in range(len(phases)):
        line1 = (
            f"{start_times[j].total_seconds():.6f}\t{start_times[j].total_seconds():.6f}\t"
            f"{phases[j]:<10}"
        )
        label_results.append(line1)
        Label_Line.append(line1)

    # Create label path
    Label_path = os.path.join(LABEL_PATH, f"{os.path.splitext(os.path.basename(input_file))[0].replace('_merged_window', '')}_Label.txt")
    
    with open(Label_path, 'w', encoding="utf-8") as f:
        f.write("\n".join(Label_Line))

    # Print summary
    print(f"\nOutput written to: {output_file}")
    print(f"Label file written to: {Label_path}")
    print(f"Total phases detected: {len(phases)}")

    # Return both the label path AND the phase results
    # return str(Label_path), phase_results, label_results
    return str(Label_path), phase_results

# -----------------------------------------------------------
# Run script
# -----------------------------------------------------------
# ... [keep all the code above unchanged until the main execution block] ...

if __name__ == "__main__":
    import sys

    # Default paths
    default_input = r"D:\Coburg\Thesis\flight phase detection\Outputs\2-merged-predcition\Whisper_OP_A320_düs_pra_merged_window.txt"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        base_name = os.path.splitext(os.path.basename(sys.argv[1]))[0].replace("_merged_window", "")
    else: 
        input_file = default_input
        base_name = os.path.splitext(os.path.basename(default_input))[0].replace("_merged_window", "")

    output_file = os.path.join(OUTPUT_DIR, f"{base_name}_final_phase.txt")
    
    # Process and get results (now returns only label_path and phase_results)
    label_path, phase_results = process_flight_phases(input_file=input_file, output_file=output_file)
    
    # Print everything that should be captured by the GUI
    print("\n" + "="*80)
    print("FINAL OUTPUT SUMMARY")
    # print("="*80)
    
    # Print phase results - these will be captured by the GUI
    for line in phase_results:
        print(line)
    
    # Print a separator so GUI knows where phase results end
    print("\n" + "="*80)
    print("END OF PHASE RESULTS")
    print("="*80)
    
    # The label path should be the VERY LAST thing printed
    print(f"\nLabel Path: {label_path}")
    print(label_path)  # Keep this as the last line for the main pipeline to capture