import sys
import os
from pathlib import Path
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox
from PyQt5.uic import loadUi
import subprocess
import threading

class AudioWorkerThread(QtCore.QThread):
    """Thread for running the audio processing pipeline without freezing the UI"""
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str, list)  # Emits (output_path, phase_results)
    error = QtCore.pyqtSignal(str)
    
    def __init__(self, input_audio_path):
        super().__init__()
        self.input_audio_path = input_audio_path
        self.current_dir = Path(__file__).resolve().parent
        
        # Define script paths
        self.SCRIPTS = {
            "whisper": self.current_dir / "1_whisper_audio_to_text.py",
            "bert": self.current_dir / "2_phase_prediction_using_bert.py",
            "sliding": self.current_dir / "3_merging_windows.py",
            "final": self.current_dir / "4_final_flight_phases.py",
        }
    
    def run_script(self, script_path, *args):
        """Run a Python script and return the result"""
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        cmd = [sys.executable, str(script_path)]
        cmd.extend(map(str, args))
        
        self.progress.emit(f"Running {script_path.name}...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        
        if result.returncode != 0:
            error_msg = f"{script_path.name} failed with exit code {result.returncode}"
            if result.stderr:
                error_msg += f"\nError: {result.stderr[:500]}"  # Limit error message length
            raise Exception(error_msg)
        
        return result
    
    def extract_output_path(self, process_result, is_final_script=False):
        """Extract last non-empty line as output path, and for final script, also extract phase results"""
        lines = [
            line.strip()
            for line in process_result.stdout.splitlines()
            if line.strip()
        ]
        
        if not lines:
            raise Exception("No output path detected from script.")
        
        # For final script, capture everything
        if is_final_script:
            # The label path is still the last line
            label_path = Path(lines[-1]).resolve()
            
            # Return all non-empty lines (excluding the label path line itself)
            # This gives us the complete output to display
            all_output_lines = [line for line in process_result.stdout.splitlines() if line.strip()]
            
            return str(label_path), all_output_lines
        else:
            return Path(lines[-1]).resolve()
    
    def extract_phase_results(self, all_output_lines):
        """Extract and clean phase results from all output lines"""
        cleaned_results = []
        in_clean_section = False
        found_actual_phases = False
        
        for line in all_output_lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for the actual phase data section (after "FINAL OUTPUT SUMMARY")
            if "FINAL OUTPUT SUMMARY" in line:
                in_clean_section = True
                continue
            
            if in_clean_section:
                # Skip the "FINAL OUTPUT SUMMARY" separator
                if "=" * 80 in line and not found_actual_phases:
                    continue
                
                # Skip the duplicate "FLIGHT PHASE ANALYSIS RESULTS" header
                if "FLIGHT PHASE ANALYSIS RESULTS" in line and not found_actual_phases:
                    found_actual_phases = True
                    # We'll add our own cleaner header in the UI, so skip this one
                    continue
                
                # Stop at "END OF PHASE RESULTS"
                if "END OF PHASE RESULTS" in line:
                    break
                
                # Skip file write messages and total phases count
                if any(msg in line for msg in ["Output written to:", "Label file written to:", "Total phases detected:", "Label Path:"]):
                    continue
                
                # Add the line if it contains phase data or relevant info
                if line:
                    # Keep diagnostic messages
                    if "Initial phase found:" in line or "phases candidate found:" in line:
                        cleaned_results.append(line)
                    # Keep actual phase data lines
                    elif "[" in line and "→" in line and "]" in line:
                        cleaned_results.append(line)
        
        return cleaned_results
    
    def run(self):
        try:
            self.progress.emit("Starting audio processing pipeline...")
            
            # Check if input audio file exists
            input_file = Path(self.input_audio_path)
            if not input_file.exists():
                raise FileNotFoundError(f"Input audio file not found: {self.input_audio_path}")
            
            self.progress.emit(f"Processing audio file: {input_file.name}")
            
            # -------------------------------------------------------
            # 1. Whisper ASR (Audio to Text)
            # -------------------------------------------------------
            self.progress.emit("Running Whisper audio transcription...")
            whisper_result = self.run_script(
                self.SCRIPTS["whisper"],
                input_file
            )
            whisper_output = self.extract_output_path(whisper_result)
            self.progress.emit(f"Whisper output: {whisper_output}")
            
            # -------------------------------------------------------
            # 2. BERT Phase Prediction
            # -------------------------------------------------------
            self.progress.emit("Running BERT phase prediction...")
            bert_result = self.run_script(
                self.SCRIPTS["bert"],
                whisper_output
            )
            bert_output = self.extract_output_path(bert_result)
            self.progress.emit(f"BERT output: {bert_output}")
            
            # -------------------------------------------------------
            # 3. Sliding Window / Merge
            # -------------------------------------------------------
            self.progress.emit("Running sliding window merging...")
            sliding_result = self.run_script(
                self.SCRIPTS["sliding"],
                bert_output
            )
            sliding_output = self.extract_output_path(sliding_result)
            self.progress.emit(f"Sliding window output: {sliding_output}")
            
            # -------------------------------------------------------
            # 4. Final Flight Phase Assignment
            # -------------------------------------------------------
            self.progress.emit("Running final flight phase assignment...")
            final_result = self.run_script(
                self.SCRIPTS["final"],
                sliding_output
            )
            final_output, all_output_lines = self.extract_output_path(final_result, is_final_script=True)
            
            # Extract and clean phase results
            cleaned_phase_results = self.extract_phase_results(all_output_lines)
            
            self.progress.emit("Pipeline completed successfully!")
            
            # Emit both the output path AND the cleaned phase results
            self.finished.emit(str(final_output), cleaned_phase_results)
            
        except Exception as e:
            self.error.emit(str(e))

class AudioMainWindow(QMainWindow):
    def __init__(self):
        super(AudioMainWindow, self).__init__()
        
        # Get the directory where this script is located
        script_dir = Path(__file__).resolve().parent
        
        # Construct the full path to the UI file
        ui_file = script_dir / "UI/cvr_ui.ui"
        
        if not ui_file.exists():
            QMessageBox.critical(self, "Error", f"UI file not found: {ui_file}")
            return
        
        loadUi(str(ui_file), self)
        
        # Connect signals
        self.browseButton.clicked.connect(self.browse_files)
        self.startButton.clicked.connect(self.start_processing)
        
        # Initialize worker thread
        self.worker_thread = None
        
        # Clear output when starting
        self.output.clear()
        
        # Enable/disable start button based on file selection
        self.filename.textChanged.connect(self.update_start_button_state)
        self.update_start_button_state()
        
        # Set window title
        self.setWindowTitle("Audio Processing Pipeline")
    
    def browse_files(self):
        """Open file dialog to select input audio file"""
        fname, _ = QFileDialog.getOpenFileName(
            self, 
            'Select CVR audio file (.wav)', 
            str(Path.home()),  # Start from home directory
            'Audio files (*.wav *.mp3 *.m4a);;All files (*.*)'
        )
        if fname:
            self.filename.setText(fname)
    
    def update_start_button_state(self):
        """Enable start button only if a file is selected"""
        has_file = bool(self.filename.text().strip())
        self.startButton.setEnabled(has_file)
    
    def start_processing(self):
        """Start the audio processing pipeline"""
        input_file = self.filename.text().strip()
        
        if not input_file:
            QMessageBox.warning(self, "Warning", "Please select an audio file first!")
            return
        
        if not Path(input_file).exists():
            QMessageBox.critical(self, "Error", f"File not found: {input_file}")
            return
        
        # Clear previous output
        self.output.clear()
        self.labelPath.clear()
        
        # Disable buttons during processing
        self.startButton.setEnabled(False)
        self.browseButton.setEnabled(False)
        
        # Add initial message to output
        self.output.appendPlainText("=" * 60)
        self.output.appendPlainText("AUDIO PROCESSING PIPELINE STARTED")
        self.output.appendPlainText("=" * 60)
        self.output.appendPlainText(f"Input audio file: {input_file}")
        self.output.appendPlainText("")
        
        # Create and start worker thread
        self.worker_thread = AudioWorkerThread(input_file)
        self.worker_thread.progress.connect(self.update_progress)
        self.worker_thread.finished.connect(self.on_processing_finished)
        self.worker_thread.error.connect(self.on_processing_error)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.error.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()
    
    def update_progress(self, message):
        """Update output text with progress messages"""
        self.output.appendPlainText(message)
        # Scroll to bottom
        self.output.verticalScrollBar().setValue(
            self.output.verticalScrollBar().maximum()
        )
        # Force UI update
        QtWidgets.QApplication.processEvents()
    
    def on_processing_finished(self, final_output_path, phase_results):
        """Handle successful completion of processing"""
        # Remove the "Pipeline completed successfully!" message if already added by update_progress
        # by checking the last few lines
        current_text = self.output.toPlainText()
        lines = current_text.split('\n')
        
        # Keep only lines that don't contain "Pipeline completed successfully!"
        filtered_lines = [line for line in lines if "Pipeline completed successfully!" not in line]
        
        # Rebuild the output text
        self.output.clear()
        for line in filtered_lines:
            if line.strip():  # Keep non-empty lines
                self.output.appendPlainText(line)
        
        # Now add the completion message
        self.output.appendPlainText("\n" + "=" * 60)
        self.output.appendPlainText("PIPELINE COMPLETED SUCCESSFULLY")
        self.output.appendPlainText(f"Final output saved to:")
        self.output.appendPlainText(final_output_path)
        self.output.appendPlainText("=" * 60)
        
        # Display cleaned phase results
        if phase_results:
            # Count actual phase data lines (those with timing format)
            phase_data_count = sum(1 for line in phase_results if "[" in line and "→" in line and "]")
            
            self.output.appendPlainText("\n" + "=" * 60)
            self.output.appendPlainText("FLIGHT PHASE ANALYSIS RESULTS")
            self.output.appendPlainText("=" * 60)
            
            # Display diagnostic messages first
            for line in phase_results:
                if "Initial phase found:" in line or "phases candidate found:" in line:
                    self.output.appendPlainText(line)
            
            # Display actual phase data
            for line in phase_results:
                if "[" in line and "→" in line and "]" in line:
                    self.output.appendPlainText(line)
            
            # Add phase count
            self.output.appendPlainText("=" * 60)
            self.output.appendPlainText(f"Total phases detected: {phase_data_count}")
            self.output.appendPlainText("=" * 60)
        
        # Update label path field
        self.labelPath.setText(final_output_path)
        
        # Re-enable buttons
        self.startButton.setEnabled(True)
        self.browseButton.setEnabled(True)
        
        QMessageBox.information(self, "Success", 
                               f"Processing completed successfully!\n\nOutput saved to:\n{final_output_path}")
    
    def on_processing_error(self, error_message):
        """Handle errors during processing"""
        self.output.appendPlainText("=" * 60)
        self.output.appendPlainText("ERROR OCCURRED")
        self.output.appendPlainText("=" * 60)
        self.output.appendPlainText(error_message)
        
        # Re-enable buttons
        self.startButton.setEnabled(True)
        self.browseButton.setEnabled(True)
        
        QMessageBox.critical(self, "Processing Error", 
                            f"An error occurred during processing:\n\n{error_message}")
    
    def closeEvent(self, event):
        """Handle window close event - stop any running threads"""
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self, 'Confirm Close',
                'Processing is still running. Are you sure you want to close?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker_thread.terminate()
                self.worker_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    
    # Create and show main window
    main_window = AudioMainWindow()
    main_window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

# 09:49 - 56