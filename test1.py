import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, 
                             QPushButton, QTextEdit, QFileDialog, QComboBox)

# Convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    az = np.radians(az)
    el = np.radians(el)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

# Check if the Doppler correlation is within the threshold
def doppler_correlation(doppler_1, doppler_2, doppler_threshold):
    return abs(doppler_1 - doppler_2) < doppler_threshold

# Check if the distance is within the range threshold
def range_gate(distance, range_threshold):
    return distance < range_threshold

# Get the next available track ID from the track list
def get_next_track_id(track_id_list):
    for idx, track in enumerate(track_id_list):
        if track['state'] == 'free':
            track_id_list[idx]['state'] = 'occupied'
            return track['id'], idx
    new_id = len(track_id_list) + 1
    track_id_list.append({'id': new_id, 'state': 'occupied'})
    return new_id, len(track_id_list) - 1

# Release a track ID by marking it as free
def release_track_id(track_id_list, idx):
    track_id_list[idx]['state'] = 'free'

# Main function for initializing tracks
def initialize_tracks(measurements, doppler_threshold, range_threshold, firm_threshold, time_threshold, mode):
    tracks = []
    track_id_list = []
    miss_counts = {}
    hit_counts = {}
    tentative_ids = {}
    firm_ids = set()

    # Define the state progression based on the mode
    state_progression = {
        3: ['Pos', 'Tentative', 'Firm'],
        5: ['Pos', 'Pos', 'Tentative', 'Tentative', 'Firm'],
        7: ['Pos', 'Pos', 'Tentative', 'Tentative', 'Tentative', 'Firm']
    }
    progression_states = state_progression[firm_threshold]

    for i, measurement in enumerate(measurements):
        measurement_cartesian = sph2cart(measurement[0], measurement[1], measurement[2])
        measurement_doppler = measurement[3]
        measurement_time = measurement[4]

        assigned = False

        for track in tracks:
            last_measurement = track['measurements'][-1]
            last_cartesian = sph2cart(last_measurement[0], last_measurement[1], last_measurement[2])
            last_doppler = last_measurement[3]
            last_time = last_measurement[4]

            distance = np.linalg.norm(np.array(measurement_cartesian) - np.array(last_cartesian))
            doppler_correlated = doppler_correlation(measurement_doppler, last_doppler, doppler_threshold)
            range_satisfied = range_gate(distance, range_threshold)
            time_diff = measurement_time - last_time

            if doppler_correlated and range_satisfied and time_diff <= time_threshold:
                track['measurements'].append(measurement)
                hit_counts[track['id']] += 1
                miss_counts[track['id']] = 0  # Reset miss count on hit
                
                # Update the state based on hit counts
                if hit_counts[track['id']] < len(progression_states):
                    track['state'] = progression_states[hit_counts[track['id']] - 1]
                if hit_counts[track['id']] >= firm_threshold:
                    firm_ids.add(track['id'])
                    track['state'] = 'Firm'
                
                assigned = True
                break

        if not assigned:
            new_track_id, new_track_idx = get_next_track_id(track_id_list)
            new_track = {
                'id': new_track_id,
                'state': progression_states[0],
                'measurements': [measurement]
            }
            tracks.append(new_track)
            hit_counts[new_track_id] = 1
            miss_counts[new_track_id] = 0

        # Check miss counts and release the track if necessary
        for track in tracks:
            if not assigned and miss_counts.get(track['id'], 0) > 0:
                miss_counts[track['id']] += 1  # Increment miss count if not assigned
                
                # Get current state of the track
                current_state = track['state']
                
                # Determine miss threshold based on current state
                if current_state == 'Pos':
                    miss_threshold = 1
                elif current_state == 'Tentative':
                    miss_threshold = 2
                elif current_state == 'Firm':
                    miss_threshold = 3
                else:
                    miss_threshold = firm_threshold  # Fallback in case of unknown state

                # Release the track if the miss count exceeds the threshold for its state
                if miss_counts[track['id']] >= miss_threshold:
                    release_track_id(track_id_list, track_id_list.index({'id': track['id'], 'state': 'occupied'}))
                    tracks.remove(track)

    return tracks, track_id_list, miss_counts, hit_counts, firm_ids

# Load measurements from a CSV file
def load_measurements_from_csv(file_path):
    df = pd.read_csv(file_path)
    measurements = []

    for i in range(len(df)):
        doppler = 1.0
        measurements.append((df['azimuth'][i], df['elevation'][i], df['range'][i], doppler, df['timestamp'][i]))

    return measurements

# Select initiation mode based on user input
def select_initiation_mode(mode):
    if mode == '3-state':
        return 3
    elif mode == '5-state':
        return 5
    elif mode == '7-state':
        return 7
    else:
        raise ValueError("Invalid initiation mode. Choose '3-state', '5-state', or '7-state'.")

# Main application class using PyQt5
class TrackApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.setWindowTitle('Track Initialization')
        self.setGeometry(100, 100, 600, 500)
        
        # Set color palette
        self.setStyleSheet("""
            QWidget {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498db;
                border-radius: 5px;
                padding: 10px;
                color: white;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QComboBox {
                background-color: #34495e;
                color: white;
            }
            QLineEdit {
                background-color: #34495e;
                color: white;
            }
            QTextEdit {
                background-color: #34495e;
                color: white;
            }
        """)
        
        # Layout
        layout = QVBoxLayout()
        
        # File selection
        self.file_label = QLabel('Select CSV File:')
        layout.addWidget(self.file_label)
        self.file_button = QPushButton('Browse')
        self.file_button.clicked.connect(self.browse_file)
        layout.addWidget(self.file_button)
        
        # Initiation mode selection
        self.mode_label = QLabel('Select Initiation Mode:')
        layout.addWidget(self.mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['3-state', '5-state', '7-state'])
        layout.addWidget(self.mode_combo)
        
        # Doppler Threshold
        self.doppler_label = QLabel('Enter Doppler Threshold:')
        layout.addWidget(self.doppler_label)
        self.doppler_input = QLineEdit()
        layout.addWidget(self.doppler_input)
        
        # Range Threshold
        self.range_label = QLabel('Enter Range Threshold:')
        layout.addWidget(self.range_label)
        self.range_input = QLineEdit()
        layout.addWidget(self.range_input)
        
        # Time Threshold
        self.time_label = QLabel('Enter Time Threshold:')
        layout.addWidget(self.time_label)
        self.time_input = QLineEdit()
        layout.addWidget(self.time_input)
        
        # Execute button
        self.execute_button = QPushButton('Initialize Tracks')
        self.execute_button.clicked.connect(self.execute_track_initialization)
        layout.addWidget(self.execute_button)
        
        # Clear Output button
        self.clear_button = QPushButton('Clear Output')
        self.clear_button.clicked.connect(self.clear_output)
        layout.addWidget(self.clear_button)
        
        # Output text box
        self.output_text = QTextEdit()
        # Output text box continued
        self.output_text.setReadOnly(True)  # Make the output box read-only
        layout.addWidget(self.output_text)

        # Set layout to the window
        self.setLayout(layout)

    # Function to browse and select a CSV file
    def browse_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)", options=options)
        if file_path:
            self.file_label.setText(f'Selected File: {file_path}')
            self.file_path = file_path

    # Function to clear the output text
    def clear_output(self):
        self.output_text.clear()

    # Function to execute track initialization based on the user inputs
    def execute_track_initialization(self):
        try:
            # Load the measurements from the selected CSV file
            measurements = load_measurements_from_csv(self.file_path)

            # Get user inputs
            doppler_threshold = float(self.doppler_input.text())
            range_threshold = float(self.range_input.text())
            time_threshold = float(self.time_input.text())
            mode_text = self.mode_combo.currentText()
            firm_threshold = select_initiation_mode(mode_text)

            # Initialize tracks based on the inputs
            tracks, track_id_list, miss_counts, hit_counts, firm_ids = initialize_tracks(
                measurements, doppler_threshold, range_threshold, firm_threshold, time_threshold, mode_text
            )

            # Output results to the output text box
            output_str = "Track Initialization Completed!\n\n"
            output_str += "Tracks:\n"
            for track in tracks:
                output_str += f"Track ID: {track['id']}, State: {track['state']}, Measurements: {len(track['measurements'])}\n"
            self.output_text.setText(output_str)

        except Exception as e:
            self.output_text.setText(f"Error: {str(e)}")


# Main function to run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrackApp()
    window.show()
    sys.exit(app.exec_())
