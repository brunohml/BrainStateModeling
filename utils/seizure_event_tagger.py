import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.colors import hsv_to_rgb
import sys

def setup_output_directory(patient_id):
    """Create output directory structure for the patient."""
    output_dir = os.path.join('output', patient_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def standardize_date(date_str):
    """Standardize date string to MM/DD/YYYY format."""
    parts = date_str.split('/')
    if len(parts) != 3:
        return date_str
    month, day, year = parts
    return f"{int(month):02d}/{int(day):02d}/{year}"

def standardize_time(time_str):
    """Standardize time string to HH:MM:SS format."""
    parts = time_str.split(':')
    if len(parts) != 3:
        return time_str
    hour, minute, second = parts
    return f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}"

def is_valid_time(time_str):
    """Check if string is a valid time in HH:MM:SS format."""
    try:
        parts = time_str.split(':')
        if len(parts) != 3:
            return False
        hour, minute, second = map(int, parts)
        return 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59
    except:
        return False

def load_seizure_metadata(excel_path, target_patient_id):
    """Load and process seizure metadata for a specific patient."""
    print("Loading seizure metadata...")
    seizure_data = pd.read_excel(excel_path)
    
    # Filter for target patient first
    patient_data = seizure_data[seizure_data['Pat ID'] == target_patient_id].copy()
    
    if len(patient_data) == 0:
        raise ValueError(f"No seizure events found for patient {target_patient_id}")
    
    # Process timestamps for the filtered data
    patient_data = patient_data[
        ~(
            patient_data['Date (MM:DD:YYYY)'].isna() |
            patient_data['Electrographical Ictal Onset (HH:MM:SS) 24Hr'].isna() |
            patient_data['Electrographic Ictal Offset (HH:MM:SS) 24Hr'].isna()
        )
    ]
    
    print(f"\nProcessing {len(patient_data)} seizure events for {target_patient_id}")
    
    # Convert date format and create datetime objects
    patient_data['Date (MM:DD:YYYY)'] = patient_data['Date (MM:DD:YYYY)'].apply(lambda x: x.split(':')[0] + '/' + x.split(':')[1] + '/' + x.split(':')[2])
    
    date_format = '%m/%d/%Y %H:%M:%S'
    patient_data['onset_datetime'] = pd.to_datetime(
        patient_data['Date (MM:DD:YYYY)'] + ' ' + 
        patient_data['Electrographical Ictal Onset (HH:MM:SS) 24Hr'],
        format=date_format
    )
    
    patient_data['offset_datetime'] = pd.to_datetime(
        patient_data['Date (MM:DD:YYYY)'] + ' ' + 
        patient_data['Electrographic Ictal Offset (HH:MM:SS) 24Hr'],
        format=date_format
    )
    
    return patient_data

def find_seizure_event(start_time, stop_time, seizure_data, patient_id, debug=False):
    """Find seizure event details for a given time window and patient."""
    # Only show debug info if requested
    if debug:
        print("\nInput data types:")
        print(f"start_time type: {type(start_time)}, value: {start_time}")
        print(f"stop_time type: {type(stop_time)}, value: {stop_time}")
        print(f"\nFiltering for patient {patient_id}")
        print(f"Total rows in seizure_data: {len(seizure_data)}")
        print(f"Unique patient IDs: {seizure_data['Pat ID'].unique()}")
    
    patient_seizures = seizure_data[
        (seizure_data['Pat ID'] == patient_id) & 
        (seizure_data['Type'] == 'Seizure')
    ]
    
    if debug:
        print(f"\nFound {len(patient_seizures)} total seizures for patient {patient_id}")
        if len(patient_seizures) > 0:
            print("\nSample of patient seizures:")
            print(patient_seizures[['Pat ID', 'Type', 'onset_datetime', 'offset_datetime']].head())
    
    if len(patient_seizures) == 0:
        return None, None
    
    overlapping_seizures = patient_seizures[
        ((patient_seizures['onset_datetime'] <= start_time) & (patient_seizures['offset_datetime'] >= start_time)) |
        ((patient_seizures['onset_datetime'] <= stop_time) & (patient_seizures['offset_datetime'] >= stop_time)) |
        ((patient_seizures['onset_datetime'] >= start_time) & (patient_seizures['offset_datetime'] <= stop_time))
    ]
    
    if len(overlapping_seizures) > 0:
        seizure = overlapping_seizures.iloc[0]
        if debug:
            print(f"Found seizure:")
            print(f"  Onset: {seizure['onset_datetime']}")
            print(f"  Offset: {seizure['offset_datetime']}")
            print(f"  Type: {seizure['Seizure Type (FAS; FIAS; FBTC; Non-electrographic; Subclinical; Unknown)']}")
        
        seizure_type = str(seizure['Seizure Type (FAS; FIAS; FBTC; Non-electrographic; Subclinical; Unknown)'])
        if pd.isna(seizure_type):
            seizure_type = 'Unknown'
        return seizure.name, seizure_type
    return None, None

def generate_distinct_colors(n):
    """Generate n visually distinct colors using HSV color space."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.9
        value = 0.9
        colors.append(hsv_to_rgb([hue, saturation, value]))
    return colors

def create_single_seizure_plot(points_2d, seizure_mask, window_times, patient_id, seizure_type, onset_time, offset_time, output_dir, no_periictal=False):
    """Create visualization for a single seizure event with color-coded progression."""
    plt.figure(figsize=(10, 8))
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Calculate indices for pre/post seizure windows (3 minutes = 36 windows since each window is 5 seconds)
    num_windows_per_3min = 36
    all_indices = np.arange(len(points_2d))
    seizure_indices = all_indices[seizure_mask]
    
    if not no_periictal:
        # Get pre-seizure indices (3 minutes before first seizure point)
        first_seizure_idx = seizure_indices[0]
        pre_seizure_indices = np.arange(
            max(0, first_seizure_idx - num_windows_per_3min),
            first_seizure_idx
        )
        
        # Get post-seizure indices (3 minutes after last seizure point)
        last_seizure_idx = seizure_indices[-1]
        post_seizure_indices = np.arange(
            last_seizure_idx + 1,
            min(len(points_2d), last_seizure_idx + 1 + num_windows_per_3min)
        )
        
        # Create masks for pre/post seizure points
        pre_seizure_mask = np.zeros(len(points_2d), dtype=bool)
        pre_seizure_mask[pre_seizure_indices] = True
        
        post_seizure_mask = np.zeros(len(points_2d), dtype=bool)
        post_seizure_mask[post_seizure_indices] = True
    else:
        pre_seizure_mask = np.zeros(len(points_2d), dtype=bool)
        post_seizure_mask = np.zeros(len(points_2d), dtype=bool)
    
    # Plot background points (excluding pre/post/seizure)
    background_mask = ~(seizure_mask | pre_seizure_mask | post_seizure_mask)
    plt.scatter(
        points_2d[background_mask, 0],
        points_2d[background_mask, 1],
        c='gray',
        s=0.5,
        alpha=0.3,
        label='Non-seizure'
    )
    
    if not no_periictal:
        # Create color progressions for pre and post seizure points
        pre_seizure_colors = plt.cm.GnBu(np.linspace(0.4, 0.9, len(pre_seizure_indices)))
        post_seizure_colors = plt.cm.RdPu(np.linspace(0.4, 0.9, len(post_seizure_indices)))
        
        # Plot pre-seizure points with color progression
        plt.scatter(
            points_2d[pre_seizure_mask, 0],
            points_2d[pre_seizure_mask, 1],
            c=pre_seizure_colors,
            marker='^',
            s=20,
            alpha=1.0,
            edgecolors='none',
            label='Pre-seizure (3 min)'
        )
        
        # Plot post-seizure points with color progression
        plt.scatter(
            points_2d[post_seizure_mask, 0],
            points_2d[post_seizure_mask, 1],
            c=post_seizure_colors,
            marker='^',
            s=20,
            alpha=1.0,
            edgecolors='none',
            label='Post-seizure (3 min)'
        )
    
    # Get seizure points and create progression
    seizure_points = points_2d[seizure_mask]
    progression = np.linspace(0, 1, len(seizure_points))
    
    # Plot seizure points with color progression
    scatter = plt.scatter(
        seizure_points[:, 0],
        seizure_points[:, 1],
        c=progression,
        cmap='YlOrRd',
        s=16.0,
        alpha=0.8,
        vmin=0,
        vmax=1,
        edgecolors='black',
        linewidth=0.7,
        label=f'{seizure_type} Seizure'
    )
    
    # Add colorbar with correct orientation
    cbar = plt.colorbar(scatter, orientation='vertical')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Start', 'End'])
    cbar.ax.invert_yaxis()
    
    # Format duration for title
    duration_mins = (offset_time - onset_time).total_seconds() / 60
    if duration_mins >= 1:
        duration_str = f"{duration_mins:.1f} minutes"
    else:
        duration_str = f"{int((offset_time - onset_time).total_seconds())} seconds"
    
    plt.title(f'Brain State - {patient_id}\n{seizure_type} Seizure at {onset_time.strftime("%Y-%m-%d %H:%M")}\nDuration: {duration_str}')
    plt.legend()
    
    # Save plot
    plots_dir = os.path.join(output_dir, 'seizure_event_plots_staged')
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_filename = f"pacmap_{patient_id}_{seizure_type}_{onset_time.strftime('%Y-%m-%d-%H-%M')}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_overlay_plot(points_2d, seizure_events_of_type, patient_id, seizure_type, output_dir):
    """Create overlay plot for all seizures of the same type for a patient."""
    plt.figure(figsize=(10, 8))
    
    # Plot all background points in gray
    plt.scatter(
        points_2d[:, 0],
        points_2d[:, 1],
        c='gray',
        s=0.5,
        alpha=0.3,
        label='Non-seizure'
    )
    
    # Plot each seizure's points with progression coloring
    for _, onset_time, offset_time, seizure_mask in seizure_events_of_type:
        seizure_points = points_2d[seizure_mask]
        if len(seizure_points) > 0:
            progression = np.linspace(0, 1, len(seizure_points))
            scatter = plt.scatter(
                seizure_points[:, 0],
                seizure_points[:, 1],
                c=progression,
                cmap='YlOrRd',
                s=2.0,
                alpha=0.8,
                vmin=0,
                vmax=1
            )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, orientation='vertical')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Start', 'End'])
    cbar.ax.invert_yaxis()
    cbar.set_label('Seizure Progression', rotation=270, labelpad=15)
    
    plt.title(f'Brain State - {patient_id}\n{seizure_type} Seizures\nTotal Seizures: {len(seizure_events_of_type)}')
    plt.legend()
    
    # Save plot
    plots_dir = os.path.join(output_dir, 'seizure_event_plots_staged')
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_filename = f"pacmap_{patient_id}_{seizure_type}_overlay.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_periictal_overlay_plot(points_2d, seizure_events_of_type, patient_id, seizure_type, output_dir):
    """Create overlay plot for peri-ictal periods of all seizures of the same type."""
    plt.figure(figsize=(10, 8))
    
    # Plot background points in gray
    plt.scatter(
        points_2d[:, 0],
        points_2d[:, 1],
        c='gray',
        s=0.5,
        alpha=0.3
    )
    
    # For each seizure, plot its peri-ictal periods
    for _, onset_time, offset_time, seizure_mask in seizure_events_of_type:
        # Get seizure indices
        all_indices = np.arange(len(points_2d))
        seizure_indices = all_indices[seizure_mask]
        
        # Calculate pre/post indices (10 minutes = 120 windows)
        num_windows_per_10min = 120
        first_seizure_idx = seizure_indices[0]
        last_seizure_idx = seizure_indices[-1]
        
        # Get pre-seizure indices and points
        pre_seizure_indices = np.arange(
            max(0, first_seizure_idx - num_windows_per_10min),
            first_seizure_idx
        )
        pre_seizure_mask = np.zeros(len(points_2d), dtype=bool)
        pre_seizure_mask[pre_seizure_indices] = True
        
        # Get post-seizure indices and points
        post_seizure_indices = np.arange(
            last_seizure_idx + 1,
            min(len(points_2d), last_seizure_idx + 1 + num_windows_per_10min)
        )
        post_seizure_mask = np.zeros(len(points_2d), dtype=bool)
        post_seizure_mask[post_seizure_indices] = True
        
        # Create color progressions
        pre_seizure_colors = plt.cm.GnBu(np.linspace(0.4, 0.9, len(pre_seizure_indices)))
        post_seizure_colors = plt.cm.RdPu(np.linspace(0.4, 0.9, len(post_seizure_indices)))
        
        # Plot pre-seizure points
        plt.scatter(
            points_2d[pre_seizure_mask, 0],
            points_2d[pre_seizure_mask, 1],
            c=pre_seizure_colors,
            marker='^',
            s=20,
            alpha=1.0,
            edgecolors='none'
        )
        
        # Plot post-seizure points
        plt.scatter(
            points_2d[post_seizure_mask, 0],
            points_2d[post_seizure_mask, 1],
            c=post_seizure_colors,
            marker='^',
            s=20,
            alpha=1.0,
            edgecolors='none'
        )
    
    plt.title(f'Brain State - {patient_id}\n{seizure_type} Peri-ictal Periods\nTotal Seizures: {len(seizure_events_of_type)}')
    
    # Save plot
    plots_dir = os.path.join(output_dir, 'seizure_event_plots_staged')
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_filename = f"pacmap_{patient_id}_{seizure_type}_periictaloverlay.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_visualizations(data, patient_id, output_dir, no_periictal=False):
    """Create individual visualizations for each seizure event."""
    points_2d = data['transformed_points_2d']
    
    # Simplify time field access since we know the exact field names from pickle_to_cloud.py
    window_starts = pd.to_datetime(data['start_times'])
    window_stops = pd.to_datetime(data['stop_times'])
    
    if window_starts is None or window_stops is None:
        raise ValueError("Could not find window start/stop times in data")
    
    # Define valid seizure types
    valid_seizure_types = {
        'FAS', 
        'FIAS', 
        'FBTC', 
        'FAS_to_FIAS', 
        'Focal, unknown awareness'
    }
    
    # Load seizure metadata
    seizure_data = load_seizure_metadata('metadata/ictal_event_metadata.xlsx', patient_id)
    seizure_events = []
    
    # Create list of seizure events with their types
    for _, seizure in seizure_data.iterrows():
        if seizure['Seizure Type (FAS; FIAS; FBTC; Non-electrographic; Subclinical; Unknown)'] not in valid_seizure_types:
            continue
            
        onset = seizure['onset_datetime']
        offset = seizure['offset_datetime']
        seizure_type = seizure['Seizure Type (FAS; FIAS; FBTC; Non-electrographic; Subclinical; Unknown)']
        
        # Find all windows that overlap with this seizure
        seizure_mask = (window_starts <= offset) & (window_stops >= onset)
        
        if seizure_mask.any():
            seizure_events.append((seizure_type, onset, offset, seizure_mask))
    
    # Generate plots
    print(f"\nGenerating {len(seizure_events)} individual seizure event plots...")
    if len(seizure_events) == 0:
        print(f"No valid seizure events found for patient {patient_id}")
        return []
        
    plot_paths = []
    
    for i, (seizure_type, onset_time, offset_time, seizure_mask) in enumerate(seizure_events, 1):
        plot_path = create_single_seizure_plot(
            points_2d,
            seizure_mask,
            window_starts,  # Pass window times for progression calculation
            patient_id,
            seizure_type,
            onset_time,
            offset_time,
            output_dir,
            no_periictal=no_periictal  # Pass the no_periictal parameter
        )
        plot_paths.append(plot_path)
        print(f"Completed plot {i}/{len(seizure_events)}: {seizure_type} at {onset_time.strftime('%Y-%m-%d %H:%M')}")
    
    # After creating all individual plots, create overlay plots grouped by seizure type
    seizure_types = {}
    for seizure_event in seizure_events:
        seizure_type = seizure_event[0]
        if seizure_type not in seizure_types:
            seizure_types[seizure_type] = []
        seizure_types[seizure_type].append(seizure_event)
    
    # Create overlay plots for types with multiple seizures
    for seizure_type, events_of_type in seizure_types.items():
        if len(events_of_type) > 1:
            # Create seizure overlay plot
            overlay_path = create_overlay_plot(
                points_2d,
                events_of_type,
                patient_id,
                seizure_type,
                output_dir
            )
            plot_paths.append(overlay_path)
            print(f"Created overlay plot for {seizure_type} seizures (n={len(events_of_type)})")
            
            # Create peri-ictal overlay plot
            periictal_path = create_periictal_overlay_plot(
                points_2d,
                events_of_type,
                patient_id,
                seizure_type,
                output_dir
            )
            plot_paths.append(periictal_path)
            print(f"Created peri-ictal overlay plot for {seizure_type} seizures (n={len(events_of_type)})")
    
    return plot_paths

def validate_patient_id(patient_id, seizure_data):
    """Validate that the patient exists in the metadata and has seizures."""
    unique_patients = seizure_data['Pat ID'].unique()
    
    # Check if patient ID exists exactly as given
    if patient_id in unique_patients:
        return True
    
    # Check for case variations (e.g., 'Epat30' vs 'epat30' vs 'EPAT30')
    patient_lower = patient_id.lower()
    matching_patients = [p for p in unique_patients if p.lower() == patient_lower]
    
    if matching_patients:
        print(f"\nFound patient ID in different case: {matching_patients[0]}")
        return True
    
    # If not found, print helpful debug information
    print("\nPatient ID not found in metadata!")
    print(f"Looking for: {patient_id}")
    print("\nAvailable patient IDs:")
    for p in sorted(unique_patients):
        print(f"  {p}")
    
    return False

def tag_points(patient_id, metadata_path):
    """Tag transformed points with seizure metadata."""
    output_dir = setup_output_directory(patient_id)
    patient_data_path = os.path.join(output_dir, f'pickle2cloud_{patient_id}.pkl')
    
    if not os.path.exists(patient_data_path):
        raise FileNotFoundError(f"No data file found for patient {patient_id}. Run pickle_to_cloud.py first.")
    
    # Load and verify data
    print("\nLoading patient data...")
    with open(patient_data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle both old format (patient_id) and new format (patient_ids)
    if 'patient_ids' in data:
        if patient_id not in data['patient_ids']:
            raise ValueError(f"Patient ID mismatch in data file. Expected {patient_id}, not found in {set(data['patient_ids'])}")
    elif 'patient_id' in data:
        if data['patient_id'] != patient_id:
            raise ValueError(f"Patient ID mismatch in data file. Expected {patient_id}, found {data['patient_id']}")
    else:
        raise ValueError("No patient identifier found in data file")
    
    # Create visualizations for each seizure event
    plot_paths = create_visualizations(data, patient_id, output_dir)
    
    return patient_data_path, plot_paths

def get_all_patients(metadata_path):
    """Get list of all patients from metadata file."""
    seizure_data = pd.read_excel(metadata_path)
    return sorted(seizure_data['Pat ID'].unique())

def tag_embeddings_with_seizures(embeddings_data, patient_id, metadata_path='metadata/ictal_event_metadata.xlsx'):
    """Tag embeddings data with seizure information.
    
    Args:
        embeddings_data (dict): Dictionary containing embeddings data with start_times and stop_times
        patient_id (str): Patient ID (e.g., 'Epat30')
        metadata_path (str): Path to Excel file containing seizure metadata
        
    Returns:
        numpy.ndarray: Array of binary seizure labels (0: non-seizure, 1: seizure)
    """
    # Load seizure metadata
    seizure_data = load_seizure_metadata(metadata_path, patient_id)
    
    # Convert times to pandas datetime
    window_starts = pd.to_datetime(embeddings_data['start_times'])
    window_stops = pd.to_datetime(embeddings_data['stop_times'])
    
    # Initialize seizure labels array
    num_windows = len(window_starts)
    seizure_labels = np.zeros(num_windows, dtype=int)
    
    # Define valid seizure types
    valid_seizure_types = {
        'FAS', 
        'FIAS', 
        'FBTC', 
        'FAS_to_FIAS', 
        'Focal, unknown awareness'
    }
    
    print(f"\nProcessing seizures for {patient_id}:")
    print(f"Total windows to process: {num_windows}")
    print(f"Time range: {window_starts.min()} to {window_stops.max()}")
    
    # Keep track of statistics
    total_seizures = 0
    total_tagged_windows = 0
    
    # Tag each window with seizure information
    for _, seizure in seizure_data.iterrows():
        seizure_type = seizure['Seizure Type (FAS; FIAS; FBTC; Non-electrographic; Subclinical; Unknown)']
        if seizure_type not in valid_seizure_types:
            continue
            
        onset = seizure['onset_datetime']
        offset = seizure['offset_datetime']
        
        # Find all windows that overlap with this seizure
        seizure_mask = (window_starts <= offset) & (window_stops >= onset)
        num_windows_tagged = seizure_mask.sum()
        
        if num_windows_tagged > 0:
            # Update statistics
            total_seizures += 1
            total_tagged_windows += num_windows_tagged
            
            # Tag windows with 1 for seizure
            seizure_labels[seizure_mask] = 1
            
            print(f"\nSeizure found:")
            print(f"  Onset: {onset}")
            print(f"  Offset: {offset}")
            print(f"  Windows tagged: {num_windows_tagged}")
    
    # Print summary statistics
    print("\nSeizure tagging summary:")
    print(f"Total seizures found: {total_seizures}")
    print(f"Total windows tagged: {total_tagged_windows} ({(total_tagged_windows/num_windows)*100:.2f}% of all windows)")
    
    return seizure_labels

def process_single_patient(patient_id, metadata_path, force=False, debug=False, no_periictal=False):
    """Process a single patient's data."""
    try:
        output_dir = setup_output_directory(patient_id)
        patient_data_path = os.path.join(output_dir, f'pickle2cloud_{patient_id}.pkl')
        
        # Skip if point2cloud file doesn't exist
        if not os.path.exists(patient_data_path):
            print(f"Skipping {patient_id}: No pickle2cloud file found")
            # Remove empty directory if it was created
            if os.path.exists(output_dir) and not os.listdir(output_dir):
                os.rmdir(output_dir)
            return False
            
        tagged_data_path = os.path.join(output_dir, f'tagged_pickle2cloud_{patient_id}.pkl')
        
        if os.path.exists(tagged_data_path) and not force:
            print(f"Found existing tagged data for {patient_id}")
            with open(tagged_data_path, 'rb') as f:
                data = pickle.load(f)
            plot_paths = create_visualizations(data, patient_id, output_dir, no_periictal=no_periictal)
        else:
            print(f"Processing {patient_id}...")
            tagged_data_path, plot_paths = tag_points(patient_id, metadata_path)
            
        return True
        
    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Tag brain state embeddings with seizure metadata.')
    parser.add_argument('--patient_id', type=str, help='Patient ID (e.g., Epat37) or "all" to process all patients')
    parser.add_argument('--metadata', type=str, 
                      default='metadata/ictal_event_metadata.xlsx',
                      help='Path to Excel file containing seizure metadata')
    parser.add_argument('--force', action='store_true',
                      help='Force reprocessing of existing tagged data')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    parser.add_argument('--no_periictal', action='store_true',
                      help='Only tag seizure points, exclude pre and post-seizure points')
    
    args = parser.parse_args()
    
    try:
        if args.patient_id.lower() == 'all':
            print("Processing all patients...")
            all_patients = get_all_patients(args.metadata)
            processed_count = 0
            
            for patient_id in all_patients:
                print(f"\nProcessing patient: {patient_id}")
                if process_single_patient(patient_id, args.metadata, args.force, args.debug, args.no_periictal):
                    processed_count += 1
                    
            print(f"\nProcessing complete! Successfully processed {processed_count}/{len(all_patients)} patients")
        else:
            # Original single patient processing
            process_single_patient(args.patient_id, args.metadata, args.force, args.debug, args.no_periictal)
            
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)

if __name__ == "__main__":
    main()



# TODO:
# add command line arguments that make it easy to subselect a given seizure type (useful for the patients with many seizures)
# add command line argument that allows single seizure to be selected and color code by ictal period