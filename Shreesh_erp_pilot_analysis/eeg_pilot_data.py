import pyxdf #to be able to read xdf files
import mne #for EEG data handling and analysis
import numpy as np #for numerical operations, if needed
import pandas as pd

# load xdf datafile
file_path = 'Pilot_1.xdf'  
streams, header = pyxdf.load_xdf(file_path)

# Let's see what streams Lab Recorder saved inside this file
for idx, stream in enumerate(streams):
    name = stream['info']['name'][0]
    stype = stream['info']['type'][0]
    print(f"Stream {idx}: Name='{name}', Type='{stype}'")

# Target the correct streams based on your printout
eeg_stream = streams[2] #EEG data stream from Menteev
marker_stream = streams[0]  # Choosing PsychoPy experimental triggers

# ==========================================
# 2. EXTRACT DATA AND SETUP VARIABLES FIRST
# ==========================================
# Extract and transpose the data matrix
eeg_data = eeg_stream['time_series'].T
eeg_data = eeg_data * 1e-6  # Convert microvolts to Volts

# Gather timing parameters
sfreq = float(eeg_stream['info']['nominal_srate'][0])
eeg_timestamps = eeg_stream['time_stamps']
eeg_start_time = eeg_timestamps[0]  # Absolute LSL clock start

# Extract the raw names directly from your successful XDF check
xdf_ch_names = [
    'p4', 'O1', 'O2', 'T3', 'T4', 'Fz', 'Cz', 'Pz', 'Fp1', 'Fp2', 
    'F3', 'F4', 'C3', 'C4', 'P3', 'Cp6', 'Po3', 'Po4', 'T5', 'T6', 
    'Fc1', 'Fc2', 'Af3', 'Cp1', 'Cp2', 'F7', 'F8', 'Fc5', 'Fc6', 'Cp5'
]

# Helper function to fix the capitalization exactly to MNE's strict standards
def standardize_name(name):
    # MNE strictly requires "Fp" (capital F, lowercase p)
    if name.lower().startswith('fp'):
        return 'Fp' + name[2:]
    # For other compound areas (FC, CP, PO, AF), MNE wants both uppercase
    elif len(name) >= 3 and name[:2].lower() in ['fc', 'cp', 'po', 'af']:
        return name[:2].upper() + name[2:]
    # For single-letter areas (Cz, P4, O1), capitalize the first letter
    return name.capitalize()

# Apply the capitalization fix
ch_names = [standardize_name(ch) for ch in xdf_ch_names]
print("Standardized Channel Names for MNE:\n", ch_names)

# Define all 30 channels as standard EEG types
ch_types = ['eeg'] * len(ch_names)

# ==========================================
# 3. CREATE RAW OBJECT AND APPLY MONTAGE
# ==========================================
# Create MNE info object using the exact metadata order
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Load the transposed data matrix into MNE Raw array ONLY ONCE
raw = mne.io.RawArray(eeg_data, info)

print("Applying standard 10-20 coordinate system...")
# Load the standard 10-20 template positions and apply to raw
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)
print("Montage applied successfully! Your topographical maps are ready.")

# ==========================================
# 4. ALIGN & INJECT PSYCHOPY TRIGGERS
# ==========================================
marker_timestamps = marker_stream['time_stamps']
trigger_labels = [str(m[0]) for m in marker_stream['time_series']]

# Align LSL clocks: make triggers relative to the EEG start time
relative_onsets = marker_timestamps - eeg_start_time
durations = np.zeros(len(relative_onsets))

# Create MNE annotations and map them to the EEG raw object
annotations = mne.Annotations(onset=relative_onsets, 
                              duration=durations, 
                              description=trigger_labels)
raw.set_annotations(annotations)

# Convert text annotations into standard MNE events for plotting
events, event_id = mne.events_from_annotations(raw)
print("Trigger Condition Map:", event_id)

# ==========================================
# 5. PREPROCESSING (YOUR 1-20 Hz FILTER)
# ==========================================
print("Applying Bandpass Filter (1-20 Hz)...")
raw.filter(1, 20, fir_design='firwin') 

# ==========================================
# 6. INDEPENDENT COMPONENT ANALYSIS (ICA)
# ==========================================
from mne.preprocessing import ICA

print("Setting up ICA for artifact removal...")
ica = ICA(n_components=15, random_state=97, max_iter='auto')

# Notice we use .copy() here to protect your 1-20Hz filter from being overwritten!
# We fit ICA on a 1Hz high-pass version of the data, which is standard practice.
print("Fitting ICA. This might take 10-30 seconds...")
ica.fit(raw.copy().filter(1.0, None)) 

# Plot the components as scalp maps!
ica.plot_components()

import matplotlib.pyplot as plt

# 1. Generate the plot and capture the figure object(s)
figs = ica.plot_components()

# 2. Save the figures to your working directory
# MNE can return a single figure or a list of figures
if isinstance(figs, list):
    for idx, fig in enumerate(figs):
        fig.savefig(f'ica_components_page_{idx}.png', dpi=300, bbox_inches='tight')
else:
    figs.savefig('ica_components.png', dpi=300, bbox_inches='tight')

print("ICA plots successfully saved to your folder!")

#After looking at the figures above we got that- ICA000 has this exact "front-only" pattern vertical EOG, which is a telltale sign of eye blinks. 
#To be sure that these are indeed eye artifacts, we can visualize their time-course and frequency properties.
# 1. Visualize the time-course of the suspect components
# This shows you the peaks in time. If these peaks match your raw data blinks, you're safe.
ica.plot_sources(raw, picks=[0, 7])

# 2. Visualize the frequency properties
# This shows you the 'fingerprint' of the component.
ica.plot_properties(raw, picks=[0, 7])

#ICA007 is alpha rhythms, which are normal brain activity. We want to keep these, so we won't exclude them.

# ==========================================
# 7. APPLY ICA TO CLEAN THE DATA
# ==========================================

# Tell MNE exactly which components are artifacts (Blinks = 0)
ica.exclude = [0]

#we can replace above line with the following to exclude multiple components if needed
# Automatically find blinks by comparing components to the Fp1 channel
#bad_idx, scores = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'], threshold=2.0)
# Tell the ICA to exclude the components the computer found
#ica.exclude = bad_idx
#print(f"Automatically found and excluding: {ica.exclude}")

print(f"Excluding components: {ica.exclude}")

# Create a fresh copy of your data so we don't destroy the original
raw_clean = raw.copy()

# Apply the filter. This surgically subtracts ICA000 from the data.
ica.apply(raw_clean)

print("Data successfully cleaned of eye artifacts!")

# Let's plot the final, clean data! 
# You should see that the massive blink spikes are completely gone.
raw_clean.plot(events=events, n_channels=15, duration=20, clipping='transparent', title="Cleaned Data")

# ==========================================
# 8. MERGE PSYCHOPY CSV WITH LSL TRIGGERS
# ==========================================
print("Loading behavioral CSV to map conditions...")

# 1. Load your PsychoPy behavioral data
csv_file = '01-Pilot_Visual rhyme_eeg_1.csv'
df = pd.read_csv(csv_file)

# 2. Filter out practice trials to keep only the main experimental trials
# (We drop any rows where 'exprhyme_cond' is blank/NaN)
df_exp = df.dropna(subset=['exprhyme_cond']).reset_index(drop=True)

# 3. Extract the ordered list of conditions (e.g., 'rhyme' or 'nonrhyme')
conditions = df_exp['exprhyme_cond'].str.lower().str.strip().tolist()

# 4. Find all 'Word 2' triggers in the EEG data
# The N400 is triggered by the SECOND word (exp_stim2), when the brain processes the rhyme.
target_trigger_id = event_id['exp_stim2']
stim2_indices = np.where(events[:, 2] == target_trigger_id)[0]

print(f"Found {len(stim2_indices)} 'exp_stim2' triggers in EEG.") 
print(f"Found {len(conditions)} experimental trials in CSV.")  #should have 88+88 trials from two blocks so 176 trials in total

if len(stim2_indices) != len(conditions):
    print("⚠️ WARNING: Mismatch between EEG triggers and CSV trials!")
    print("Double check if the recording was stopped early or if triggers were dropped.")

# 5. Re-code the generic 'exp_stim2' events into specific conditions
# We create new custom IDs that MNE will use to separate the epochs
new_event_id = {'Rhyme': 101, 'NonRhyme': 102}

for i, idx in enumerate(stim2_indices):
    if i < len(conditions):
        cond = conditions[i]
        if cond == 'rhyme':
            events[idx, 2] = new_event_id['Rhyme']
        elif cond == 'nonrhyme':
            events[idx, 2] = new_event_id['NonRhyme']

# ==========================================
# 9. EPOCH THE DATA
# ==========================================
print("Cutting data into trial epochs...")

# Define the classic N400 time window: 200ms before Word 2, 800ms after
tmin = -0.2 
tmax = 0.8  

# Epoch the clean data using our NEW condition-specific events
# Reject threshold drops epochs with extreme muscle artifacts > 150 µV
epochs = mne.Epochs(raw_clean, events, event_id=new_event_id, tmin=tmin, tmax=tmax, 
                    baseline=(None, 0), preload=True, 
                    reject=dict(eeg=150e-6)) 

print(f"Successfully kept {len(epochs)} clean trials for the main experiment.")

# ==========================================
# 10. CALCULATE ERPs (EVOKED POTENTIALS)
# ==========================================
print("Averaging trials to compute N400 ERPs...")

# Average the epochs for each condition separately
evoked_rhyme = epochs['Rhyme'].average()
evoked_nonrhyme = epochs['NonRhyme'].average()

# ==========================================
# 11. PLOT THE N400
# ==========================================
# The N400 is most prominent at Central-Parietal electrodes.
rois = ['Cz', 'Pz', 'CP1', 'CP2']

evokeds_dict = {
    'Rhyme (Expected)': evoked_rhyme,
    'NonRhyme (Surprise)': evoked_nonrhyme
}

# Plot 1: Waveforms 
# Look for the Red line to separate and bulge UP (negative) away from the Blue line
mne.viz.plot_compare_evokeds(
    evokeds_dict, 
    picks=rois, 
    combine='mean', # Averages our ROI electrodes together into one clean line
    title="N400 ERP: Rhyme vs NonRhyme (Time-locked to Word 2)",
    invert_y=True,  # ERP convention: Plot negative voltages UP
    colors={'Rhyme (Expected)': 'blue', 'NonRhyme (Surprise)': 'red'}
)

# Plot 2: Topographical scalp map of the difference wave (NonRhyme minus Rhyme)
diff_wave = mne.combine_evoked([evoked_nonrhyme, evoked_rhyme], weights=[1, -1])
diff_wave.plot_topomap(times=[0.300, 0.400, 0.500], ch_type='eeg', 
                       title="N400 Scalp Distribution (NonRhyme - Rhyme)")


###According to further classification of ortho and phono
# ==========================================
# ==========================================
print("Loading behavioral CSV to map Orthography/Phonology conditions...")

# 1. Load your PsychoPy behavioral data
csv_file = '01-Pilot_Visual rhyme_eeg_1.csv'
df = pd.read_csv(csv_file)

# 2. Filter out practice trials to keep only the main experimental trials
# (We drop any rows where 'expcondition' is blank/NaN)
df_exp = df.dropna(subset=['expcondition']).reset_index(drop=True)

# 3. Extract the ordered list of 4 conditions (O+P+, O+P-, O-P+, O-P-)
conditions = df_exp['expcondition'].str.upper().str.strip().tolist()

# 4. Find all 'Word 2' triggers in the EEG data
# IMPORTANT SAFETY RESET: Re-extract events from annotations so we have a fresh copy!
# (This prevents the '0 triggers found' error if you run this script multiple times)
events, event_id_dict = mne.events_from_annotations(raw_clean)

# The N400 is triggered by the SECOND word (exp_stim2), when the brain processes the rhyme.
target_trigger_id = event_id_dict['exp_stim2']
stim2_indices = np.where(events[:, 2] == target_trigger_id)[0]

print(f"Found {len(stim2_indices)} 'exp_stim2' triggers in EEG.")
print(f"Found {len(conditions)} experimental trials in CSV.")

if len(stim2_indices) != len(conditions):
    print("⚠️ WARNING: Mismatch between EEG triggers and CSV trials!")
    print("Double check if the recording was stopped early or if triggers were dropped.")

# 5. Re-code the generic 'exp_stim2' events into the 4 specific O/P conditions
# We create new custom IDs that MNE will use to separate the epochs
new_event_id = {'O+P+': 201, 'O+P-': 202, 'O-P+': 203, 'O-P-': 204}

for i, idx in enumerate(stim2_indices):
    if i < len(conditions):
        cond = conditions[i]
        if cond in new_event_id:
            events[idx, 2] = new_event_id[cond]

# ==========================================
#EPOCH THE DATA
# ==========================================
print("Cutting data into trial epochs...")

# Define the classic N400 time window: 200ms before Word 2, 800ms after
tmin = -0.2 
tmax = 0.8  

# Epoch the clean data using our NEW condition-specific events
# Reject threshold drops epochs with extreme muscle artifacts > 150 µV
epochs = mne.Epochs(raw_clean, events, event_id=new_event_id, tmin=tmin, tmax=tmax, 
                    baseline=(None, 0), preload=True, 
                    reject=dict(eeg=150e-6)) 

print(f"Successfully kept {len(epochs)} clean trials for the main experiment.")

# ==========================================
# CALCULATE ERPs (EVOKED POTENTIALS)
# ==========================================
print("Averaging trials to compute N400 ERPs...")

# Average the epochs for all 4 conditions separately
evoked_OP_both = epochs['O+P+'].average()
evoked_O_only = epochs['O+P-'].average()
evoked_P_only = epochs['O-P+'].average()
evoked_neither = epochs['O-P-'].average()

# ==========================================
# PLOT THE N400
# ==========================================
# The N400 is most prominent at Central-Parietal electrodes.
rois = ['Cz', 'Pz', 'CP1', 'CP2']

evokeds_dict = {
    'O+P+ (Similar Spell, Rhymes)': evoked_OP_both,
    'O+P- (Similar Spell, No Rhyme)': evoked_O_only,
    'O-P+ (Diff Spell, Rhymes)': evoked_P_only,
    'O-P- (Diff Spell, No Rhyme)': evoked_neither
}

# Plot 1: Waveforms (4 Conditions)
# We use shades of Blue for Rhymes, shades of Red for Non-Rhymes
mne.viz.plot_compare_evokeds(
    evokeds_dict, 
    picks=rois, 
    combine='mean', # Averages our ROI electrodes together into one clean line
    title="N400 ERP: Orthographic & Phonological Interactions",
    invert_y=True,  # ERP convention: Plot negative voltages UP
    colors={'O+P+ (Similar Spell, Rhymes)': 'darkblue', 
            'O-P+ (Diff Spell, Rhymes)': 'dodgerblue',
            'O+P- (Similar Spell, No Rhyme)': 'darkred',
            'O-P- (Diff Spell, No Rhyme)': 'lightcoral'}
)

# Plot 2: Topographical scalp map - Pure Phonological Violation (O+P- minus O+P+)
# Capture the Figure object to safely set the title
diff_phono = mne.combine_evoked([evoked_O_only, evoked_OP_both], weights=[1, -1])
fig_phono = diff_phono.plot_topomap(times=[0.300, 0.400, 0.500], ch_type='eeg')
fig_phono.suptitle("Pure Phonological Violation (O+P- minus O+P+)", fontsize=14, y=1.02)

# Plot 3: Topographical scalp map - Pure Orthographic Violation (O-P+ minus O+P+)
# Capture the Figure object to safely set the title
diff_ortho = mne.combine_evoked([evoked_P_only, evoked_OP_both], weights=[1, -1])
fig_ortho = diff_ortho.plot_topomap(times=[0.300, 0.400, 0.500], ch_type='eeg')
fig_ortho.suptitle("Pure Orthographic Violation (O-P+ minus O+P+)", fontsize=14, y=1.02)
