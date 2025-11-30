# !pip install mne xgboost lightgbm ripser
# !pip install --upgrade numpy pandas mne scikit-learn

import os
import zipfile
import warnings
import numpy as np
import pandas as pd
import mne
import io
from scipy import signal
from ripser import ripser

# ML Imports
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

BASE_PATH = 'StrokeEEG'
mne.set_log_file(fname=os.devnull, overwrite=True)

# ==========================================
# 2. DATA LOADING
# ==========================================

stroke_eeg = {}
total_signals_loaded = 0

if os.path.exists(BASE_PATH):
    patient_folders = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
    patient_folders.sort()
    print(f"\nDetected {len(patient_folders)} patient folders.\n")

    for patient_folder_name in patient_folders:
        eeg_data_path = os.path.join(BASE_PATH, patient_folder_name, "eeg")
        if not os.path.exists(eeg_data_path): continue

        edf_files = [f for f in os.listdir(eeg_data_path) if f.endswith(".edf")]
        if not edf_files: continue

        patient_data_arrays = []
        for edf_file_name in edf_files:
            full_edf_path = os.path.join(eeg_data_path, edf_file_name)
            try:
                raw = mne.io.read_raw_edf(full_edf_path, preload=True)
                patient_data_arrays.append(raw.get_data())
                total_signals_loaded += 1
            except Exception as e:
                print(f"Error loading {full_edf_path}: {e}")
                continue

        if patient_data_arrays:
            stroke_eeg[patient_folder_name] = patient_data_arrays[0]
else:
    print("Base path not found. Check extraction.")

print(f"| Total Patients Loaded: {len(stroke_eeg)}")

# ==========================================
# 3. METADATA SETUP
# ==========================================

metadata_table = """
Participant_ID	Gender	Age	Duration	ParalysisSide	Handedness	IsFirstTime	StrokeLocation	NIHSS	MBI	mRS
sub-01	male	45	1	right	right	yes	Left pons	11	50	4
sub-02	male	60	2	left	right	yes	Right pons	3	55	4
sub-03	male	60	2	left	right	no	"Left cerebellum, bilateral paraventricular, Right corona radiata"	3	90	1
sub-04	male	56	14	right	right	yes	"Left frontal parietal cortex, Left centrum semiovale"	6	90	3
sub-05	female	44	4	left	right	yes	Left pons	4	60	4
sub-06	male	66	6	left	left	no	Right pons	3	85	3
sub-07	male	62	5	right	right	no	Left pons	2	100	1
sub-08	male	64	5	left	right	yes	Right basal ganglia	3	85	2
sub-09	male	57	3	right	right	yes	Left paraventricular	6	55	1
sub-10	male	55	2	left	right	no	Right pons	3	55	0
sub-11	male	31	7	left	right	yes	Right paraventricular	5	55	4
sub-12	male	58	1	right	right	yes	Left medulla oblongata	1	100	1
sub-13	male	46	3	right	right	no	Left paraventricular	9	55	4
sub-14	female	67	2	right	right	yes	Left pons	2	75	1
sub-15	male	63	1	left	left	yes	"Right fronto-parietal temporo-occipital lobe, Right inner watershed"	7	55	1
sub-16	male	57	1	right	right	no	Left basal ganglia	4	90	2
sub-17	male	60	3	left	right	yes	"Right paraventricular, Right basal ganglia"	3	85	1
sub-18	female	60	1	left	right	yes	Right basal ganglia	10	45	4
sub-19	female	62	1	left	right	no	"Right paraventricular, Right basal ganglia"	8	40	4
sub-20	male	34	24	right	right	no	"Left paraventricular, Right temporal lobe"	2	95	1
sub-21	male	41	5	left	right	yes	Pons	2	90	1
sub-22	male	52	6	left	right	yes	"Right temporo-parietal occipital lobe and insula, Right basal ganglia, Right paraventricular"	11	45	4
sub-23	male	57	2	left	right	yes	"Right paraventricular, Right basal ganglia"	4	40	4
sub-24	female	55	3	right	right	yes	Left paraventricular	1	60	4
sub-25	male	47	10	left	right	yes	Right paraventricular	5	55	4
sub-26	male	61	1	left	right	yes	Right thalamus	4	70	4
sub-27	female	52	5	right	right	yes	"Left basal ganglia, Left paraventricular"	3	70	3
sub-28	female	42	2	right	right	yes	Left thalamus	1	85	1
sub-29	male	53	16	right	right	yes	Left thalamus	1	95	0
sub-30	male	68	4	left	right	yes	Right paraventricular	4	100	1
sub-31	female	59	5	right	right	no	"Left corona radiata, Left centrum semiovale"	3	80	3
sub-32	male	74	2	right	right	yes	Left pons	3	81	4
sub-33	male	63	7	right	right	yes	Pons	3	58	4
sub-34	female	69	1	left	right	no	Right frontal lobe	1	85	2
sub-35	male	69	11	left	right	no	"Right cerebellum,bilateral occipital lobes"	1	52	4
sub-36	male	69	30	left	right	yes	"Right paraventricular,right basal ganglia"	6	63	3
sub-37	male	49	7	left	right	yes	Right internal capsule	3	88	4
sub-38	male	53	2	right	right	yes	Right pons	3	64	3
sub-39	male	56	2	right	right	yes	"Left cerebellar hemisphere, Left medulla oblongata"	11	32	5
sub-40	female	56	6	left	right	no	Right pons	6	65	4
sub-41	male	77	2	right	right	no	Left pons	7	60	4
sub-42	male	54	2	right	right	no	Left pons	7	60	4
sub-43	male	32	3	left	right	yes	Right frontal lobe	2	65	4
sub-44	male	59	1	left	right	yes	"Right subfrontal cortex, Right basal ganglia, Right lateral ventricle, Right corona radiata"	1	80	1
sub-45	male	64	30	right	right	yes	Left pons	6	84	4
sub-46	male	66	10	right	right	yes	Left parietal lobe	1	85	1
sub-47	male	40	4	left	right	yes	Right medulla oblongata	7	55	4
sub-48	male	75	18	left	right	yes	"Right subcortical cerebral hemisphere, Right basal ganglia, Left subparietal cortex"	2	90	1
sub-49	male	52	3	left	right	yes	Right basal ganglia	1	85	1
sub-50	female	64	1	left	right	yes	Right pons	3	85	2
"""

subject_labels_df = pd.read_csv(io.StringIO(metadata_table), sep='\t')
subject_labels_df.rename(columns={'Participant_ID': 'subject_id'}, inplace=True)
loaded_subjects = list(stroke_eeg.keys())
final_labels_df = subject_labels_df[subject_labels_df['subject_id'].isin(loaded_subjects)]

# ==========================================
# 4. PREPROCESSING & EPOCHING
# ==========================================

SFREQ = 250.0
L_FREQ, H_FREQ, NOTCH_FREQ = 0.5, 45.0, 50.0
TMAX = 5.0
STEP_S = TMAX * 0.9

EXPLICIT_31_CHANNEL_NAMES = [
    'FP1', 'FP2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FCz', 'FC3', 'FC4',
    'FT7', 'FT8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'CP3', 'CP4', 'TP7',
    'TP8', 'Pz', 'P3', 'P4', 'T5', 'T6', 'Oz', 'O1', 'O2',
    'HEOL', 'VEOR'
]
FINAL_CHANNEL_NAMES_33 = EXPLICIT_31_CHANNEL_NAMES + ['STATUS1', 'STATUS2']

stroke_epochs = {}

print("--- Preprocessing Pipeline ---")
for patient_id, eeg_array in stroke_eeg.items():
    try:
        info = mne.create_info(ch_names=FINAL_CHANNEL_NAMES_33, sfreq=SFREQ, ch_types='eeg')
        raw = mne.io.RawArray(eeg_array, info, verbose=False)
        raw.set_channel_types({'HEOL': 'eog', 'VEOR': 'eog','STATUS1':'stim','STATUS2':'misc'})

        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore', verbose=False)

        if raw.info['sfreq'] != SFREQ: raw.resample(SFREQ, verbose=False)
        raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)
        raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)
        raw.set_eeg_reference('average', projection=False, verbose=False)

        raw_epoched = raw.copy().pick_types(eeg=True, exclude='bads')
        events = mne.make_fixed_length_events(raw_epoched, duration=STEP_S, start=0.0)
        epochs = mne.Epochs(raw_epoched, events, tmin=0, tmax=TMAX, baseline=None, preload=True, verbose=False)

        stroke_epochs[patient_id] = epochs
    except Exception as e:
        print(f"Skipping {patient_id}: {e}")

print(f"Total subjects epoched: {len(stroke_epochs)}")

# ==========================================
# 5. FAST TDA FEATURE COMPUTATION
# ==========================================

def compute_fast_tda_features(band_power_sequence):
    """
    Fast TDA using correlation-based point cloud and H0 only.
    Input: band_power_sequence (n_channels,) - band power values
    Output: dict of TDA features
    """
    try:
        # Create simple 2D embedding from band powers
        n_ch = len(band_power_sequence)
        if n_ch < 3:
            return {'tda_h0_max': 0, 'tda_h0_mean': 0, 'tda_h0_count': 0}
        
        # Use correlation structure as point cloud (fast, no PCA needed)
        points = np.column_stack([
            band_power_sequence,
            np.roll(band_power_sequence, 1),
            np.roll(band_power_sequence, 2)
        ])[:10]  # Use only first 10 channels for speed
        
        # Compute only H0 (connected components) - much faster than H1
        result = ripser(points, maxdim=0, thresh=2.0)
        dgm = result['dgms'][0]
        
        # Extract simple statistics
        if len(dgm) > 0:
            pers = dgm[:, 1] - dgm[:, 0]
            pers = pers[np.isfinite(pers)]
            if len(pers) > 0:
                return {
                    'tda_h0_max': np.max(pers),
                    'tda_h0_mean': np.mean(pers),
                    'tda_h0_count': len(pers)
                }
        
        return {'tda_h0_max': 0, 'tda_h0_mean': 0, 'tda_h0_count': 0}
    except:
        return {'tda_h0_max': 0, 'tda_h0_mean': 0, 'tda_h0_count': 0}

# ==========================================
# 6. FEATURE EXTRACTION (WITH FAST TDA)
# ==========================================

BANDS = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
PAIRS = [('F3', 'F4'), ('C3', 'C4'), ('P3', 'P4'), ('O1', 'O2'), ('F7', 'F8'), ('T3', 'T4'), ('T5', 'T6')]

all_features = []
first_sub = list(stroke_epochs.keys())[0]
ch_names = stroke_epochs[first_sub].ch_names

print("--- Starting Feature Extraction (with Fast TDA) ---")

for patient_id, epochs in stroke_epochs.items():
    data = epochs.get_data()

    for idx, epoch in enumerate(data):
        feat_dict = {'subject_id': patient_id}

        freqs, psd = signal.welch(epoch, fs=SFREQ, nperseg=256)

        for band, (lf, hf) in BANDS.items():
            freq_mask = (freqs >= lf) & (freqs <= hf)
            band_power = np.trapz(psd[:, freq_mask], freqs[freq_mask], axis=1)

            feat_dict[f'{band}_mean'] = np.mean(band_power)

            # Asymmetry features
            for (left_ch, right_ch) in PAIRS:
                if left_ch in ch_names and right_ch in ch_names:
                    idx_l = ch_names.index(left_ch)
                    idx_r = ch_names.index(right_ch)
                    p_l = band_power[idx_l]
                    p_r = band_power[idx_r]
                    asym = (p_l - p_r) / (p_l + p_r + 1e-9)
                    feat_dict[f'{band}_asym_{left_ch}_{right_ch}'] = asym
            
            # Add fast TDA features for each band
            tda_feats = compute_fast_tda_features(band_power)
            for k, v in tda_feats.items():
                feat_dict[f'{band}_{k}'] = v

        all_features.append(feat_dict)

feature_df = pd.DataFrame(all_features)
print(f"\nExtraction Complete. Shape: {feature_df.shape}")
print(f"Features per epoch: {len(feature_df.columns) - 1}")  # -1 for subject_id

# ==========================================
# 7. PREPARE DATA FOR PARALYSIS SIDE PREDICTION
# ==========================================

target_col = 'ParalysisSide'

print("\n" + "="*60)
print(f" MULTI-MODEL CLASSIFICATION FOR {target_col.upper()} ")
print("="*60)

merged_df = feature_df.merge(final_labels_df[['subject_id', target_col]], on='subject_id', how='inner')
merged_df.dropna(subset=[target_col], inplace=True)

le = LabelEncoder()
merged_df['target'] = le.fit_transform(merged_df[target_col].astype(str))
class_names = le.classes_

print(f"Classes found: {class_names}")

drop_cols = ['subject_id', 'target', target_col]
X = merged_df.drop(columns=[c for c in drop_cols if c in merged_df.columns])
y = merged_df['target']
groups = merged_df['subject_id']

# Split Data
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
test_subjects = groups.iloc[test_idx]

print(f"\nTrain subjects: {len(groups.iloc[train_idx].unique())}")
print(f"Test subjects: {len(test_subjects.unique())}")
print(f"Test subject IDs: {sorted(test_subjects.unique())}")

# ==========================================
# 8. DEFINE MULTIPLE MODELS
# ==========================================

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, class_weight='balanced', random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, class_weight='balanced', random_state=42, verbose=-1),
    'Linear SVM': SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42, probability=True),
    'RBF SVM': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42, probability=True)
}

# ==========================================
# 9. TRAIN AND EVALUATE ALL MODELS
# ==========================================

results_summary = []

for model_name, classifier in models.items():
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print('='*60)
    
    # Create Pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42), threshold='median')),
        ('clf', classifier)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict at epoch level
    y_pred_epochs = pipeline.predict(X_test)
    
    # Aggregate to patient level (majority voting)
    results_df = pd.DataFrame({
        'subject': test_subjects.values, 
        'y_true': y_test.values, 
        'y_pred': y_pred_epochs
    })
    
    # Patient-level aggregation with proper mode handling
    patient_preds = []
    patient_trues = []
    
    for subject in results_df['subject'].unique():
        subject_data = results_df[results_df['subject'] == subject]
        
        # True label (should be same for all epochs of a subject)
        true_label = subject_data['y_true'].iloc[0]
        patient_trues.append(true_label)
        
        # Predicted label (majority vote across epochs)
        pred_counts = subject_data['y_pred'].value_counts()
        pred_label = pred_counts.idxmax()
        patient_preds.append(pred_label)
    
    patient_trues = np.array(patient_trues)
    patient_preds = np.array(patient_preds)
    
    # Calculate Accuracy
    acc = accuracy_score(patient_trues, patient_preds)
    
    print(f"\n✅ {model_name} PATIENT ACCURACY: {acc:.4f}")
    print(f"Total test patients: {len(patient_trues)}")
    print("-" * 60)
    print(classification_report(patient_trues, patient_preds, target_names=class_names))
    
    # Store results
    results_summary.append({
        'Model': model_name,
        'Accuracy': acc
    })
    
    # Plot Confusion Matrix
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(patient_trues, patient_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

# ==========================================
# 10. COMPARE ALL MODELS
# ==========================================

print("\n" + "="*60)
print(" MODEL COMPARISON SUMMARY ")
print("="*60)

results_df_summary = pd.DataFrame(results_summary).sort_values('Accuracy', ascending=False)
print(results_df_summary.to_string(index=False))

# Bar plot comparison
plt.figure(figsize=(10, 6))
plt.barh(results_df_summary['Model'], results_df_summary['Accuracy'], color='steelblue')
plt.xlabel('Patient-Level Accuracy')
plt.title(f'Model Comparison for {target_col} Prediction')
plt.xlim(0, 1)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✅ All models trained and evaluated successfully!")