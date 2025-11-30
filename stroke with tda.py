# !pip install mne xgboost lightgbm ripser
# !pip install --upgrade numpy pandas mne scikit-learn

import os
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
# 1. DATA LOADING
# ==========================================

stroke_eeg = {}
if os.path.exists(BASE_PATH):
    patient_folders = sorted([d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))])
    print(f"\nDetected {len(patient_folders)} patient folders.\n")

    for patient_folder_name in patient_folders:
        eeg_data_path = os.path.join(BASE_PATH, patient_folder_name, "eeg")
        if not os.path.exists(eeg_data_path): continue

        edf_files = [f for f in os.listdir(eeg_data_path) if f.endswith(".edf")]
        if not edf_files: continue

        for edf_file_name in edf_files:
            full_edf_path = os.path.join(eeg_data_path, edf_file_name)
            try:
                raw = mne.io.read_raw_edf(full_edf_path, preload=True)
                stroke_eeg[patient_folder_name] = raw.get_data()
                break  # Only first EDF file
            except Exception as e:
                print(f"Error loading {full_edf_path}: {e}")
                continue

print(f"| Total Patients Loaded: {len(stroke_eeg)}")

# ==========================================
# 2. METADATA + STROKE LOCATION CLEANING
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

# Clean stroke location
def categorize_stroke_location(loc):
    if pd.isna(loc): return 'unclear'
    s = str(loc).lower()
    if 'bilateral' in s or s.strip() == 'pons': return 'bilateral'
    left_c, right_c = s.count('left'), s.count('right')
    if left_c > 0 and right_c > 0: return 'bilateral'
    if left_c > 0: return 'left'
    if right_c > 0: return 'right'
    return 'unclear'

subject_labels_df['StrokeLocation_Category'] = subject_labels_df['StrokeLocation'].apply(categorize_stroke_location)
subject_labels_df = subject_labels_df[subject_labels_df['StrokeLocation_Category'].isin(['left', 'right'])]

# Match with loaded EEG
loaded_subjects = list(stroke_eeg.keys())
final_labels_df = subject_labels_df[subject_labels_df['subject_id'].isin(loaded_subjects)].copy()
print(f"\n✅ Usable subjects after filtering: {len(final_labels_df)}")

# ==========================================
# 3. PREPROCESSING & EPOCHING
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
print("--- Preprocessing ---")

for patient_id, eeg_array in stroke_eeg.items():
    if patient_id not in final_labels_df['subject_id'].values:
        continue
    try:
        info = mne.create_info(ch_names=FINAL_CHANNEL_NAMES_33, sfreq=SFREQ, ch_types='eeg')
        raw = mne.io.RawArray(eeg_array, info, verbose=False)
        raw.set_channel_types({'HEOL': 'eog', 'VEOR': 'eog', 'STATUS1': 'stim', 'STATUS2': 'misc'})
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore', verbose=False)
        if raw.info['sfreq'] != SFREQ:
            raw.resample(SFREQ, verbose=False)
        raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)
        raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)
        raw.set_eeg_reference('average', verbose=False)
        raw_epoched = raw.copy().pick_types(eeg=True)
        events = mne.make_fixed_length_events(raw_epoched, duration=STEP_S, start=0.0)
        epochs = mne.Epochs(raw_epoched, events, tmin=0, tmax=TMAX, baseline=None, preload=True, verbose=False)
        stroke_epochs[patient_id] = epochs
    except Exception as e:
        print(f"Skipping {patient_id}: {e}")

print(f"Total subjects epoched: {len(stroke_epochs)}")

# ==========================================
# 4. FAST TDA FEATURE FUNCTION
# ==========================================

def compute_fast_tda_features(band_power):
    """Fast TDA: H0 persistence from 3D delay embedding of band power (first 10 channels)"""
    try:
        n = len(band_power)
        if n < 3:
            return {'tda_h0_max': 0, 'tda_h0_mean': 0, 'tda_h0_count': 0}
        
        # Use first 10 channels for speed
        x = band_power[:10]
        if len(x) < 3:
            x = np.pad(x, (0, 3 - len(x)), constant_values=0)
        
        # 3D delay embedding
        points = np.column_stack([x, np.roll(x, -1), np.roll(x, -2)])[:len(x)]
        
        # Compute H0 only (connected components)
        dgm = ripser(points, maxdim=0, thresh=2.0)['dgms'][0]
        if len(dgm) == 0:
            return {'tda_h0_max': 0, 'tda_h0_mean': 0, 'tda_h0_count': 0}
        
        pers = dgm[:, 1] - dgm[:, 0]
        pers = pers[np.isfinite(pers) & (pers > 0)]
        if len(pers) == 0:
            return {'tda_h0_max': 0, 'tda_h0_mean': 0, 'tda_h0_count': 0}
        
        return {
            'tda_h0_max': float(np.max(pers)),
            'tda_h0_mean': float(np.mean(pers)),
            'tda_h0_count': int(len(pers))
        }
    except:
        return {'tda_h0_max': 0, 'tda_h0_mean': 0, 'tda_h0_count': 0}

# ==========================================
# 5. FEATURE EXTRACTION (WITH TDA)
# ==========================================

BANDS = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
PAIRS = [('F3', 'F4'), ('C3', 'C4'), ('P3', 'P4'), ('O1', 'O2'), ('F7', 'F8'), ('T3', 'T4'), ('T5', 'T6')]

all_features = []
first_sub = list(stroke_epochs.keys())[0]
ch_names = stroke_epochs[first_sub].ch_names

print("--- Feature Extraction with TDA ---")

for patient_id, epochs in stroke_epochs.items():
    data = epochs.get_data()
    for epoch in data:
        feat = {'subject_id': patient_id}
        freqs, psd = signal.welch(epoch, fs=SFREQ, nperseg=256)
        
        for band, (lf, hf) in BANDS.items():
            mask = (freqs >= lf) & (freqs <= hf)
            band_power = np.trapz(psd[:, mask], freqs[mask], axis=1)
            
            feat[f'{band}_mean'] = np.mean(band_power)
            
            # Asymmetry
            for l_ch, r_ch in PAIRS:
                if l_ch in ch_names and r_ch in ch_names:
                    il, ir = ch_names.index(l_ch), ch_names.index(r_ch)
                    pl, pr = band_power[il], band_power[ir]
                    asym = (pl - pr) / (pl + pr + 1e-9)
                    feat[f'{band}_asym_{l_ch}_{r_ch}'] = asym
            
            # TDA features
            tda = compute_fast_tda_features(band_power)
            for k, v in tda.items():
                feat[f'{band}_{k}'] = v
        
        all_features.append(feat)

feature_df = pd.DataFrame(all_features)
print(f"✅ Feature extraction done. Shape: {feature_df.shape}")

# ==========================================
# 6. PREPARE DATA FOR STROKE LOCATION PREDICTION
# ==========================================

target_col = 'StrokeLocation_Category'
merged_df = feature_df.merge(final_labels_df[['subject_id', target_col]], on='subject_id', how='inner')
merged_df = merged_df.dropna(subset=[target_col])

le = LabelEncoder()
merged_df['target'] = le.fit_transform(merged_df[target_col])
class_names = le.classes_
print(f"Classes: {class_names}")

X = merged_df.drop(columns=['subject_id', 'target', target_col])
y = merged_df['target']
groups = merged_df['subject_id']

# 75-25 subject-wise split
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
test_subjects = groups.iloc[test_idx]

print(f"\nTrain subjects: {len(groups.iloc[train_idx].unique())}")
print(f"Test subjects: {len(test_subjects.unique())}")

# ==========================================
# 7. MODELS & EVALUATION
# ==========================================

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss', use_label_encoder=False),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, class_weight='balanced', random_state=42, verbose=-1),
    'Linear SVM': SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42, probability=True),
    'RBF SVM': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42, probability=True)
}

results_summary = []

for name, clf in models.items():
    print(f"\n{'='*50}\nTraining: {name}\n{'='*50}")
    
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('selector', SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42), threshold='median')),
        ('clf', clf)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred_epoch = pipe.predict(X_test)
    
    # Patient-level aggregation
    res_df = pd.DataFrame({'subject': test_subjects, 'y_true': y_test, 'y_pred': y_pred_epoch})
    patient_preds, patient_trues = [], []
    for subj in res_df['subject'].unique():
        sub_data = res_df[res_df['subject'] == subj]
        patient_trues.append(sub_data['y_true'].iloc[0])
        patient_preds.append(sub_data['y_pred'].mode()[0])  # Majority vote
    
    acc = accuracy_score(patient_trues, patient_preds)
    print(f"✅ Patient Accuracy: {acc:.4f}")
    print(classification_report(patient_trues, patient_preds, target_names=class_names))
    
    results_summary.append({'Model': name, 'Accuracy': acc})
    
    # Confusion matrix
    plt.figure(figsize=(5,4))
    cm = confusion_matrix(patient_trues, patient_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

# ==========================================
# 8. FINAL COMPARISON
# ==========================================

summary = pd.DataFrame(results_summary).sort_values('Accuracy', ascending=False)
print("\n" + "="*60)
print("MODEL COMPARISON (Patient-Level Accuracy)")
print("="*60)
print(summary.to_string(index=False))

plt.figure(figsize=(10,6))
plt.barh(summary['Model'], summary['Accuracy'], color='steelblue')
plt.xlabel('Accuracy'); plt.title('Stroke Location Prediction (with TDA Features)')
plt.xlim(0,1); plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✅ Stroke location prediction with TDA completed successfully!")