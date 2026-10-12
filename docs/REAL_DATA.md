# Using real data (MIMIC-IV, MIMIC-CXR, PhysioNet)

ClinicalFusion ships with a synthetic cohort so the whole pipeline can be built, tested and demonstrated
without any restricted data. Synthetic results say **nothing** about clinical performance. To run on
real data you prepare a cohort directory and point the platform at it.

## 1. Get access and check permissions first

* MIMIC-IV, MIMIC-CXR(-JPG) and most PhysioNet clinical datasets are **credentialed**: you need a PhysioNet
  account, the required human-subjects training, and a signed data use agreement.
* The agreements restrict sharing and, for some datasets, how the data may be processed (e.g. sending it to
  third-party hosted AI services is generally **not** permitted). Run ClinicalFusion locally or inside an
  approved environment, and do not deploy the API with real data on a public host.
* Different datasets are **not** automatically linkable. MIMIC-CXR and MIMIC-IV share `subject_id` values
  because they come from the same hospital, but you must verify the linkage, the time alignment of each
  X-ray with the clinical record, and that your intended use is permitted. Report the cohort overlap
  (`clinicalfusion.data.validation.match_patient_ids`) before training.

## 2. Cohort layout

```
cohort/
  labels.csv         patient_id,y                 (y in {0,1}; one row per patient/stay)
  tabular.csv        patient_id + 12 columns       age,sex,bmi,sbp,dbp,heart_rate_adm,creatinine,lactate,wbc,hemoglobin,sodium,glucose
  timeseries.npz     patient_id (M,), values (M, 2, 48)    hourly grid; channels = heart_rate, spo2; NaN = not measured
  images/<patient_id>.png     chest X-ray, grayscale (resized to 64x64 by default)
```

A patient absent from `tabular.csv`, `timeseries.npz` or `images/` is treated as **missing that modality**; this
is exactly the situation the fusion models and the missing-modality analysis are built for.

`clinicalfusion export-synthetic --out cohort_demo` writes a synthetic cohort in this layout so you can see it.

## 3. Mapping from MIMIC (a sketch, not a turnkey script)

| ClinicalFusion field | Typical MIMIC source | Things you must decide |
|---|---|---|
| `y` | outcome from `admissions` / `icustays` (e.g. in-hospital mortality) | prediction time, exclusion criteria, avoiding label leakage |
| tabular | `labevents`, first `chartevents` values, `patients`, `admissions` | use only values observed **before** the prediction time |
| `timeseries` | `chartevents` heart rate / SpO2 (itemids differ by MIMIC version) | resample irregular charting onto an hourly grid; window = the 48 h before the prediction time |
| images | MIMIC-CXR-JPG studies linked by `subject_id` and study time | choose one study per stay, within a defined window; use the frontal view |

Splitting is at the patient (row) level. If one person can appear with several stays, build your cohort with one
row per person or extend `stratified_split` to be group-aware, otherwise the same patient can land in both
train and test.

## 4. Run

```bash
clinicalfusion run --preset standard --data-dir /path/to/cohort --out runs/mimic_v1 --mlflow
```

or set `CF_DATA_DIR` for the API and tick "use server data dir" in a run request. The API never accepts a
filesystem path from a client.

## 5. Interpreting results responsibly

* Report the held-out test numbers with confidence intervals, the calibration, and the missing-data analysis,
  including when fusion does **not** beat the best single modality.
* Image models trained on small or single-site data often learn shortcuts (scanner, markers, patient position).
  Grad-CAM can reveal this but cannot prove the model is clinically right.
* Attributions, heatmaps and attention are descriptions of model behaviour, not evidence of causation.
* This software is a research tool, not a medical device.
