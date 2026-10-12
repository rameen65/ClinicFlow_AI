# Data sources and reuse

## Simulated clinic sample

- File: `data/simulated/appointments_demo.csv`
- Size: 1,200 synthetic rows.
- Origin: retained from the provided ClinicFlow AI Java submission.
- Use: interface walkthrough and a small model demonstration only.
- The generated patterns and labels do not represent a real clinic or patient population.

## Public appointment no-show sample

This GitHub copy intentionally excludes the row-level CSV to avoid publishing appointment-level records. The source and license are documented below; download the dataset separately for local evaluation.


- File: `data/public/KaggleV2-May-2016.csv`
- Dataset: **Medical Appointment No Shows**, published on Kaggle by Joni Hoppen (`joniarroba`).
- Primary dataset page and license: <https://www.kaggle.com/datasets/joniarroba/noshowappointments>
- License shown on the Kaggle dataset page: **CC BY-NC-SA 4.0**. Reuse must give attribution, remain non-commercial, and share adaptations under the same terms. Read the license before redistributing or using the file commercially: <https://creativecommons.org/licenses/by-nc-sa/4.0/>.
- Public download mirror used for the CSV: <https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dd2e9a_noshowappointments-kagglev2-may-2016/noshowappointments-kagglev2-may-2016.csv>.
- Verification: CSV has the documented 14 original columns, 110,527 appointment records plus a header, and is 10,739,535 bytes. SHA-256: `9132D3E7D0246617DF9041D3764F20AD6F08E7B0D9F0997FA254FC5E52EDA27`.
- The dataset describes appointments in Brazil in 2016. It includes pseudonymous patient IDs and sensitive demographic/health attributes. The app does not show those IDs or use them as model features; `PatientId` is used only to keep the same person's rows on one side of the train/test split.
- This historical sample is not representative evidence for a present-day clinic. Its inclusion does not imply clinical suitability.

The two sources are selected separately in the app and are never concatenated.
