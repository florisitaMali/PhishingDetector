import pandas as pd

BASE_DATASET = "spam_ham_dataset.csv"
NEW_EMAILS = "emails_clean.csv"
OUTPUT = "spam_ham_dataset_merged.csv"

# =========================
# LOAD DATA
# =========================
df_base = pd.read_csv(BASE_DATASET)
df_new = pd.read_csv(NEW_EMAILS)

# =========================
# FIX BASE DATASET
# =========================
# Remove auto index column if present
if "Unnamed: 0" in df_base.columns:
    df_base = df_base.drop(columns=["Unnamed: 0"])

# Create ID if missing
if "id" not in df_base.columns:
    df_base = df_base.reset_index(drop=True)
    df_base["id"] = df_base.index

# =========================
# FIX NEW EMAILS
# =========================
if "text" not in df_new.columns:
    raise ValueError("emails_clean.csv must contain a 'text' column")

if "label_num" not in df_new.columns:
    if "label" in df_new.columns:
        df_new["label_num"] = df_new["label"].map({"ham": 0, "spam": 1})
    else:
        raise ValueError("emails_clean.csv must contain 'label' or 'label_num'")

if "label" not in df_new.columns:
    df_new["label"] = df_new["label_num"].map({0: "ham", 1: "spam"})

# =========================
# REMOVE DUPLICATES
# =========================
df_new = df_new[~df_new["text"].isin(df_base["text"])]

# =========================
# ASSIGN IDS
# =========================
next_id = df_base["id"].max() + 1
df_new = df_new.copy()
df_new["id"] = range(next_id, next_id + len(df_new))

# =========================
# ALIGN SCHEMA
# =========================
df_base = df_base[["id", "label", "text", "label_num"]]
df_new = df_new[["id", "label", "text", "label_num"]]

# =========================
# MERGE
# =========================
df_final = pd.concat([df_base, df_new], ignore_index=True)

# =========================
# SAVE
# =========================
df_final.to_csv(OUTPUT, index=False)

print("✅ Merge successful")
print("Original size:", len(df_base))
print("Added emails:", len(df_new))
print("Final size:", len(df_final))
