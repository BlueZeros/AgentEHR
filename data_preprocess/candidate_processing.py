import pandas as pd
import sqlite3
import os


def extract_unique_to_db(df, columns, table_name, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    unique_values = df[columns].dropna().unique()
    
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.execute(f"CREATE TABLE {table_name} (candidate TEXT PRIMARY KEY)")
    
    for value in unique_values:
        cursor.execute(f"INSERT INTO {table_name} (candidate) VALUES (?)", (str(value),))
    
    print(f"表 {table_name}: {len(unique_values)} 个不同值")
    
    conn.commit()
    conn.close()


def convert_d_to_db(df, column_name, table_name, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    processed_df = df.rename(columns={column_name: "candidate"})
    
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    columns_sql = ", ".join([f"{col} TEXT" for col in processed_df.columns])
    create_table_sql = f"CREATE TABLE {table_name} ({columns_sql})"
    cursor.execute(create_table_sql)

    for _, row in processed_df.dropna(subset=["candidate"]).iterrows():
        placeholders = ", ".join(["?"] * len(processed_df.columns))
        insert_sql = f"INSERT INTO {table_name} ({', '.join(processed_df.columns)}) VALUES ({placeholders})"
        cursor.execute(insert_sql, tuple(str(x) if pd.notnull(x) else "" for x in row))
    print(f"表 {table_name}: {processed_df['candidate'].count()} 条记录（未去重）")

    conn.commit()
    conn.close()


def main():
    db_path = "/home/ma-user/work/liaoyusheng/projects/EHRAgent/datas/datas/sample/db/patients_new/candidate_table.db"
    pref_path = "/home/ma-user/work/liaoyusheng/projects/EHRAgent/datas/mnt/petrelfs/liaoyusheng/projects/ClinicalAgent/MIMICIV-2.2/"

    file_names = ["hosp/microbiologyevents", "note/radiology_detail", "hosp/transfers"]
    column_names = ["test_name", "field_value", "careunit"]
    table_names = ["microbiologyevents_candidates", "radiology_candidates", "transfers_candidates"]

    for i, (file_name, column_name) in enumerate(zip(file_names, column_names)):
        df = pd.read_csv(os.path.join(pref_path, f"{file_name}.csv"))
        if file_name == "note/radiology_detail":
            df = df[df["field_name"] == "exam_name"]
        elif file_name == "hosp/transfers":
            df = df[df["eventtype"] == "transfer"]

        extract_unique_to_db(df, column_name, table_names[i], db_path)
    

    d_files = ["hosp/d_ccs_diagnoses", "hosp/d_ccs_procedures", "hosp/d_labitems", "hosp/d_atc_prescriptions"]
    d_table_names = ["diagnoses_ccs_candidates", "procedures_ccs_candidates", "labevents_candidates", "prescriptions_atc_candidates"]
    d_column_names = ["long_title", "long_title", "label", "name"]
    for i, file_name in enumerate(d_files):
        df = pd.read_csv(os.path.join(pref_path, f"{file_name}.csv"))
        convert_d_to_db(df, d_column_names[i], d_table_names[i], db_path)

if __name__ == "__main__":
    main()

    """
    pref_path = "/remote-home/chuanxuan/datas/mnt/petrelfs/liaoyusheng/projects/ClinicalAgent/MIMICIV-2.2"
    file_name = "hosp/d_labitems"
    df = pd.read_csv(os.path.join(pref_path, f"{file_name}.csv"))
    # 找到label重复的记录
    duplicate_labels = df[df.duplicated(subset=['label'], keep=False)]
    
    if not duplicate_labels.empty:
        print(f"发现 {len(duplicate_labels)} 条重复的label记录:")
        print(duplicate_labels[['itemid', 'label']].sort_values('label'))
    else:
        print("没有发现重复的label")
    """