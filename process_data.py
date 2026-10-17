import os
import pandas as pd
import numpy as np

from model_architecture import extract_image_features

def process_and_fuse_data(
    metadata_csv="banana_metadata_v3.csv",
    gas_csv="gas_features.csv",
    image_dir="IR_fusion_images",
    output_csv="fused_multimodal_features.csv"
):
    print("Step 1: Loading metadata and gas features...")
    df_meta = pd.read_csv(metadata_csv)
    df_gas = pd.read_csv(gas_csv)

    # Clean whitespace
    df_meta['Day'] = df_meta['Day'].str.strip()
    df_meta['Session'] = df_meta['Session'].str.strip()
    df_gas['Day'] = df_gas['Day'].str.strip()
    df_gas['Session'] = df_gas['Session'].str.strip()

    # Session encoding mapping
    session_encoder = {'Morning': 0, 'Evening': 1, 'Night': 2}

    print("Step 2: Processing thermal images per session and mean pooling 128-d vectors...")
    aggregated_records = []

    # Group by Day and Session
    grouped = df_meta.groupby(['Day', 'Session'], sort=False)

    for (day_str, session_str), group in grouped:
        print(f" -> Processing {day_str} - {session_str} ({len(group)} rows)...")
        image_vectors = []

        for _, row in group.iterrows():
            filename = row['Filename']
            if pd.notna(filename) and isinstance(filename, str):
                img_path = os.path.join(image_dir, filename)
                if os.path.exists(img_path):
                    vector = extract_image_features(img_path)
                    image_vectors.append(vector)
                else:
                    print(f"    Warning: Image {img_path} not found.")

        if image_vectors:
            # Mean pooling of 128-d vectors for the session
            session_image_vector = np.mean(image_vectors, axis=0)
        else:
            print(f"    Notice: No valid image files for {day_str} {session_str}. Using zero vector imputation.")
            session_image_vector = np.zeros(128)

        # Ambient temperatures average
        temp1_avg = group['Temp1_C'].mean() if pd.notna(group['Temp1_C'].mean()) else 30.0
        temp2_avg = group['Temp2_C'].mean() if pd.notna(group['Temp2_C'].mean()) else 32.0

        # Day numeric value extraction (e.g. 'Day 1' -> 1)
        day_num = int(day_str.replace('Day', '').strip())
        session_encoded = session_encoder.get(session_str, 0)

        # Spoilage targets calculation (days remaining to threshold)
        # Day 4 = spoil_50, Day 5 = spoil_75, Day 6 = spoil_90, Day 7 = spoil_95
        spoil_50 = max(0.0, 4.0 - day_num)
        spoil_75 = max(0.0, 5.0 - day_num)
        spoil_90 = max(0.0, 6.0 - day_num)
        spoil_95 = max(0.0, 7.0 - day_num)

        # Base record
        record = {
            'Day': day_str,
            'Day_Num': day_num,
            'Session': session_str,
            'Session_Encoded': session_encoded,
            'Temp1_C': temp1_avg,
            'Temp2_C': temp2_avg,
            'spoil_50': spoil_50,
            'spoil_75': spoil_75,
            'spoil_90': spoil_90,
            'spoil_95': spoil_95
        }

        # Add 128 image feature dimensions
        for i in range(128):
            record[f'img_feat_{i}'] = session_image_vector[i]

        aggregated_records.append(record)

    df_aggregated = pd.DataFrame(aggregated_records)

    print("Step 3: Merging aggregated image vectors with gas features...")
    fused_df = pd.merge(df_aggregated, df_gas, on=['Day', 'Session'], how='left')

    # Fill any missing gas features if any
    fused_df = fused_df.ffill().bfill()

    print(f"Step 4: Exporting fused multimodal dataset to {output_csv}...")
    fused_df.to_csv(output_csv, index=False)
    print(f"Successfully exported {len(fused_df)} session vectors to {output_csv}.")
    return fused_df

if __name__ == "__main__":
    process_and_fuse_data()
