import re
import pandas as pd
import numpy as np

def generate_gas_features():
    with open('Methane_gas_readings.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    case_map = {1: 'Morning', 2: 'Evening', 3: 'Night'}
    records = []

    days = re.split(r'Day\s+(\d+)', text)
    for i in range(1, len(days), 2):
        day_num = int(days[i])
        day_str = f"Day {day_num}"
        content = days[i+1]
        cases = re.split(r'case\s+(\d+)', content)

        for j in range(1, len(cases), 2):
            case_num = int(cases[j])
            session_str = case_map.get(case_num, f'Case_{case_num}')
            case_content = cases[j+1]

            r_match = re.search(r'R\s*([\s\S]*?)(?=L|-null-|$)', case_content)
            l_match = re.search(r'L\s*([\s\S]*?)(?=-null-|$)', case_content)

            r_vals = [float(x) for x in re.findall(r'Methane:\s*([\d.]+)\s*ppm', r_match.group(1))] if r_match else []
            l_vals = [float(x) for x in re.findall(r'Methane:\s*([\d.]+)\s*ppm', l_match.group(1))] if l_match else []

            ppm_left_mean = np.mean(l_vals) if len(l_vals) > 0 else np.nan
            ppm_right_mean = np.mean(r_vals) if len(r_vals) > 0 else np.nan
            ppm_left_std = np.std(l_vals) if len(l_vals) > 0 else np.nan
            ppm_right_std = np.std(r_vals) if len(r_vals) > 0 else np.nan

            records.append({
                'Day': day_str,
                'Session': session_str,
                'ppm_left_mean': ppm_left_mean,
                'ppm_right_mean': ppm_right_mean,
                'ppm_left_std': ppm_left_std,
                'ppm_right_std': ppm_right_std,
                'ppm_delta_mean': (ppm_left_mean - ppm_right_mean) if not np.isnan(ppm_left_mean) and not np.isnan(ppm_right_mean) else np.nan
            })

    df_gas = pd.DataFrame(records)

    # Fill any missing NaN values (like Day 1 Night -null-) with global or forward fill
    df_gas = df_gas.ffill().bfill()

    df_gas.to_csv('gas_features.csv', index=False)
    print("gas_features.csv generated successfully.")
    print(df_gas)

if __name__ == "__main__":
    generate_gas_features()
