import pandas as pd

# Load Excel
file_path = "Skill Tag Groups.xlsx"  # Update path if needed
excel_data = pd.read_excel(file_path, sheet_name=None)

# Prepare output list
output = []

# Process all sheets
for sheet_name, df in excel_data.items():
    for i in range(len(df)):
        row = df.iloc[i]
        if str(row[0]).strip().lower() == "business type":
            business_type = row[1]  # Cell to the right
            business_group = df.iloc[i - 1, 1] if i > 0 else None  # Cell above
            if pd.notna(business_type) and pd.notna(business_group):
                output.append(
                    {
                        "Business Group": str(business_group).strip(),
                        "Business Type": str(business_type).strip(),
                    }
                )

# Save to CSV
output_df = pd.DataFrame(output)
output_df.to_csv("business_group_type_pairs.csv", index=False)

print("Saved to business_group_type_pairs.csv")
