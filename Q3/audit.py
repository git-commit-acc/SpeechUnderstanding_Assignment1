import os
import pandas as pd
import matplotlib.pyplot as plt

def audit_librispeech(metadata_path):
    # Field names as defined in the file header
    columns = ['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME']
    
    try:
        # Read file, ignoring lines starting with ';' (comments)
        df = pd.read_csv(
            metadata_path, 
            sep='|', 
            comment=';', 
            names=columns, 
            skipinitialspace=True,
            engine='python',
            on_bad_lines='skip'
        )

        missing = df.isnull().sum()
        print("Missing Metadata:\n", missing)
        
        # Clean white space and ensure column names match exactly
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        # Audit for the specific project subset: train-clean-100
        subset_df = df[df['SUBSET'].str.contains('train-clean-100', na=False, case=False)]
        
        if subset_df.empty:
            print("Warning: No matching rows for 'train-clean-100' found.")
            return

        # Identification of Bias: Gender and Representation
        gender_counts = subset_df['SEX'].value_counts()
        total_minutes = subset_df.groupby('SEX')['MINUTES'].sum()
        
        print("--- Audit Results: train-clean-100 ---")
        print(f"Total Speakers: {len(subset_df)}")
        print(f"Gender Breakdown:\n{gender_counts}")
        print(f"Minutes Distribution:\n{total_minutes}")
        
        # Visualize "Documentation Debt" and Bias
        plt.figure(figsize=(10, 5))
        
        # Plot 1: Representation Count
        plt.subplot(1, 2, 1)
        gender_counts.plot(kind='bar', color=['#3498db', '#e74c3c'])
        plt.title("Speaker Representation (Count)")
        plt.ylabel("Number of Readers")
        
        # Plot 2: Resource Bias (Minutes)
        plt.subplot(1, 2, 2)
        total_minutes.plot(kind='pie', autopct='%1.1f%%', colors=['#3498db', '#e74c3c'])
        plt.title("Audio Duration Distribution (Minutes)")
        
        os.makedirs("q3", exist_ok=True)
        plt.tight_layout()
        plt.savefig("q3/audit_plots.pdf")
        print("\nAudit complete! Review 'q3/audit_plots.pdf' for your report.")

    except Exception as e:
        print(f"Audit failed: {e}")

if __name__ == "__main__":
    path = r"D:\Projects\SpeechUnderstanding\M25DE1035\Assignment1\Q2\data\LibriSpeech\SPEAKERS.TXT"
    audit_librispeech(path)