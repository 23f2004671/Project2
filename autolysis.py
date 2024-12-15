import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import argparse

# Set up OpenAI API key from environment variable
openai.api_key = os.getenv("AIPROXY_TOKEN")

def load_data(file_path):
    """Load a CSV file into a pandas DataFrame."""
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded dataset: {file_path}")
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        exit(1)

def basic_analysis(data):
    """Perform basic statistical analysis on the dataset."""
    summary = data.describe(include="all").to_dict()
    missing_values = data.isnull().sum().to_dict()
    return {"summary": summary, "missing_values": missing_values}

def detect_outliers(data):
    """Detect outliers in numeric columns using the IQR method."""
    outliers = {}
    for col in data.select_dtypes(include=["float", "int"]).columns:
        Q1, Q3 = data[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers[col] = data[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)].index.tolist()
    return outliers

def cluster_analysis(data):
    """Perform clustering on numeric data."""
    numeric_data = data.select_dtypes(include=["float", "int"]).dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(scaled_data)
    return kmeans.labels_

def create_visualizations(data, output_dir):
    """Generate visualizations and save them as PNG files."""
    os.makedirs(output_dir, exist_ok=True)

    # Correlation matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()

    # Histograms for numeric columns
    for col in data.select_dtypes(include=["float", "int"]).columns:
        plt.figure()
        sns.histplot(data[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"{output_dir}/{col}_distribution.png")
        plt.close()

def generate_readme(data_summary, outliers, insights, output_dir):
    """Generate a README.md file summarizing the dataset analysis."""
    readme_content = f"""
# Automated Analysis Results

## Dataset Summary
{data_summary}

## Outlier Detection
{outliers}

## Insights
{insights}

## Visualizations
![Correlation Matrix](correlation_matrix.png)
"""
    with open(f"{output_dir}/README.md", "w") as f:
        f.write(readme_content)

def prompt_llm(prompt, max_tokens=500):
    """Query the OpenAI LLM to generate insights."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

def process_dataset(file_path):
    """Main function to process a single dataset."""
    data = load_data(file_path)
    analysis = basic_analysis(data)
    outliers = detect_outliers(data)
    clusters = cluster_analysis(data)

    output_dir = file_path.split(".")[0]
    create_visualizations(data, output_dir)

    insights = prompt_llm(f"Analyze the following data insights: {analysis} and {outliers}")
    generate_readme(analysis, outliers, insights, output_dir)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Automated Data Analysis with LLMs")
    parser.add_argument("csv_files", nargs='+', type=str, help="Paths to one or more CSV files")
    args = parser.parse_args()

    for file_path in args.csv_files:
        process_dataset(file_path)

if __name__ == "__main__":
    main()
