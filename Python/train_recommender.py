import pandas as pd
import numpy as np
import pickle
import os

csv_path = '/Users/sudhanva/Documents/retail_sales_reccomender/retail_sales_clean.csv'
print("Loading the transaction data...")
df = pd.read_csv(csv_path)

print(f" Loaded {len(df)} transactions from {df['age_group'].nunique()} age groups")

segment_rules = df.groupby(['age_group', 'gender', 'day_of_week', 'product_category', 'price_range']).agg({
    'transaction_id': 'count',  # frequency
    'total_amount': 'mean'
}).reset_index()


segment_rules['segment_size'] = segment_rules.groupby(['age_group', 'gender', 'day_of_week'])['transaction_id'].transform('sum')

segment_rules['confidence'] = segment_rules['transaction_id'] / segment_rules['segment_size']
top_rules = segment_rules[segment_rules['confidence'] >= 0.05].nlargest(100, 'confidence')

print(f"✅ Trained {len(top_rules)} recommendation rules")
print("\nTop 5 rules:")
print(top_rules[['age_group', 'gender', 'day_of_week', 'product_category', 'price_range', 'confidence']].head())

os.makedirs('ai_recommender', exist_ok=True)
top_rules.to_csv('ai_recommender/rules.csv', index=False)
with open('ai_recommender/model.pkl', 'wb') as f:
    pickle.dump(top_rules, f)