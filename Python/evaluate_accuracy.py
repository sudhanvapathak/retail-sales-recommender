import pandas as pd
from predict import RetailRecommender

# Load data + model
df = pd.read_csv('/Users/sudhanva/Documents/retail_sales_reccomender/retail_sales_clean.csv')
model = RetailRecommender()

# Test on LAST transaction per customer (simulates "predict next purchase")
test_data = df.groupby('customer_id').last().reset_index()
print(f"🧪 Testing on {len(test_data)} holdout transactions")

hits = 0
for idx, row in test_data.iterrows():
    # Model prediction
    recs = model.recommend(row['age_group'], row['gender'], row['day_of_week'])
    top_rec = recs[0]['product_category'] if recs[0]['confidence'] != 'Default' else None
    
    # Did customer actually buy what we recommended?
    actual = row['product_category']
    if top_rec == actual:
        hits += 1

hit_rate = hits / len(test_data)
print(f"\n🎯 Hit Rate: {hit_rate:.1%} ({hits}/{len(test_data)})")
print(f"✅ Model correctly predicted {hits} out of {len(test_data)} next purchases")
