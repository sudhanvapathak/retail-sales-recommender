from predict import RetailRecommender

model = RetailRecommender()

test_scenarios = [
    ("20-30", "Male", "Saturday", "Young weekend shoppers"),
    ("30-40", "Female", "Friday", "Working professionals"),
    ("50-60", "Male", "Sunday", "Weekend run"),
    ("30-40","Male","Tuesday","Weekday Run"),
]

print("🏪 RETAIL SMART RECOMMENDER DEMO\n")
for age, gender, day, desc in test_scenarios:
    print(f"{desc}:")
    recs = model.recommend(age, gender, day)
    for rec in recs:
        print(f"  → {rec['product_category']} ({rec['price_range']}) [{rec['confidence']}]")
    print()
