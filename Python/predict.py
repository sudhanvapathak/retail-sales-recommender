import pandas as pd
import pickle

class RetailRecommender:
    def __init__(self, model_path='ai_recommender/model.pkl'):
        with open(model_path, 'rb') as f:
            self.rules = pickle.load(f)
        print(f"🤖 Loaded {len(self.rules)} rules")
    
    def recommend(self, age_group, gender, day_of_week, top_k=3):
        matches = self.rules[
            (self.rules['age_group'] == age_group) &
            (self.rules['gender'] == gender) &
            (self.rules['day_of_week'] == day_of_week)
        ].nlargest(top_k, 'confidence')
        
        if matches.empty:
            return [{"product_category": "Beauty", "price_range": "Economy (50-100)", "confidence": "Default"}]
        
        recs = []
        for _, row in matches.iterrows():
            recs.append({
                'product_category': row['product_category'],
                'price_range': row['price_range'],
                'confidence': f"{row['confidence']:.1%}",
                'avg_amount': f"${row['total_amount']:.0f}"
            })
        return recs

# Quick test
if __name__ == "__main__":
    model = RetailRecommender()
    print("\n🎯 Example predictions:")
    print(model.recommend("30-40", "Male", "Saturday"))
    print(model.recommend("20-30", "Female", "Friday"))
