from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

app = FastAPI(title="Sentiment Analysis API by Gatsby")

texts = [
    
    "I love this product", "great amazing wonderful", "very good excellent",
    "fantastic awesome", "perfect excellent", "I like it", "super good",
    "absolutely brilliant", "outstanding quality", "beyond expectations",
    "highly recommend", "best purchase ever", "couldnt be better",
    "exceeds expectations", "truly impressive", "very satisfied",
    "worth every penny", "excellent value", "top notch", "five stars",
    "remarkable experience", "delighted with purchase", "would buy again",
    "pleasantly surprised", "exceeded my hopes", "flawless performance",
    "amazing quality", "very happy", "great value", "works perfectly",
    "beautiful design", "very pleased", "exceptional service", "quick delivery",
    "responsive support", "easy to use", "user friendly", "intuitive interface",
    "powerful features", "reliable product", "durable material", "comfortable fit",
    "great battery life", "fast shipping", "secure packaging", "clear instructions",
    "helpful staff", "professional service", "smooth transaction", "fair price",
    "excellent choice", "wise decision", "happy customer", "positive experience",
    "will buy more", "good investment", "solid performance", "decent quality",
    "above average", "good enough", "satisfactory result", "nice addition",
    "cool feature", "fun to use", "enjoyable experience", "pleasure to use",
    "works like charm", "does the job well", "efficient tool", "time saver",
    "life changer", "game changer", "must have", "highly effective", "proven results",
    "trusted brand", "industry leader", "innovative design", "state of the art",
    "cutting edge", "ahead of competition", "best in class", "number one",
    "top seller", "customer favorite", "award winning", "critically acclaimed",
    "rave reviews", "highly rated", "bestseller", "popular choice", "trendy item",
    "stylish look", "elegant design", "luxurious feel", "premium quality",
    "professional grade", "commercial standard", "enterprise ready", "rock solid",
    "sturdy construction", "heavy duty", "long lasting", "built to last",
    "energy efficient", "eco friendly", "sustainable product", "ethical company",
    "great support", "easy setup", "simple installation", "quick start",
    "seamless integration", "compatible device", "universal fit", "standard size",
    "light weight", "compact design", "space saving", "portable device",
    "wireless connection", "bluetooth enabled", "smart technology", "automatic mode",
    "voice control", "app integration", "cloud sync", "real time updates",
    "accurate reading", "precise measurement", "consistent results", "reliable data",
    "secure connection", "private data", "safe to use", "child friendly",
    "pet friendly", "travel friendly", "weather resistant", "waterproof case",
    "shock proof", "scratch resistant", "easy to clean", "low maintenance","nice one"
    
    



    "I hate this product", "terrible awful bad", "very poor horrible",
    "disgusting worst ever", "not good at all", "very bad",
    "complete waste of money", "total disappointment", "never buy again",
    "faulty product", "broken on arrival", "stopped working quickly",
    "poor quality control", "cheap material", "fell apart easily",
    "terrible customer service", "rude staff", "unhelpful support",
    "slow shipping", "lost package", "damaged delivery",
    "impossible to use", "confusing instructions", "bad design",
    "worse than expected", "below standards", "not worth price",
    "overpriced garbage", "useless item", "defective unit",
    "does not work", "frequent crashes", "software bugs", "glitchy app",
    "battery drains fast", "overheats constantly", "loud noise", "bad smell",
    "uncomfortable fit", "too small", "too large", "wrong size",
    "missing parts", "incomplete set", "misleading description",
    "false advertising", "scam alert", "fake product", "counterfeit item",
    "no refund policy", "return denied", "warranty void", "no support",
    "spam emails", "hidden fees", "subscription trap", "billing issues",
    "data breach", "privacy concern", "security risk", "unsafe product",
    "health hazard", "toxic material", "allergic reaction", "caused rash",
    "dangerous design", "fire hazard", "electrical shock", "sharp edges",
    "choking hazard", "not child safe", "illegal product", "banned item",
    "expired product", "rotten food", "moldy item", "rusty parts",
    "leaking container", "spilled content", "messy application", "difficult cleanup",
    "stains clothes", "ruined carpet", "damaged furniture", "scratched surface",
    "poor adhesion", "peeling off", "fading color", "rusts quickly",
    "breaks easily", "cracked screen", "loose button", "stuck key",
    "loud fan", "vibrates too much", "unstable stand", "wobbly legs",
    "uneven surface", "crooked cut", "misaligned parts", "gaps between pieces",
    "sticky residue", "bad odor", "chemical smell", "tastes awful","bad one"
]

positive_examples = texts[:150]
negative_examples = texts[-150:]

texts = positive_examples + negative_examples
labels = [1] * 150 + [0] * 150

print(f"Dataset is ready : {len(texts)} exemples")
print(f"   - Positifs: {sum(labels)}")
print(f"   - Négatifs: {len(labels) - sum(labels)}")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

texts_clean = [clean_text(t) for t in texts]


vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X = vectorizer.fit_transform(texts_clean)
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X, labels)

print(f"Model trained on {len(texts)} exemples, not successfully cause not enouth data")


class TextInput(BaseModel):
    text: str

class SentimentOutput(BaseModel):
    sentiment: str
    score: float

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API - 300 examples model", "status": "running"}

@app.post("/predict", response_model=SentimentOutput)
def predict_sentiment(input_data: TextInput):
    text_clean = clean_text(input_data.text)
    X_input = vectorizer.transform([text_clean])
    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    score = float(max(proba))
    
    return SentimentOutput(sentiment=sentiment, score=score)

@app.get("/health")
def health():
    return {"status": "good", "positive_examples": 150, "negative_examples": 150, "total": 300}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
