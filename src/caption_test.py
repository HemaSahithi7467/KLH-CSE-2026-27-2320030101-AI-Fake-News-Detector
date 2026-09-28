import joblib

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# 12 short, Instagram-caption-style examples
# label: 1 = Real (verified), 0 = Fake (debunked)
examples = [
    (0, "BREAKING: World leaders including Xi Jinping fell seriously ill right after the BRICS Summit in India, sources claim."),
    (0, "Viral photo shows Kharge deliberately kept away from Rahul Gandhi's breakfast table -- shocking snub caught on camera!"),
    (0, "Leaked NASA document 'Project Anchor' reveals secret cosmic event was hidden from the public since 2024."),
    (0, "Video proves woman bathing in the Ganga in a bikini went viral -- outrage across the internet."),
    (0, "US Treasury just declared the government is officially insolvent, shocking new report reveals."),
    (0, "ICE agents are secretly paid a bonus for every immigrant they arrest, whistleblower claims."),
    (1, "MEA fact-check confirms only South African President Ramaphosa fell ill after the BRICS Summit; other claims were false."),
    (1, "Tap water supply in a Kolkata neighborhood has remained disrupted for over 45 days, residents say."),
    (1, "About 60 trees were removed from a Washington D.C. park as part of routine maintenance work, officials confirmed."),
    (1, "A viral video allegedly showing a recent incident was found to actually be from several months earlier, fact-checkers confirm."),
    (1, "Authorities say only one confirmed case was verified despite widespread claims of a larger incident."),
    (1, "Officials clarified that a proposed bonus program for agents was scrapped before being implemented."),
]

correct = 0
for true_label, text in examples:
    tfidf = vectorizer.transform([text])
    pred = model.predict(tfidf)[0]
    conf = max(model.predict_proba(tfidf)[0]) * 100
    is_correct = "✓" if pred == true_label else "✗"
    if pred == true_label:
        correct += 1
    label_name = "REAL" if pred == 1 else "FAKE"
    true_name = "REAL" if true_label == 1 else "FAKE"
    print(f"{is_correct} Predicted: {label_name} ({conf:.1f}%) | Actual: {true_name} | {text[:60]}...")

print(f"\nAccuracy on short caption-style text: {correct}/{len(examples)} = {correct/len(examples)*100:.1f}%")