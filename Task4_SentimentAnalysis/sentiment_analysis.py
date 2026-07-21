# ============================================================
# Project      : Sentiment Analysis using TextBlob
# Internship   : CodeAlpha Data Analytics Internship
# Author       : Ganesh Jadhav
#
# Description:
# This project analyzes customer reviews and classifies
# them as Positive, Negative, or Neutral using TextBlob.
# ============================================================


# ============================================================
# 1. Import Required Library
# ============================================================

from textblob import TextBlob


# ============================================================
# 2. Sample Customer Reviews
# ============================================================

reviews = [
    "This product is amazing.",
    "I love this phone.",
    "Very bad experience.",
    "The service was excellent.",
    "Worst product ever.",
    "It is okay."
]


# ============================================================
# 3. Initialize Sentiment Counters
# ============================================================

positive_count = 0
negative_count = 0
neutral_count = 0


# ============================================================
# 4. Start Sentiment Analysis
# ============================================================

print("=" * 70)
print("SENTIMENT ANALYSIS USING TEXTBLOB")
print("=" * 70)

for index, review in enumerate(reviews, start=1):

    analysis = TextBlob(review)

    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity

    if polarity > 0:
        sentiment = "Positive"
        positive_count += 1

    elif polarity < 0:
        sentiment = "Negative"
        negative_count += 1

    else:
        sentiment = "Neutral"
        neutral_count += 1

    print(f"\nReview {index}")
    print("-" * 50)
    print(f"Text         : {review}")
    print(f"Polarity     : {polarity:.2f}")
    print(f"Subjectivity : {subjectivity:.2f}")
    print(f"Sentiment    : {sentiment}")


# ============================================================
# 5. Display Analysis Summary
# ============================================================

print("\n" + "=" * 70)
print("SENTIMENT ANALYSIS SUMMARY")
print("=" * 70)

print(f"Total Reviews    : {len(reviews)}")
print(f"Positive Reviews : {positive_count}")
print(f"Negative Reviews : {negative_count}")
print(f"Neutral Reviews  : {neutral_count}")


# ============================================================
# 6. Key Insights
# ============================================================

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

print("1. Positive reviews indicate customer satisfaction.")
print("2. Negative reviews indicate dissatisfaction.")
print("3. Neutral reviews do not express strong emotion.")
print("4. Polarity determines the overall sentiment.")
print("5. Subjectivity shows whether the review is opinion-based.")


# ============================================================
# 7. Completion Message
# ============================================================

print("\n" + "=" * 70)
print("SENTIMENT ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)