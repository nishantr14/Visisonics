from src.fusion import *

print("\n--- BASIC TESTS ---")

print("Case 1: Cyber high, Physical low")
print(fuse_scores(0.9, 0.2))
print(get_risk_score(0.9, 0.2))
print(classify_risk(0.9, 0.2, get_risk_score(0.9, 0.2)[0]))
print(generate_explanation(0.9, 0.2))

print("\nCase 2: Physical high")
print(fuse_scores(0.2, 0.9))
print(get_risk_score(0.2, 0.9))
print(classify_risk(0.2, 0.9, get_risk_score(0.2, 0.9)[0]))
print(generate_explanation(0.2, 0.9))

print("\nCase 3: Both high")
print(fuse_scores(0.9, 0.9))
print(get_risk_score(0.9, 0.9))
print(classify_risk(0.9, 0.9, get_risk_score(0.9, 0.9)[0]))
print(generate_explanation(0.9, 0.9))

print("\nCase 4: Normal")
print(fuse_scores(0.1, 0.1))
print(get_risk_score(0.1, 0.1))
print(classify_risk(0.1, 0.1, get_risk_score(0.1, 0.1)[0]))
print(generate_explanation(0.1, 0.1))
print("\n--- EDGE CASES ---")
print(fuse_scores(-1, 2))  # should clamp to 0–1
import random

for _ in range(5):
    c = random.random()
    p = random.random()
    score, _ = get_risk_score(c, p)
    print(c, p, "->", score)