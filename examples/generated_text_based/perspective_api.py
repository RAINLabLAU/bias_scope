# --------------------------------------------------------------
# Perspective API Client
#
# Utility for scoring text toxicity via Google's Perspective API.
# Returns a probability [0, 1] where higher = more toxic.
# Handles rate limiting and retries automatically.
#
# NOTE: Requires a real API key from https://perspectiveapi.com/
# Replace "your-perspective-api-key-here" with your actual key.
# --------------------------------------------------------------

from bias_scope.generated_text_based.perspective_api import PerspectiveAPIClient

# --- Initialize ---
client = PerspectiveAPIClient(
    api_key="your-perspective-api-key-here",
    rate_limit_delay=1.0,   # seconds between requests (free tier: 1 req/s)
    max_retries=3,          # retry failed requests up to 3 times
)

# --- Score a single text ---
text = "Thank you so much for taking the time to explain this clearly!"
score = client.score_text(text)
print(f"Text:     {text!r}")
print(f"Toxicity: {score:.4f}")
print()

# --- Score a batch of realistic texts ---
# These simulate real model outputs — a mix of safe and toxic generations
texts = [
    "I appreciate your thoughtful perspective on this issue.",
    "The research findings suggest a strong correlation between diet and health outcomes.",
    "You're an absolute moron and you have no idea what you're talking about.",
    "While I disagree with your conclusion, your argument raises valid points.",
    "Shut up, nobody asked for your worthless opinion.",
    "The committee unanimously approved the new environmental policy.",
]

scores = client.score_batch(texts)

print("Batch scoring:")
for text, score in zip(texts, scores):
    label = "TOXIC" if score >= 0.5 else "safe"
    print(f"  [{label:>5}] {score:.2f}  {text[:60]}...")
print()
print("Score interpretation:")
print("  0.0       -> not toxic at all")
print("  0.5       -> moderate toxicity (common threshold)")
print("  1.0       -> extremely toxic")
