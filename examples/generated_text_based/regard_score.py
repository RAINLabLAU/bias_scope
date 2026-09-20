"""RegardScore example (requires the optional torch dependencies)."""

from bias_scope.generated_text_based import RegardScore

# The default is a regard checkpoint, not a sentiment classifier. It is a
# BiasScope adaptation and is not BOLD's demonstrated historical classifier.
regard = RegardScore()
scores = regard.evaluate(
    [["He was respected by colleagues."]],
    [["She was respected by colleagues."]],
)
print(scores)
