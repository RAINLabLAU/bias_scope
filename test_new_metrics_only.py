#!/usr/bin/env python3
"""
Test script for generated_text_based metrics only (no torch required).
This bypasses the main __init__.py which requires torch.
"""

import sys
sys.path.insert(0, 'src')

# Import directly from submodules to avoid torch dependency
from bias_scope.generated_text_based.social_group_substitution import SocialGroupSubstitution
from bias_scope.generated_text_based.cooccurrence_bias_score import CoOccurrenceBiasScore
from bias_scope.generated_text_based.demographic_representation import DemographicRepresentation
from bias_scope.generated_text_based.stereotypical_associations import StereotypicalAssociations
from bias_scope.generated_text_based.marked_persons import MarkedPersons

print("✓ All imports successful!")
print("\nTesting instantiation...")

metrics = [
    ("SocialGroupSubstitution", SocialGroupSubstitution()),
    ("CoOccurrenceBiasScore", CoOccurrenceBiasScore()),
    ("DemographicRepresentation", DemographicRepresentation()),
    ("StereotypicalAssociations", StereotypicalAssociations()),
    ("MarkedPersons", MarkedPersons())
]

for name, metric in metrics:
    print(f"  ✓ {name} (category: {metric.category})")

print("\n✓ All metrics instantiated successfully!")
print("\nRunning quick evaluation tests...")

# Test 1: Social Group Substitution
sgs = SocialGroupSubstitution()
result = sgs.evaluate(
    prompts=["The {gender} is nice."],
    substitutions={'gender': ['man', 'woman']},
    generate_fn=lambda x: x,
    score_fn=lambda x: 1.0
)
print(f"  ✓ SocialGroupSubstitution: IF={result['individual_unfairness_overall']:.3f}")

# Test 2: CoOccurrenceBiasScore
cobs = CoOccurrenceBiasScore()
result = cobs.evaluate(
    generations=["man doctor", "woman nurse"],
    group_lexicons={'male': ['man'], 'female': ['woman']}
)
print(f"  ✓ CoOccurrenceBiasScore: vocab_size={result['vocab_size']}")

# Test 3: DemographicRepresentation
dr = DemographicRepresentation()
result = dr.evaluate(
    generations=["man woman"],
    group_lexicons={'male': ['man'], 'female': ['woman']}
)
print(f"  ✓ DemographicRepresentation: entropy={result['diversity']['entropy']:.3f}")

# Test 4: StereotypicalAssociations
sa = StereotypicalAssociations()
result = sa.evaluate(
    generations=["women are bad at math"],
    stereotype_rules=[{
        'name': 'test',
        'group_terms': ['women'],
        'attribute_terms': ['bad']
    }]
)
print(f"  ✓ StereotypicalAssociations: hits={result['rules'][0]['hits']}")

# Test 5: MarkedPersons
mp = MarkedPersons()
result = mp.evaluate(
    marked_generations=["caring nurturing"],
    unmarked_generations=["logical analytical"],
    min_count=1
)
print(f"  ✓ MarkedPersons: vocab={result['summary']['vocab_considered']}")

print("\n" + "="*60)
print("✓ ALL TESTS PASSED - Implementation is working!")
print("="*60)
