#!/usr/bin/env python3
"""
Verification script for generated text-based metrics.

Run this script to verify that all new metrics are correctly implemented
and can be imported and instantiated.
"""

import sys
sys.path.insert(0, 'src')

def test_imports():
    """Test that all metrics can be imported."""
    print("Testing imports...")
    
    try:
        from bias_scope.generated_text_based import (
            SocialGroupSubstitution,
            CoOccurrenceBiasScore,
            DemographicRepresentation,
            StereotypicalAssociations,
            MarkedPersons
        )
        print("✓ All metrics imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_instantiation():
    """Test that all metrics can be instantiated."""
    print("\nTesting instantiation...")
    
    try:
        from bias_scope.generated_text_based import (
            SocialGroupSubstitution,
            CoOccurrenceBiasScore,
            DemographicRepresentation,
            StereotypicalAssociations,
            MarkedPersons
        )
        
        metrics = [
            SocialGroupSubstitution(),
            CoOccurrenceBiasScore(),
            DemographicRepresentation(),
            StereotypicalAssociations(),
            MarkedPersons()
        ]
        
        for metric in metrics:
            print(f"  ✓ {metric.name} (category: {metric.category})")
        
        print("✓ All metrics instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Instantiation failed: {e}")
        return False

def test_simple_evaluation():
    """Test a simple evaluation on each metric."""
    print("\nTesting simple evaluations...")
    
    try:
        from bias_scope.generated_text_based import (
            SocialGroupSubstitution,
            CoOccurrenceBiasScore,
            DemographicRepresentation,
            StereotypicalAssociations,
            MarkedPersons
        )
        
        # Test SocialGroupSubstitution
        sgs = SocialGroupSubstitution()
        result = sgs.evaluate(
            prompts=["The {gender} is nice."],
            substitutions={'gender': ['man', 'woman']},
            generate_fn=lambda x: x,
            score_fn=lambda x: 1.0
        )
        print(f"  ✓ SocialGroupSubstitution: IF={result['individual_unfairness_overall']:.3f}")
        
        # Test CoOccurrenceBiasScore
        cobs = CoOccurrenceBiasScore()
        result = cobs.evaluate(
            generations=["man doctor", "woman nurse"],
            group_lexicons={'male': ['man'], 'female': ['woman']}
        )
        print(f"  ✓ CoOccurrenceBiasScore: vocab_size={result['vocab_size']}")
        
        # Test DemographicRepresentation
        dr = DemographicRepresentation()
        result = dr.evaluate(
            generations=["man woman"],
            group_lexicons={'male': ['man'], 'female': ['woman']}
        )
        print(f"  ✓ DemographicRepresentation: entropy={result['diversity']['entropy']:.3f}")
        
        # Test StereotypicalAssociations
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
        
        # Test MarkedPersons
        mp = MarkedPersons()
        result = mp.evaluate(
            marked_generations=["caring nurturing"],
            unmarked_generations=["logical analytical"],
            min_count=1
        )
        print(f"  ✓ MarkedPersons: vocab={result['summary']['vocab_considered']}")
        
        print("✓ All simple evaluations passed")
        return True
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("="*60)
    print("Generated Text-Based Metrics Verification")
    print("="*60)
    
    results = []
    results.append(test_imports())
    results.append(test_instantiation())
    results.append(test_simple_evaluation())
    
    print("\n" + "="*60)
    if all(results):
        print("✓ ALL CHECKS PASSED")
        print("="*60)
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())


