"""Tests for DemographicRepresentation metric."""

import pytest
import json
import numpy as np
from bias_scope.generated_text_based import DemographicRepresentation


class TestDemographicRepresentation:
    """Test suite for DemographicRepresentation."""
    
    def test_happy_path_mentions_normalization(self):
        """Test with mentions normalization."""
        dr = DemographicRepresentation()
        
        generations = [
            "the man walked",
            "the woman drove",
            "a man talked"
        ]
        
        group_lexicons = {
            'male': ['man'],
            'female': ['woman']
        }
        
        result = dr.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            normalize='mentions'
        )
        
        # Check structure
        assert result['metric'] == 'DemographicRepresentation'
        assert result['category'] == 'generated_text'
        
        # Check counts: man appears 2x, woman 1x
        assert result['counts']['male'] == 2
        assert result['counts']['female'] == 1
        assert result['total_mentions'] == 3
        
        # Check distribution: 2/3 and 1/3
        assert pytest.approx(result['distribution']['male'], abs=0.01) == 2/3
        assert pytest.approx(result['distribution']['female'], abs=0.01) == 1/3
    
    def test_tokens_normalization(self):
        """Test with tokens normalization."""
        dr = DemographicRepresentation()
        
        generations = [
            "man woman test test test"  # 5 tokens total
        ]
        
        group_lexicons = {
            'male': ['man'],
            'female': ['woman']
        }
        
        result = dr.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            normalize='tokens'
        )
        
        # man: 1/5, woman: 1/5
        assert pytest.approx(result['distribution']['male'], abs=0.01) == 1/5
        assert pytest.approx(result['distribution']['female'], abs=0.01) == 1/5
    
    def test_diversity_metrics(self):
        """Test diversity calculations."""
        dr = DemographicRepresentation()
        
        # Perfectly balanced: [0.5, 0.5]
        generations = [
            "man woman"
        ]
        
        group_lexicons = {
            'male': ['man'],
            'female': ['woman']
        }
        
        result = dr.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            normalize='mentions'
        )
        
        # For uniform distribution over 2 groups:
        # entropy = -2 * (0.5 * log(0.5)) = log(2)
        # normalized_entropy = log(2) / log(2) = 1.0
        # gini = 1 - 2*(0.5^2) = 0.5
        
        assert pytest.approx(result['diversity']['entropy'], abs=0.01) == np.log(2)
        assert pytest.approx(result['diversity']['normalized_entropy'], abs=0.01) == 1.0
        assert pytest.approx(result['diversity']['gini_impurity'], abs=0.01) == 0.5
    
    def test_compare_to_reference(self):
        """Test comparison to reference distribution."""
        dr = DemographicRepresentation()
        
        generations = [
            "man man woman"  # 2/3 male, 1/3 female
        ]
        
        group_lexicons = {
            'male': ['man'],
            'female': ['woman']
        }
        
        # Reference: uniform
        reference = {
            'male': 0.5,
            'female': 0.5
        }
        
        result = dr.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            normalize='mentions',
            compare_to=reference
        )
        
        # Should have reference comparison
        assert result['reference']['provided'] == True
        assert result['reference']['distribution'] == reference
        assert result['reference']['kl_pq'] is not None
        assert result['reference']['jsd'] is not None
        
        # KL and JSD should be >= 0
        assert result['reference']['kl_pq'] >= 0
        assert result['reference']['jsd'] >= 0
    
    def test_validation_total_mentions_zero(self):
        """Test validation: no mentions found."""
        dr = DemographicRepresentation()
        
        generations = ["no group terms here"]
        group_lexicons = {'male': ['man'], 'female': ['woman']}
        
        with pytest.raises(ValueError, match="No group mentions found"):
            dr.evaluate(
                generations=generations,
                group_lexicons=group_lexicons,
                normalize='mentions'
            )
    
    def test_validation_compare_to_missing_keys(self):
        """Test validation: compare_to missing keys."""
        dr = DemographicRepresentation()
        
        generations = ["man woman"]
        group_lexicons = {'male': ['man'], 'female': ['woman']}
        
        reference = {'male': 1.0}  # Missing 'female'
        
        with pytest.raises(ValueError, match="keys must match"):
            dr.evaluate(
                generations=generations,
                group_lexicons=group_lexicons,
                compare_to=reference
            )
    
    def test_validation_compare_to_extra_keys(self):
        """Test validation: compare_to has extra keys."""
        dr = DemographicRepresentation()
        
        generations = ["man woman"]
        group_lexicons = {'male': ['man'], 'female': ['woman']}
        
        reference = {
            'male': 0.4,
            'female': 0.4,
            'extra': 0.2
        }
        
        with pytest.raises(ValueError, match="keys must match"):
            dr.evaluate(
                generations=generations,
                group_lexicons=group_lexicons,
                compare_to=reference
            )
    
    def test_validation_compare_to_not_sum_to_one(self):
        """Test validation: compare_to doesn't sum to ~1."""
        dr = DemographicRepresentation()
        
        generations = ["man woman"]
        group_lexicons = {'male': ['man'], 'female': ['woman']}
        
        reference = {
            'male': 0.3,
            'female': 0.3  # Sums to 0.6
        }
        
        with pytest.raises(ValueError, match="must sum to ~1"):
            dr.evaluate(
                generations=generations,
                group_lexicons=group_lexicons,
                compare_to=reference
            )
    
    def test_validation_invalid_smoothing(self):
        """Test validation: smoothing <= 0."""
        dr = DemographicRepresentation()
        
        with pytest.raises(ValueError, match="smoothing must be > 0"):
            dr.evaluate(
                generations=["man"],
                group_lexicons={'male': ['man'], 'female': ['woman']},
                smoothing=-0.01
            )
    
    def test_validation_invalid_normalize_mode(self):
        """Test validation: invalid normalize mode."""
        dr = DemographicRepresentation()
        
        with pytest.raises(ValueError, match="normalize must be"):
            dr.evaluate(
                generations=["man"],
                group_lexicons={'male': ['man'], 'female': ['woman']},
                normalize='invalid'
            )
    
    def test_json_serializable(self):
        """Test that result is JSON-serializable."""
        dr = DemographicRepresentation()
        
        generations = ["man woman"]
        group_lexicons = {'male': ['man'], 'female': ['woman']}
        
        result = dr.evaluate(
            generations=generations,
            group_lexicons=group_lexicons
        )
        
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
    
    def test_return_structure_exact(self):
        """Test exact return dictionary structure."""
        dr = DemographicRepresentation()
        
        generations = ["man woman"]
        group_lexicons = {'male': ['man'], 'female': ['woman']}
        
        result = dr.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            normalize='mentions'
        )
        
        # Check all required keys
        assert 'metric' in result
        assert 'category' in result
        assert 'groups' in result
        assert 'counts' in result
        assert 'total_mentions' in result
        assert 'distribution' in result
        assert 'diversity' in result
        assert 'reference' in result
        
        # Check diversity sub-keys
        assert 'entropy' in result['diversity']
        assert 'normalized_entropy' in result['diversity']
        assert 'gini_impurity' in result['diversity']
        
        # Check reference sub-keys
        assert 'provided' in result['reference']
        assert 'distribution' in result['reference']
        assert 'kl_pq' in result['reference']
        assert 'jsd' in result['reference']


