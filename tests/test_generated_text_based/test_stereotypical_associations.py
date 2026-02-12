"""Tests for StereotypicalAssociations metric."""

import pytest
import json
from bias_scope.generated_text_based import StereotypicalAssociations


class TestStereotypicalAssociations:
    """Test suite for StereotypicalAssociations."""
    
    def test_token_window_happy_path(self):
        """Test token_window matcher with basic hit."""
        sa = StereotypicalAssociations()
        
        generations = [
            "Women are bad at math",
            "The doctor is skilled"
        ]
        
        rules = [
            {
                'name': 'women_math_negative',
                'group_terms': ['woman', 'women'],
                'attribute_terms': ['bad', 'poor']
            }
        ]
        
        result = sa.evaluate(
            generations=generations,
            stereotype_rules=rules,
            context_window=5
        )
        
        # Check structure
        assert result['metric'] == 'StereotypicalAssociations'
        assert result['category'] == 'generated_text'
        assert result['matcher'] == 'token_window'
        
        # First generation should hit
        assert result['rules'][0]['name'] == 'women_math_negative'
        assert result['rules'][0]['hits'] >= 1
        
        # Per-generation
        assert result['per_generation'][0]['any_hit'] == True
        assert 'women_math_negative' in result['per_generation'][0]['hits']
    
    def test_token_window_non_hit(self):
        """Test when group and attribute are too far apart."""
        sa = StereotypicalAssociations()
        
        # 'women' and 'bad' are far apart
        generations = [
            "women " + " ".join(["the"] * 20) + " bad"
        ]
        
        rules = [
            {
                'name': 'test_rule',
                'group_terms': ['women'],
                'attribute_terms': ['bad']
            }
        ]
        
        result = sa.evaluate(
            generations=generations,
            stereotype_rules=rules,
            context_window=3  # Too small
        )
        
        # Should not hit
        assert result['rules'][0]['hits'] == 0
        assert result['per_generation'][0]['any_hit'] == False
    
    def test_multiple_rules(self):
        """Test with multiple rules, some hitting."""
        sa = StereotypicalAssociations()
        
        generations = [
            "men are strong leaders"
        ]
        
        rules = [
            {
                'name': 'men_strong',
                'group_terms': ['men', 'man'],
                'attribute_terms': ['strong', 'powerful']
            },
            {
                'name': 'men_leader',
                'group_terms': ['men', 'man'],
                'attribute_terms': ['leader', 'leaders']
            }
        ]
        
        result = sa.evaluate(
            generations=generations,
            stereotype_rules=rules,
            context_window=5
        )
        
        # Both rules should hit
        assert result['rules'][0]['hits'] == 1
        assert result['rules'][1]['hits'] == 1
        
        # Generation should list both hits
        assert len(result['per_generation'][0]['hits']) == 2
    
    def test_regex_matcher_happy_path(self):
        """Test regex matcher."""
        sa = StereotypicalAssociations()
        
        generations = [
            "Women are naturally nurturing",
            "Men are logical"
        ]
        
        rules = [
            {
                'name': 'women_naturally',
                'pattern': r'women\s+are\s+naturally'
            }
        ]
        
        result = sa.evaluate(
            generations=generations,
            stereotype_rules=rules,
            matcher='regex'
        )
        
        assert result['matcher'] == 'regex'
        assert result['rules'][0]['hits'] == 1
        assert result['per_generation'][0]['any_hit'] == True
    
    def test_case_insensitive_true(self):
        """Test case-insensitive matching."""
        sa = StereotypicalAssociations()
        
        generations = [
            "WOMEN are BAD at math"
        ]
        
        rules = [
            {
                'name': 'test',
                'group_terms': ['women'],
                'attribute_terms': ['bad']
            }
        ]
        
        result = sa.evaluate(
            generations=generations,
            stereotype_rules=rules,
            context_window=5,
            case_insensitive=True
        )
        
        # Should match despite case difference
        assert result['rules'][0]['hits'] >= 1
    
    def test_validation_empty_rules(self):
        """Test validation: empty rules."""
        sa = StereotypicalAssociations()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            sa.evaluate(
                generations=["test"],
                stereotype_rules=[]
            )
    
    def test_validation_rule_missing_name(self):
        """Test validation: rule missing 'name'."""
        sa = StereotypicalAssociations()
        
        rules = [
            {
                'group_terms': ['women'],
                'attribute_terms': ['bad']
            }
        ]
        
        with pytest.raises(ValueError, match="missing required key 'name'"):
            sa.evaluate(
                generations=["test"],
                stereotype_rules=rules
            )
    
    def test_validation_token_window_missing_group_terms(self):
        """Test validation: token_window missing group_terms."""
        sa = StereotypicalAssociations()
        
        rules = [
            {
                'name': 'test',
                'attribute_terms': ['bad']
            }
        ]
        
        with pytest.raises(ValueError, match="missing 'group_terms'"):
            sa.evaluate(
                generations=["test"],
                stereotype_rules=rules,
                matcher='token_window'
            )
    
    def test_validation_token_window_missing_attribute_terms(self):
        """Test validation: token_window missing attribute_terms."""
        sa = StereotypicalAssociations()
        
        rules = [
            {
                'name': 'test',
                'group_terms': ['women']
            }
        ]
        
        with pytest.raises(ValueError, match="missing 'attribute_terms'"):
            sa.evaluate(
                generations=["test"],
                stereotype_rules=rules,
                matcher='token_window'
            )
    
    def test_validation_regex_missing_pattern(self):
        """Test validation: regex missing pattern."""
        sa = StereotypicalAssociations()
        
        rules = [
            {
                'name': 'test'
            }
        ]
        
        with pytest.raises(ValueError, match="missing 'pattern'"):
            sa.evaluate(
                generations=["test"],
                stereotype_rules=rules,
                matcher='regex'
            )
    
    def test_validation_invalid_matcher(self):
        """Test validation: invalid matcher."""
        sa = StereotypicalAssociations()
        
        rules = [
            {
                'name': 'test',
                'group_terms': ['women'],
                'attribute_terms': ['bad']
            }
        ]
        
        with pytest.raises(ValueError, match="matcher must be"):
            sa.evaluate(
                generations=["test"],
                stereotype_rules=rules,
                matcher='invalid'
            )
    
    def test_validation_context_window_too_small(self):
        """Test validation: context_window < 1."""
        sa = StereotypicalAssociations()
        
        rules = [
            {
                'name': 'test',
                'group_terms': ['women'],
                'attribute_terms': ['bad']
            }
        ]
        
        with pytest.raises(ValueError, match="context_window must be >= 1"):
            sa.evaluate(
                generations=["test"],
                stereotype_rules=rules,
                context_window=0
            )
    
    def test_validation_invalid_regex_pattern(self):
        """Test validation: invalid regex."""
        sa = StereotypicalAssociations()
        
        rules = [
            {
                'name': 'bad_regex',
                'pattern': '[invalid(regex'
            }
        ]
        
        with pytest.raises(ValueError, match="invalid regex pattern"):
            sa.evaluate(
                generations=["test"],
                stereotype_rules=rules,
                matcher='regex'
            )
    
    def test_rate_per_1k_calculation(self):
        """Test rate per 1k calculation."""
        sa = StereotypicalAssociations()
        
        # Create 100 generations
        generations = ["women are bad"] * 50 + ["neutral text"] * 50
        
        rules = [
            {
                'name': 'test',
                'group_terms': ['women'],
                'attribute_terms': ['bad']
            }
        ]
        
        result = sa.evaluate(
            generations=generations,
            stereotype_rules=rules,
            context_window=5
        )
        
        # 50 hits out of 100 = 500 per 1k
        assert pytest.approx(result['rules'][0]['rate_per_1k'], abs=1) == 500
    
    def test_json_serializable(self):
        """Test that result is JSON-serializable."""
        sa = StereotypicalAssociations()
        
        generations = ["women are bad at math"]
        rules = [
            {
                'name': 'test',
                'group_terms': ['women'],
                'attribute_terms': ['bad']
            }
        ]
        
        result = sa.evaluate(
            generations=generations,
            stereotype_rules=rules
        )
        
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)


