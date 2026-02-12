"""Tests for SocialGroupSubstitution metric."""

import pytest
import json
from bias_scope.generated_text_based import SocialGroupSubstitution


class TestSocialGroupSubstitution:
    """Test suite for SocialGroupSubstitution."""
    
    def test_happy_path_single_placeholder(self):
        """Test basic functionality with single placeholder."""
        sgs = SocialGroupSubstitution()
        
        prompts = ["The {group} is nice."]
        substitutions = {'group': ['man', 'woman']}
        
        # Deterministic generate function
        def generate_fn(prompt):
            return f"OUT:{prompt}"
        
        # Score function: 1.0 for 'man', 3.0 for 'woman'
        def score_fn(text):
            if 'woman' in text.lower():
                return 3.0
            else:
                return 1.0
        
        result = sgs.evaluate(
            prompts=prompts,
            substitutions=substitutions,
            generate_fn=generate_fn,
            score_fn=score_fn
        )
        
        # Check structure
        assert result['metric'] == 'SocialGroupSubstitution'
        assert result['category'] == 'generated_text'
        assert result['num_prompts'] == 1
        assert result['num_samples'] == 1
        
        # Check scores
        assert result['scores']['group']['man'] == [1.0]
        assert result['scores']['group']['woman'] == [3.0]
        
        # Check individual unfairness: |1.0 - 3.0| = 2.0
        assert result['individual_unfairness']['group'] == [2.0]
        assert result['individual_unfairness_overall'] == 2.0
        
        # Check group disparity: max(1.0, 3.0) - min(1.0, 3.0) = 2.0
        assert result['group_disparity']['group'] == 2.0
        assert result['group_disparity']['_overall'] == 2.0
    
    def test_multiple_placeholders(self):
        """Test with multiple placeholders."""
        sgs = SocialGroupSubstitution()
        
        prompts = ["The {gender} is a {profession}."]
        substitutions = {
            'gender': ['man', 'woman'],
            'profession': ['doctor', 'nurse']
        }
        
        def generate_fn(prompt):
            return f"OUT:{prompt}"
        
        def score_fn(text):
            # Simple scoring based on keywords
            score = 0.5
            if 'man' in text:
                score += 0.1
            if 'doctor' in text:
                score += 0.2
            return score
        
        result = sgs.evaluate(
            prompts=prompts,
            substitutions=substitutions,
            generate_fn=generate_fn,
            score_fn=score_fn
        )
        
        # Should have scores for both placeholders
        assert 'gender' in result['scores']
        assert 'profession' in result['scores']
        assert len(result['placeholders']) == 2
    
    def test_batched_generate_fn(self):
        """Test with batch-capable generate function."""
        sgs = SocialGroupSubstitution()
        
        prompts = ["Hello {name}"]
        substitutions = {'name': ['Alice', 'Bob']}
        
        # Batch function
        def generate_fn(prompts_batch):
            if isinstance(prompts_batch, list):
                return [f"Response to: {p}" for p in prompts_batch]
            return f"Response to: {prompts_batch}"
        
        def score_fn(text):
            return 1.0
        
        result = sgs.evaluate(
            prompts=prompts,
            substitutions=substitutions,
            generate_fn=generate_fn,
            score_fn=score_fn
        )
        
        assert result['metadata']['supports_batched_generate_fn'] == False  # Heuristic may not detect
    
    def test_non_batched_generate_fn(self):
        """Test with single-input generate function."""
        sgs = SocialGroupSubstitution()
        
        prompts = ["Test {x}"]
        substitutions = {'x': ['a', 'b']}
        
        def generate_fn(prompt: str) -> str:
            return prompt.upper()
        
        def score_fn(text):
            return len(text)
        
        result = sgs.evaluate(
            prompts=prompts,
            substitutions=substitutions,
            generate_fn=generate_fn,
            score_fn=score_fn
        )
        
        assert isinstance(result['scores'], dict)
    
    def test_num_samples_aggregation_mean(self):
        """Test with num_samples > 1 and mean aggregation."""
        sgs = SocialGroupSubstitution()
        
        prompts = ["Say {word}"]
        substitutions = {'word': ['hello', 'goodbye']}
        
        call_count = [0]
        
        def generate_fn(prompt):
            call_count[0] += 1
            # Alternate outputs
            return f"out_{call_count[0]}"
        
        def score_fn(text):
            # Extract number from out_N
            num = int(text.split('_')[1])
            return float(num)
        
        result = sgs.evaluate(
            prompts=prompts,
            substitutions=substitutions,
            generate_fn=generate_fn,
            score_fn=score_fn,
            num_samples=3,
            aggregation='mean'
        )
        
        # Should aggregate 3 samples per (prompt, placeholder, value)
        # Total calls = 1 prompt * 1 placeholder * 2 values * 3 samples = 6
        assert call_count[0] == 6
    
    def test_aggregation_median(self):
        """Test median aggregation."""
        sgs = SocialGroupSubstitution()
        
        prompts = ["Test {x}"]
        substitutions = {'x': ['a', 'b']}
        
        outputs = [[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]]
        output_idx = [0]
        value_idx = [0]
        
        def generate_fn(prompt):
            return "text"
        
        def score_fn(text):
            val = outputs[value_idx[0]][output_idx[0] % 3]
            output_idx[0] += 1
            if output_idx[0] % 3 == 0:
                value_idx[0] = (value_idx[0] + 1) % 2
            return val
        
        result = sgs.evaluate(
            prompts=prompts,
            substitutions=substitutions,
            generate_fn=generate_fn,
            score_fn=score_fn,
            num_samples=3,
            aggregation='median'
        )
        
        assert result['aggregation'] == 'median'
    
    def test_validation_empty_prompts(self):
        """Test validation: empty prompts."""
        sgs = SocialGroupSubstitution()
        
        with pytest.raises(ValueError, match="prompts cannot be empty"):
            sgs.evaluate(
                prompts=[],
                substitutions={'x': ['a', 'b']},
                generate_fn=lambda x: x,
                score_fn=lambda x: 1.0
            )
    
    def test_validation_prompts_as_string(self):
        """Test validation: prompts provided as single string."""
        sgs = SocialGroupSubstitution()
        
        with pytest.raises(TypeError, match="must be a sequence"):
            sgs.evaluate(
                prompts="Single string",
                substitutions={'x': ['a', 'b']},
                generate_fn=lambda x: x,
                score_fn=lambda x: 1.0
            )
    
    def test_validation_empty_substitutions(self):
        """Test validation: empty substitutions."""
        sgs = SocialGroupSubstitution()
        
        with pytest.raises(ValueError, match="substitutions cannot be empty"):
            sgs.evaluate(
                prompts=["Test"],
                substitutions={},
                generate_fn=lambda x: x,
                score_fn=lambda x: 1.0
            )
    
    def test_validation_substitution_value_too_few(self):
        """Test validation: substitution with < 2 values."""
        sgs = SocialGroupSubstitution()
        
        with pytest.raises(ValueError, match="must have at least 2 values"):
            sgs.evaluate(
                prompts=["Test {x}"],
                substitutions={'x': ['only_one']},
                generate_fn=lambda x: x,
                score_fn=lambda x: 1.0
            )
    
    def test_validation_placeholder_not_in_prompt(self):
        """Test validation: placeholder not in any prompt."""
        sgs = SocialGroupSubstitution()
        
        with pytest.raises(ValueError, match="not found in any prompt"):
            sgs.evaluate(
                prompts=["Test {x}"],
                substitutions={'y': ['a', 'b']},  # 'y' not in prompts
                generate_fn=lambda x: x,
                score_fn=lambda x: 1.0
            )
    
    def test_validation_invalid_prompt_formatting(self):
        """Test validation: prompt cannot be formatted."""
        sgs = SocialGroupSubstitution()
        
        with pytest.raises(ValueError, match="cannot be formatted"):
            sgs.evaluate(
                prompts=["Test {x} {y}"],  # needs both x and y
                substitutions={'x': ['a', 'b']},  # only provides x
                generate_fn=lambda x: x,
                score_fn=lambda x: 1.0
            )
    
    def test_validation_generate_fn_not_callable(self):
        """Test validation: generate_fn not callable."""
        sgs = SocialGroupSubstitution()
        
        with pytest.raises(TypeError, match="must be callable"):
            sgs.evaluate(
                prompts=["Test {x}"],
                substitutions={'x': ['a', 'b']},
                generate_fn="not a function",
                score_fn=lambda x: 1.0
            )
    
    def test_validation_score_fn_not_callable(self):
        """Test validation: score_fn not callable."""
        sgs = SocialGroupSubstitution()
        
        with pytest.raises(TypeError, match="must be callable"):
            sgs.evaluate(
                prompts=["Test {x}"],
                substitutions={'x': ['a', 'b']},
                generate_fn=lambda x: x,
                score_fn=42
            )
    
    def test_validation_score_fn_returns_nan(self):
        """Test validation: score_fn returns NaN."""
        sgs = SocialGroupSubstitution()
        
        def score_fn(text):
            return float('nan')
        
        with pytest.raises(ValueError, match="NaN"):
            sgs.evaluate(
                prompts=["Test {x}"],
                substitutions={'x': ['a', 'b']},
                generate_fn=lambda x: x,
                score_fn=score_fn
            )
    
    def test_validation_score_fn_returns_inf(self):
        """Test validation: score_fn returns Inf."""
        sgs = SocialGroupSubstitution()
        
        def score_fn(text):
            return float('inf')
        
        with pytest.raises(ValueError, match="Inf"):
            sgs.evaluate(
                prompts=["Test {x}"],
                substitutions={'x': ['a', 'b']},
                generate_fn=lambda x: x,
                score_fn=score_fn
            )
    
    def test_determinism(self):
        """Test that same inputs yield identical outputs."""
        sgs = SocialGroupSubstitution()
        
        prompts = ["The {animal} runs."]
        substitutions = {'animal': ['cat', 'dog']}
        
        def generate_fn(prompt):
            return f"Generated: {prompt}"
        
        def score_fn(text):
            return len(text) / 10.0
        
        result1 = sgs.evaluate(
            prompts=prompts,
            substitutions=substitutions,
            generate_fn=generate_fn,
            score_fn=score_fn
        )
        
        result2 = sgs.evaluate(
            prompts=prompts,
            substitutions=substitutions,
            generate_fn=generate_fn,
            score_fn=score_fn
        )
        
        assert result1['individual_unfairness_overall'] == result2['individual_unfairness_overall']
        assert result1['scores'] == result2['scores']
    
    def test_json_serializable(self):
        """Test that result is JSON-serializable."""
        sgs = SocialGroupSubstitution()
        
        prompts = ["Test {x}"]
        substitutions = {'x': ['a', 'b']}
        
        result = sgs.evaluate(
            prompts=prompts,
            substitutions=substitutions,
            generate_fn=lambda x: x,
            score_fn=lambda x: 1.0
        )
        
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
    
    # === A) Missing Edge Cases for SocialGroupSubstitution ===
    
    def test_multiple_placeholders_independence(self):
        """Test that multiple placeholders produce independent unfairness values."""
        sgs = SocialGroupSubstitution()
        
        prompts = ["{group} is a {role}"]
        substitutions = {
            'group': ['man', 'woman'],
            'role': ['doctor', 'nurse']
        }
        
        # Score depends on both group and role
        def score_fn(text):
            score = 0.0
            if 'man' in text:
                score += 2.0
            if 'woman' in text:
                score += 1.0
            if 'doctor' in text:
                score += 0.5
            if 'nurse' in text:
                score += 0.1
            return score
        
        result = sgs.evaluate(
            prompts=prompts,
            substitutions=substitutions,
            generate_fn=lambda x: x,
            score_fn=score_fn
        )
        
        # Check that both placeholders have unfairness computed
        assert 'group' in result['individual_unfairness']
        assert 'role' in result['individual_unfairness']
        
        # Unfairness values should differ (different substitutions create different score patterns)
        if_group = result['individual_unfairness']['group'][0]
        if_role = result['individual_unfairness']['role'][0]
        
        # They should be different because the scoring is asymmetric
        # group affects score more (2.0 vs 1.0) than role (0.5 vs 0.1)
        assert if_group != if_role
        assert if_group > if_role
    
    def test_tie_case_deterministic(self):
        """Test that all identical scores produce zero unfairness and disparity."""
        sgs = SocialGroupSubstitution()
        
        prompts = ["Test {x}"]
        substitutions = {'x': ['a', 'b', 'c']}
        
        # Always return same score regardless of input
        def constant_score_fn(text):
            return 5.0
        
        result = sgs.evaluate(
            prompts=prompts,
            substitutions=substitutions,
            generate_fn=lambda x: x,
            score_fn=constant_score_fn
        )
        
        # All scores identical -> no unfairness
        assert result['individual_unfairness_overall'] == 0.0
        assert result['individual_unfairness']['x'][0] == 0.0
        
        # All group means identical -> no disparity
        assert result['group_disparity']['x'] == 0.0
        assert result['group_disparity']['_overall'] == 0.0
    
    def test_batched_vs_nonbatched_identical_output(self):
        """Test that batched and non-batched generate functions produce identical results."""
        sgs = SocialGroupSubstitution()
        
        prompts = ["Say {word}"]
        substitutions = {'word': ['hello', 'goodbye']}
        
        def score_fn(text):
            return len(text)
        
        # Non-batched version
        def generate_single(prompt):
            return f"Generated: {prompt}"
        
        result_single = sgs.evaluate(
            prompts=prompts,
            substitutions=substitutions,
            generate_fn=generate_single,
            score_fn=score_fn
        )
        
        # Batched version (returns list)
        def generate_batch(prompts_input):
            if isinstance(prompts_input, list):
                return [f"Generated: {p}" for p in prompts_input]
            return f"Generated: {prompts_input}"
        
        result_batch = sgs.evaluate(
            prompts=prompts,
            substitutions=substitutions,
            generate_fn=generate_batch,
            score_fn=score_fn
        )
        
        # Outputs should be identical
        assert result_single['individual_unfairness_overall'] == result_batch['individual_unfairness_overall']
        assert result_single['group_disparity']['_overall'] == result_batch['group_disparity']['_overall']
        
        # Scores should be same (both generate same text)
        for word in ['hello', 'goodbye']:
            assert result_single['scores']['word'][word] == result_batch['scores']['word'][word]


