"""
Test Suite for LLM DJ System
============================

This script tests the core functionality of the LLM DJ system without requiring
actual audio files or API keys. It includes unit tests and mock tests to verify
the system's components work correctly.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

# Import system components
from llm_dj_system import (
    AdvancedTrackFeatures, 
    TransitionEffect, 
    AdvancedTransitionPlan,
    AdvancedAudioAnalyzer,
    TransitionEffects,
    LLMTransitionReasoner,
    TransitionEvaluator,
    LLMDJSystem
)

class TestAdvancedTrackFeatures(unittest.TestCase):
    """Test the AdvancedTrackFeatures data class"""
    
    def setUp(self):
        """Set up test data"""
        self.features = AdvancedTrackFeatures(
            filename="happy.wav",
            duration=180.0,
            bpm=128.0,
            key="Am",
            energy=0.7,
            danceability=0.8,
            valence=0.6,
            speechiness=0.2,
            beats=np.array([0, 0.5, 1.0, 1.5]),
            onset_frames=np.array([0, 100, 200]),
            segments=[],
            chroma=np.random.rand(12, 100),
            mfcc=np.random.rand(13, 100),
            spectral_centroid=2000.0,
            zero_crossing_rate=0.05,
            lyrics=[],
            lyric_density=0.5,
            intro_duration=8.0,
            outro_duration=12.0,
            breakdown_points=[60.0, 120.0],
            buildup_points=[45.0, 105.0],
            drop_points=[50.0, 110.0]
        )
    
    def test_to_llm_description(self):
        """Test LLM description generation"""
        description = self.features.to_llm_description()
        
        self.assertIn("happy.wav", description)
        self.assertIn("128.0", description)
        self.assertIn("Am", description)
        self.assertIn("High", description)  # Energy level
        self.assertIn("8.0s", description)  # Intro duration

class TestTransitionEffect(unittest.TestCase):
    """Test the TransitionEffect data class"""
    
    def test_effect_creation(self):
        """Test creating transition effects"""
        effect = TransitionEffect(
            name="crossfade_linear",
            parameters={"duration": 4000},
            duration_ms=4000,
            description="Linear crossfade"
        )
        
        self.assertEqual(effect.name, "crossfade_linear")
        self.assertEqual(effect.duration_ms, 4000)
        self.assertIn("duration", effect.parameters)

class TestTransitionEffects(unittest.TestCase):
    """Test the TransitionEffects library"""
    
    def setUp(self):
        """Set up mock audio segments"""
        self.mock_track1 = Mock()
        self.mock_track2 = Mock()
        self.mock_track1.__len__ = Mock(return_value=120000)  # 2 minutes
        self.mock_track2.__len__ = Mock(return_value=180000)  # 3 minutes
    
    @patch('llm_dj_system.AudioSegment')
    def test_crossfade_linear(self, mock_audio_segment):
        """Test linear crossfade"""
        self.mock_track1.append = Mock(return_value="mixed_audio")
        
        result = TransitionEffects.crossfade(
            self.mock_track1, self.mock_track2, 4000, "linear"
        )
        
        self.mock_track1.append.assert_called_once_with(
            self.mock_track2, crossfade=4000
        )
    
    def test_beatmatch_cut(self):
        """Test beat-matched cut transition"""
        self.mock_track1.__getitem__ = Mock(return_value="track1_segment")
        self.mock_track2.__getitem__ = Mock(return_value="track2_segment")
        mock_result = Mock()
        mock_result.__add__ = Mock(return_value="final_result")
        self.mock_track1.__getitem__.return_value = mock_result
        
        result = TransitionEffects.beatmatch_cut(
            self.mock_track1, self.mock_track2, 30.0, 5.0
        )
        
        # Verify the cut points were used
        self.mock_track1.__getitem__.assert_called_with(slice(None, 30000, None))

class TestLLMTransitionReasoner(unittest.TestCase):
    """Test the LLM reasoning component"""
    
    def setUp(self):
        """Set up test data"""
        self.track1_features = AdvancedTrackFeatures(
            filename="happy.wav", duration=180.0, bpm=128.0, key="Am",
            energy=0.7, danceability=0.8, valence=0.6, speechiness=0.2,
            beats=np.array([]), onset_frames=np.array([]), segments=[],
            chroma=np.random.rand(12, 100), mfcc=np.random.rand(13, 100),
            spectral_centroid=2000.0, zero_crossing_rate=0.05, lyrics=[],
            lyric_density=0.5, intro_duration=8.0, outro_duration=12.0,
            breakdown_points=[], buildup_points=[], drop_points=[]
        )
        
        self.track2_features = AdvancedTrackFeatures(
            filename="girlslikeyou.wav", duration=200.0, bpm=132.0, key="C",
            energy=0.8, danceability=0.9, valence=0.7, speechiness=0.1,
            beats=np.array([]), onset_frames=np.array([]), segments=[],
            chroma=np.random.rand(12, 100), mfcc=np.random.rand(13, 100),
            spectral_centroid=2200.0, zero_crossing_rate=0.04, lyrics=[],
            lyric_density=0.3, intro_duration=6.0, outro_duration=10.0,
            breakdown_points=[], buildup_points=[], drop_points=[]
        )
    
    def test_prepare_transition_context(self):
        """Test context preparation for LLM"""
        reasoner = LLMTransitionReasoner()
        
        context = reasoner._prepare_transition_context(
            self.track1_features, self.track2_features, "Make it smooth"
        )
        
        self.assertIn("happy.wav", context)
        self.assertIn("girlslikeyou.wav", context)
        self.assertIn("BPM Difference: 4.0", context)
        self.assertIn("Make it smooth", context)
    
    def test_fallback_parse(self):
        """Test fallback parsing when JSON extraction fails"""
        reasoner = LLMTransitionReasoner()
        
        # Test with non-JSON output
        llm_output = "This is a crossfade_s_curve transition that should work well."
        
        result = reasoner._fallback_parse(llm_output)
        
        self.assertIn("analysis", result)
        self.assertIn("primary_effect", result)
        self.assertEqual(result["primary_effect"], "crossfade_s_curve")

class TestTransitionEvaluator(unittest.TestCase):
    """Test the transition evaluation system"""
    
    def setUp(self):
        """Set up test data"""
        self.track1_features = AdvancedTrackFeatures(
            filename="happy.wav", duration=180.0, bpm=128.0, key="Am",
            energy=0.7, danceability=0.8, valence=0.6, speechiness=0.2,
            beats=np.array([]), onset_frames=np.array([]), segments=[],
            chroma=np.random.rand(12, 100), mfcc=np.random.rand(13, 100),
            spectral_centroid=2000.0, zero_crossing_rate=0.05, lyrics=[],
            lyric_density=0.5, intro_duration=8.0, outro_duration=12.0,
            breakdown_points=[], buildup_points=[], drop_points=[]
        )
        
        self.track2_features = AdvancedTrackFeatures(
            filename="girlslikeyou.wav", duration=200.0, bpm=128.0, key="Am",
            energy=0.75, danceability=0.85, valence=0.65, speechiness=0.15,
            beats=np.array([]), onset_frames=np.array([]), segments=[],
            chroma=np.random.rand(12, 100), mfcc=np.random.rand(13, 100),
            spectral_centroid=2100.0, zero_crossing_rate=0.05, lyrics=[],
            lyric_density=0.4, intro_duration=6.0, outro_duration=10.0,
            breakdown_points=[], buildup_points=[], drop_points=[]
        )
        
        self.transition_plan = AdvancedTransitionPlan(
            track1_features=self.track1_features,
            track2_features=self.track2_features,
            llm_analysis="Test analysis",
            transition_reasoning="Test reasoning",
            primary_effect=TransitionEffect("crossfade_linear", {}, 4000, "Linear crossfade"),
            secondary_effects=[],
            transition_start_time=170.0,
            transition_duration=4.0,
            track1_fadeout_time=172.0,
            track2_fadein_time=0.0,
            predicted_quality=0.8,
            confidence=0.9
        )
    
    def test_evaluate_harmonic_compatibility(self):
        """Test harmonic compatibility evaluation"""
        evaluator = TransitionEvaluator()
        
        # Test same key (perfect match)
        score = evaluator._evaluate_harmonic_compatibility(self.transition_plan)
        self.assertEqual(score, 1.0)
        
        # Test different keys
        self.track2_features.key = "C"
        score = evaluator._evaluate_harmonic_compatibility(self.transition_plan)
        self.assertGreater(score, 0.4)  # Should be reasonable compatibility
    
    def test_evaluate_structural_appropriateness(self):
        """Test structural appropriateness evaluation"""
        evaluator = TransitionEvaluator()
        
        score = evaluator._evaluate_structural_appropriateness(self.transition_plan)
        self.assertGreaterEqual(score, 0.7)  # Base score
        self.assertLessEqual(score, 1.0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    @patch('llm_dj_system.client')
    @patch('llm_dj_system.librosa')
    @patch('llm_dj_system.AudioSegment')
    def test_system_integration(self, mock_audio_segment, mock_librosa, mock_client):
        """Test the complete system integration"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
        {
            "analysis": "Test analysis",
            "transition_reasoning": "Test reasoning",
            "primary_effect": "crossfade_linear",
            "secondary_effects": [],
            "timing": {
                "transition_start_time": 170,
                "transition_duration": 4.0,
                "track1_fadeout_time": 172,
                "track2_fadein_time": 0
            },
            "predicted_quality": 0.8,
            "confidence": 0.9
        }
        """
        mock_client.chat.completions.create.return_value = mock_response
        
        # Mock audio loading
        mock_librosa.load.return_value = (np.random.rand(44100 * 60), 22050)  # 1 minute
        mock_librosa.beat.beat_track.return_value = (128.0, np.array([0, 0.5, 1.0]))
        mock_librosa.feature.chroma_cqt.return_value = np.random.rand(12, 100)
        mock_librosa.feature.mfcc.return_value = np.random.rand(13, 100)
        mock_librosa.feature.spectral_centroid.return_value = np.array([[2000.0]])
        mock_librosa.feature.zero_crossing_rate.return_value = np.array([[0.05]])
        mock_librosa.feature.rms.return_value = np.array([[0.5]])
        mock_librosa.onset.onset_detect.return_value = np.array([0, 100, 200])
        mock_librosa.onset.onset_strength.return_value = np.random.rand(1000)
        mock_librosa.frames_to_time.return_value = np.array([0, 1, 2])
        mock_librosa.segment.agglomerative.return_value = np.array([0, 50, 100])
        
        # Mock audio segment operations
        mock_segment = Mock()
        mock_segment.append.return_value = mock_segment
        mock_segment.export = Mock()
        mock_segment.duration_seconds = 120.0
        mock_audio_segment.from_file.return_value = mock_segment
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            track1_path = os.path.join(temp_dir, "happy.wav")
            track2_path = os.path.join(temp_dir, "girlslikeyou.wav")
            output_path = os.path.join(temp_dir, "output.wav")
            
            # Create dummy files
            Path(track1_path).touch()
            Path(track2_path).touch()
            
            # Test system
            system = LLMDJSystem()
            
            # This should not raise any exceptions
            plan = system.create_transition(
                track1_path, track2_path, output_path, "Test prompt"
            )
            
            self.assertIsInstance(plan, AdvancedTransitionPlan)
            self.assertEqual(plan.primary_effect.name, "crossfade_linear")

def run_basic_tests():
    """Run basic tests without external dependencies"""
    print("🧪 Running LLM DJ System Tests")
    print("=" * 50)
    
    # Test data structures
    print("Testing data structures...")
    features = AdvancedTrackFeatures(
        filename="happy.wav", duration=180.0, bpm=128.0, key="Am",
        energy=0.7, danceability=0.8, valence=0.6, speechiness=0.2,
        beats=np.array([]), onset_frames=np.array([]), segments=[],
        chroma=np.random.rand(12, 100), mfcc=np.random.rand(13, 100),
        spectral_centroid=2000.0, zero_crossing_rate=0.05, lyrics=[],
        lyric_density=0.5, intro_duration=8.0, outro_duration=12.0,
        breakdown_points=[], buildup_points=[], drop_points=[]
    )
    
    description = features.to_llm_description()
    assert "happy.wav" in description
    print("✅ AdvancedTrackFeatures working correctly")
    
    # Test transition effects
    print("Testing transition effects...")
    effect = TransitionEffect("crossfade_linear", {}, 4000, "Test effect")
    assert effect.name == "crossfade_linear"
    print("✅ TransitionEffect working correctly")
    
    # Test LLM reasoner components
    print("Testing LLM reasoner...")
    reasoner = LLMTransitionReasoner()
    context = reasoner._prepare_transition_context(features, features, "test prompt")
    assert "happy.wav" in context
    print("✅ LLMTransitionReasoner context preparation working")
    
    # Test evaluator
    print("Testing evaluator...")
    evaluator = TransitionEvaluator()
    plan = AdvancedTransitionPlan(
        track1_features=features, track2_features=features,
        llm_analysis="test", transition_reasoning="test",
        primary_effect=effect, secondary_effects=[],
        transition_start_time=170.0, transition_duration=4.0,
        track1_fadeout_time=172.0, track2_fadein_time=0.0,
        predicted_quality=0.8, confidence=0.9
    )
    
    harmony_score = evaluator._evaluate_harmonic_compatibility(plan)
    assert 0.0 <= harmony_score <= 1.0
    print("✅ TransitionEvaluator working correctly")
    
    print("\n🎉 All basic tests passed!")
    print("💡 To run full tests with audio processing, use: python -m unittest test_dj_system")

if __name__ == "__main__":
    # Check if running basic tests or full test suite
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--basic":
        run_basic_tests()
    else:
        # Run full test suite
        unittest.main(verbosity=2) 