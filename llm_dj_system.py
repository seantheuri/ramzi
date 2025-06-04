"""
LLM-Powered DJ System
=====================

A sophisticated DJ system that uses Large Language Models to make intelligent
transition decisions based on comprehensive audio analysis and musical context.

Features:
- Advanced audio feature extraction (BPM, key, energy, lyrics, structure)
- LLM-powered transition reasoning with musical context
- Professional DJ transition effects (loops, filters, cuts, etc.)
- Transition quality evaluation and feedback
- Interactive user prompting for custom transitions

Author: AI Assistant
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
from scipy import signal
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import librosa
import soundfile as sf
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
import click

# Import existing analyzer
from analyzer import _analyze_single, TransitionPlan

# Setup logging and console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class AdvancedTrackFeatures:
    """Comprehensive track features for LLM reasoning"""
    # Basic info
    filename: str
    duration: float
    
    # Musical features
    bpm: float
    key: Optional[str]
    energy: float
    danceability: float
    valence: float
    speechiness: float
    
    # Advanced features
    beats: np.ndarray
    onset_frames: np.ndarray
    segments: List[Dict]  # Structural segments
    chroma: np.ndarray
    mfcc: np.ndarray
    spectral_centroid: float
    zero_crossing_rate: float
    
    # Lyrical analysis
    lyrics: List[Dict]  # Word-level timestamps
    lyric_density: float
    
    # DJ-specific features
    intro_duration: float
    outro_duration: float
    breakdown_points: List[float]
    buildup_points: List[float]
    drop_points: List[float]
    
    def to_llm_description(self) -> str:
        """Convert features to natural language for LLM reasoning"""
        key_str = self.key if self.key else "Unknown"
        energy_desc = "High" if self.energy > 0.7 else "Medium" if self.energy > 0.4 else "Low"
        speech_desc = "Vocal-heavy" if self.speechiness > 0.5 else "Some vocals" if self.speechiness > 0.2 else "Instrumental"
        
        description = f"""
Track: {self.filename}
Duration: {self.duration:.1f}s
Musical Properties:
- BPM: {self.bpm:.1f}
- Key: {key_str}
- Energy Level: {energy_desc} ({self.energy:.2f})
- Danceability: {self.danceability:.2f}
- Valence (Positivity): {self.valence:.2f}
- Vocal Content: {speech_desc} ({self.speechiness:.2f})

Structure:
- Intro: {self.intro_duration:.1f}s
- Outro: {self.outro_duration:.1f}s
- Breakdown points: {len(self.breakdown_points)} found
- Buildup points: {len(self.buildup_points)} found
- Drop points: {len(self.drop_points)} found

Audio Characteristics:
- Spectral Centroid: {self.spectral_centroid:.0f}Hz (brightness)
- Zero Crossing Rate: {self.zero_crossing_rate:.3f} (texture)
- Lyric Density: {self.lyric_density:.2f} words/second
"""
        return description

@dataclass 
class TransitionEffect:
    """Represents a specific transition effect with parameters"""
    name: str
    parameters: Dict[str, Any]
    duration_ms: int
    description: str

@dataclass
class AdvancedTransitionPlan:
    """Enhanced transition plan with detailed effects and reasoning"""
    track1_features: AdvancedTrackFeatures
    track2_features: AdvancedTrackFeatures
    
    # LLM reasoning
    llm_analysis: str
    transition_reasoning: str
    
    # Transition details
    primary_effect: TransitionEffect
    secondary_effects: List[TransitionEffect]
    
    # Timing
    transition_start_time: float  # When to start transition in track1
    transition_duration: float    # Total transition duration
    track1_fadeout_time: float   # When track1 starts fading
    track2_fadein_time: float    # When track2 starts fading in
    
    # Quality prediction
    predicted_quality: float     # 0-1 score
    confidence: float           # LLM confidence in decision
    
    # Evaluation
    actual_quality: Optional[float] = None
    evaluation_feedback: Optional[str] = None

class AdvancedAudioAnalyzer:
    """Enhanced audio analyzer with DJ-specific feature extraction"""
    
    def __init__(self):
        self.sr = 22050
        
    def extract_comprehensive_features(self, audio_path: str) -> AdvancedTrackFeatures:
        """Extract all features needed for intelligent DJ transitions"""
        console.print(f"[blue]Analyzing {audio_path}...[/blue]")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        duration = len(y) / sr
        
        # Get basic features from existing analyzer
        basic_features = _analyze_single(audio_path)
        
        # Advanced spectral features
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        
        # Structural analysis
        segments = self._detect_segments(y, sr)
        intro_duration, outro_duration = self._detect_intro_outro(y, sr)
        breakdown_points = self._detect_breakdowns(y, sr)
        buildup_points = self._detect_buildups(y, sr) 
        drop_points = self._detect_drops(y, sr)
        
        # Lyrics (if available)
        lyrics = basic_features.get('lyrics', [])
        lyric_density = basic_features.get('lyric_density', 0.0)
        
        return AdvancedTrackFeatures(
            filename=Path(audio_path).name,
            duration=duration,
            bpm=basic_features['bpm'],
            key=basic_features.get('key'),
            energy=basic_features.get('energy', 0.5),
            danceability=basic_features.get('danceability', 0.5),
            valence=basic_features.get('valence', 0.5),
            speechiness=basic_features.get('speechiness', 0.0),
            beats=basic_features.get('beats', np.array([])),
            onset_frames=onset_frames,
            segments=segments,
            chroma=chroma,
            mfcc=mfcc,
            spectral_centroid=spectral_centroid,
            zero_crossing_rate=zero_crossing_rate,
            lyrics=lyrics,
            lyric_density=lyric_density,
            intro_duration=intro_duration,
            outro_duration=outro_duration,
            breakdown_points=breakdown_points,
            buildup_points=buildup_points,
            drop_points=drop_points
        )
    
    def _detect_segments(self, y: np.ndarray, sr: int) -> List[Dict]:
        """Detect structural segments in the track"""
        # Use librosa's segment detection
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Simple segmentation based on chroma changes
        segment_frames = librosa.segment.agglomerative(chroma, k=8)
        segment_times = librosa.frames_to_time(segment_frames, sr=sr)
        
        segments = []
        for i, (start, end) in enumerate(zip(segment_times[:-1], segment_times[1:])):
            segments.append({
                'id': i,
                'start': float(start),
                'end': float(end),
                'duration': float(end - start),
                'type': self._classify_segment(y[int(start*sr):int(end*sr)], sr)
            })
        
        return segments
    
    def _classify_segment(self, segment_audio: np.ndarray, sr: int) -> str:
        """Classify segment type based on energy and spectral features"""
        if len(segment_audio) == 0:
            return "unknown"
            
        energy = np.mean(librosa.feature.rms(y=segment_audio))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment_audio, sr=sr))
        
        if energy < 0.01:
            return "breakdown"
        elif energy > 0.05 and spectral_centroid > 2000:
            return "drop"
        elif spectral_centroid > 3000:
            return "buildup"
        else:
            return "verse"
    
    def _detect_intro_outro(self, y: np.ndarray, sr: int) -> Tuple[float, float]:
        """Detect intro and outro durations"""
        # Simple energy-based detection
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find where energy stabilizes (intro end) and drops (outro start)
        intro_end = 0
        outro_start = len(y) / sr
        
        # Look for intro (energy ramp up)
        for i in range(min(len(rms), int(30 * sr / hop_length))):  # Max 30s intro
            if rms[i] > 0.7 * np.max(rms[:int(60 * sr / hop_length)]):
                intro_end = i * hop_length / sr
                break
        
        # Look for outro (energy ramp down)
        for i in range(len(rms) - 1, max(0, len(rms) - int(30 * sr / hop_length)), -1):
            if rms[i] > 0.7 * np.max(rms[-int(60 * sr / hop_length):]):
                outro_start = i * hop_length / sr
                break
        
        return float(intro_end), float(len(y)/sr - outro_start)
    
    def _detect_breakdowns(self, y: np.ndarray, sr: int) -> List[float]:
        """Detect breakdown sections (low energy periods)"""
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=512)
        
        # Find points where energy drops significantly
        threshold = 0.3 * np.mean(rms)
        breakdown_points = []
        
        in_breakdown = False
        for i, energy in enumerate(rms):
            if energy < threshold and not in_breakdown:
                breakdown_points.append(float(times[i]))
                in_breakdown = True
            elif energy > threshold * 2:
                in_breakdown = False
        
        return breakdown_points
    
    def _detect_buildups(self, y: np.ndarray, sr: int) -> List[float]:
        """Detect buildup sections (increasing energy/tension)"""
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=512)
        
        # Look for sustained energy increases
        buildup_points = []
        window = 20  # frames to look ahead
        
        for i in range(len(rms) - window):
            if i > window:
                recent_trend = np.polyfit(range(window), rms[i-window:i], 1)[0]
                future_trend = np.polyfit(range(window), rms[i:i+window], 1)[0]
                
                if recent_trend > 0.001 and future_trend > 0.001:  # Sustained increase
                    buildup_points.append(float(times[i]))
        
        return buildup_points
    
    def _detect_drops(self, y: np.ndarray, sr: int) -> List[float]:
        """Detect drop sections (sudden high energy)"""
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Find the strongest onsets that might be drops
        strong_onsets = []
        threshold = 0.8 * np.max(onset_strength)
        
        for i, time in enumerate(onset_times):
            if i < len(onset_strength) and onset_strength[onset_frames[i]] > threshold:
                strong_onsets.append(float(time))
        
        return strong_onsets

class TransitionEffects:
    """Library of professional DJ transition effects"""
    
    @staticmethod
    def crossfade(track1: AudioSegment, track2: AudioSegment, 
                  duration_ms: int, curve: str = "linear") -> AudioSegment:
        """Professional crossfade with different curve options"""
        if curve == "linear":
            return track1.append(track2, crossfade=duration_ms)
        elif curve == "exponential":
            # Custom exponential crossfade
            fade_out = track1.fade_out(duration_ms)
            fade_in = track2.fade_in(duration_ms)
            return fade_out.overlay(fade_in, position=len(track1) - duration_ms)
        elif curve == "s_curve":
            # S-curve crossfade (smoother)
            return TransitionEffects._s_curve_crossfade(track1, track2, duration_ms)
    
    @staticmethod
    def _s_curve_crossfade(track1: AudioSegment, track2: AudioSegment, duration_ms: int) -> AudioSegment:
        """S-curve crossfade for smoother transitions"""
        # Split tracks at transition point
        t1_pre = track1[:-duration_ms]
        t1_fade = track1[-duration_ms:]
        t2_fade = track2[:duration_ms]
        t2_post = track2[duration_ms:]
        
        # Apply S-curve gains
        samples = len(t1_fade)
        for i in range(samples):
            progress = i / samples
            # S-curve function
            s_curve = 0.5 * (1 + np.sin(np.pi * (progress - 0.5)))
            
            # Apply to segments (simplified - would need sample-level processing)
            if i % 1000 == 0:  # Apply every 1000 samples for efficiency
                pos_ms = int(i * 1000 / t1_fade.frame_rate)
                t1_fade = t1_fade[:pos_ms] + t1_fade[pos_ms:pos_ms+1000].apply_gain(-20 * s_curve) + t1_fade[pos_ms+1000:]
                t2_fade = t2_fade[:pos_ms] + t2_fade[pos_ms:pos_ms+1000].apply_gain(-20 * (1-s_curve)) + t2_fade[pos_ms+1000:]
        
        return t1_pre + t1_fade.overlay(t2_fade) + t2_post
    
    @staticmethod
    def beatmatch_cut(track1: AudioSegment, track2: AudioSegment, 
                      cut_point1: float, cut_point2: float) -> AudioSegment:
        """Cut transition on beat boundaries"""
        cut_ms1 = int(cut_point1 * 1000)
        cut_ms2 = int(cut_point2 * 1000)
        
        return track1[:cut_ms1] + track2[cut_ms2:]
    
    @staticmethod
    def filter_sweep(track: AudioSegment, start_freq: int = 200, 
                     end_freq: int = 8000, duration_ms: int = 4000) -> AudioSegment:
        """Apply a filter sweep effect"""
        # This is a simplified version - real implementation would need sample-level processing
        # For now, we'll use EQ approximation
        if start_freq < end_freq:
            # High-pass sweep (removing low frequencies gradually)
            return track.high_pass_filter(start_freq).low_pass_filter(end_freq)
        else:
            # Low-pass sweep (removing high frequencies gradually)
            return track.low_pass_filter(start_freq).high_pass_filter(end_freq)
    
    @staticmethod
    def echo_loop(track: AudioSegment, loop_duration_ms: int = 500, 
                  decay: float = 0.6, repeats: int = 4) -> AudioSegment:
        """Create echo/loop effect"""
        result = track
        for i in range(repeats):
            delay_ms = loop_duration_ms * (i + 1)
            gain_db = -6 * (i + 1) * decay  # Exponential decay
            delayed = AudioSegment.silent(delay_ms) + track.apply_gain(gain_db)
            result = result.overlay(delayed)
        return result
    
    @staticmethod
    def stutter_edit(track: AudioSegment, stutter_duration_ms: int = 125, 
                     repeats: int = 8) -> AudioSegment:
        """Create stutter/glitch effect"""
        stutter_segment = track[:stutter_duration_ms]
        result = AudioSegment.empty()
        
        for _ in range(repeats):
            result += stutter_segment
        result += track[stutter_duration_ms * repeats:]
        
        return result
    
    @staticmethod
    def reverse_buildup(track: AudioSegment, buildup_duration_ms: int = 2000) -> AudioSegment:
        """Create reverse buildup effect"""
        buildup_segment = track[:buildup_duration_ms]
        reversed_buildup = buildup_segment.reverse()
        return reversed_buildup + track

class LLMTransitionReasoner:
    """Uses LLM to reason about optimal transitions"""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.transition_vocabulary = [
            "crossfade_linear", "crossfade_exponential", "crossfade_s_curve",
            "beatmatch_cut", "filter_sweep_high", "filter_sweep_low", 
            "echo_loop", "stutter_edit", "reverse_buildup",
            "quick_cut", "long_blend", "breakdown_mix"
        ]
    
    def analyze_transition(self, track1_features: AdvancedTrackFeatures, 
                          track2_features: AdvancedTrackFeatures,
                          user_prompt: Optional[str] = None) -> AdvancedTransitionPlan:
        """Use LLM to analyze and plan optimal transition"""
        
        console.print("[yellow]Consulting AI DJ for transition analysis...[/yellow]")
        
        # Prepare context for LLM
        context = self._prepare_transition_context(track1_features, track2_features, user_prompt)
        
        # Get LLM analysis
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": context}
            ],
            temperature=0.7
        )
        
        llm_output = response.choices[0].message.content
        
        # Parse LLM response
        transition_plan = self._parse_llm_response(
            llm_output, track1_features, track2_features
        )
        
        return transition_plan
    
    def _get_system_prompt(self) -> str:
        """System prompt for the LLM DJ assistant"""
        return f"""You are an expert DJ with 20+ years of experience in electronic music, hip-hop, and club music. You have an intuitive understanding of energy flow, harmonic mixing, and crowd psychology.

Your job is to analyze two tracks and recommend the optimal transition between them. Consider:

MUSICAL COMPATIBILITY:
- BPM differences (±6 BPM is mixable, ±3 BPM is ideal)
- Harmonic compatibility (Camelot wheel, circle of fifths)
- Energy flow (maintain, build, or break energy appropriately)
- Rhythmic patterns and drum compatibility

TRANSITION TECHNIQUES AVAILABLE:
{', '.join(self.transition_vocabulary)}

STRUCTURAL CONSIDERATIONS:
- Use breakdown sections for complex transitions
- Align drops with energy moments
- Consider intro/outro lengths
- Respect song structures

CREATIVE FACTORS:
- Surprise vs. predictability
- Crowd energy management
- Genre conventions and rule-breaking
- Emotional narrative

OUTPUT FORMAT:
Provide your response as a JSON object with these keys:
- "analysis": Your detailed musical analysis
- "transition_reasoning": Why you chose this specific transition
- "primary_effect": The main transition technique
- "secondary_effects": List of additional effects to apply
- "timing": {
    "transition_start_time": seconds into track1 to start transition,
    "transition_duration": total transition duration in seconds,
    "track1_fadeout_time": when track1 starts fading,
    "track2_fadein_time": when track2 starts fading in
  }
- "predicted_quality": your confidence score (0-1) for this transition
- "confidence": your confidence in this decision (0-1)

Be creative but technically sound. Consider both safe and adventurous options."""

    def _prepare_transition_context(self, track1: AdvancedTrackFeatures, 
                                   track2: AdvancedTrackFeatures,
                                   user_prompt: Optional[str]) -> str:
        """Prepare context for LLM analysis"""
        context = f"""TRANSITION ANALYSIS REQUEST

TRACK 1 (Outgoing):
{track1.to_llm_description()}

TRACK 2 (Incoming):
{track2.to_llm_description()}

COMPATIBILITY QUICK FACTS:
- BPM Difference: {abs(track1.bpm - track2.bpm):.1f} BPM
- Key Relationship: {track1.key} → {track2.key}
- Energy Change: {track2.energy - track1.energy:+.2f}
- Danceability Change: {track2.danceability - track1.danceability:+.2f}
- Speechiness Change: {track2.speechiness - track1.speechiness:+.2f}

AVAILABLE TRANSITION TECHNIQUES:
{', '.join(self.transition_vocabulary)}
"""
        
        if user_prompt:
            context += f"\nUSER REQUEST: {user_prompt}"
            
        context += "\n\nPlease analyze these tracks and recommend the optimal transition strategy."
        
        return context
    
    def _parse_llm_response(self, llm_output: str, 
                           track1: AdvancedTrackFeatures,
                           track2: AdvancedTrackFeatures) -> AdvancedTransitionPlan:
        """Parse LLM response into structured transition plan"""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError):
            # Fallback parsing if JSON extraction fails
            console.print("[red]Warning: Could not parse LLM JSON response, using fallback[/red]")
            response_data = self._fallback_parse(llm_output)
        
        # Create transition effects
        primary_effect = TransitionEffect(
            name=response_data.get("primary_effect", "crossfade_linear"),
            parameters={},
            duration_ms=int(response_data.get("timing", {}).get("transition_duration", 4.0) * 1000),
            description=f"Primary transition: {response_data.get('primary_effect', 'crossfade')}"
        )
        
        secondary_effects = []
        for effect_name in response_data.get("secondary_effects", []):
            secondary_effects.append(TransitionEffect(
                name=effect_name,
                parameters={},
                duration_ms=2000,  # Default 2 seconds
                description=f"Secondary effect: {effect_name}"
            ))
        
        timing = response_data.get("timing", {})
        
        return AdvancedTransitionPlan(
            track1_features=track1,
            track2_features=track2,
            llm_analysis=response_data.get("analysis", "Analysis not available"),
            transition_reasoning=response_data.get("transition_reasoning", "Reasoning not available"),
            primary_effect=primary_effect,
            secondary_effects=secondary_effects,
            transition_start_time=timing.get("transition_start_time", track1.duration - 10),
            transition_duration=timing.get("transition_duration", 4.0),
            track1_fadeout_time=timing.get("track1_fadeout_time", track1.duration - 8),
            track2_fadein_time=timing.get("track2_fadein_time", 0),
            predicted_quality=response_data.get("predicted_quality", 0.7),
            confidence=response_data.get("confidence", 0.8)
        )
    
    def _fallback_parse(self, llm_output: str) -> Dict:
        """Fallback parsing when JSON extraction fails"""
        # Simple keyword-based extraction
        fallback = {
            "analysis": llm_output[:200] + "...",
            "transition_reasoning": "Fallback: Basic transition selected",
            "primary_effect": "crossfade_linear",
            "secondary_effects": [],
            "timing": {
                "transition_start_time": 30,
                "transition_duration": 4.0,
                "track1_fadeout_time": 32,
                "track2_fadein_time": 0
            },
            "predicted_quality": 0.6,
            "confidence": 0.5
        }
        
        # Try to extract effect names
        for effect in self.transition_vocabulary:
            if effect.lower() in llm_output.lower():
                fallback["primary_effect"] = effect
                break
                
        return fallback

class TransitionEvaluator:
    """Evaluates the quality of completed transitions"""
    
    def __init__(self):
        self.analyzer = AdvancedAudioAnalyzer()
    
    def evaluate_transition(self, mixed_audio_path: str, 
                           transition_plan: AdvancedTransitionPlan) -> Tuple[float, str]:
        """Evaluate the quality of a completed transition"""
        console.print("[cyan]Evaluating transition quality...[/cyan]")
        
        # Extract features from the mixed audio
        mixed_features = self.analyzer.extract_comprehensive_features(mixed_audio_path)
        
        # Analyze transition region
        transition_start = transition_plan.transition_start_time
        transition_end = transition_start + transition_plan.transition_duration
        
        quality_score = self._calculate_quality_score(
            mixed_features, transition_start, transition_end, transition_plan
        )
        
        feedback = self._generate_feedback(quality_score, transition_plan)
        
        return quality_score, feedback
    
    def _calculate_quality_score(self, mixed_features: AdvancedTrackFeatures,
                                transition_start: float, transition_end: float,
                                plan: AdvancedTransitionPlan) -> float:
        """Calculate objective quality score"""
        scores = []
        
        # 1. Energy flow continuity (0.3 weight)
        energy_score = self._evaluate_energy_flow(mixed_features, transition_start, transition_end)
        scores.append(('energy', energy_score, 0.3))
        
        # 2. Rhythmic alignment (0.25 weight)
        rhythm_score = self._evaluate_rhythmic_alignment(mixed_features, transition_start, transition_end)
        scores.append(('rhythm', rhythm_score, 0.25))
        
        # 3. Harmonic compatibility (0.2 weight)
        harmony_score = self._evaluate_harmonic_compatibility(plan)
        scores.append(('harmony', harmony_score, 0.2))
        
        # 4. Structural appropriateness (0.15 weight)
        structure_score = self._evaluate_structural_appropriateness(plan)
        scores.append(('structure', structure_score, 0.15))
        
        # 5. Creative factor (0.1 weight)
        creative_score = self._evaluate_creativity(plan)
        scores.append(('creativity', creative_score, 0.1))
        
        # Weighted average
        total_score = sum(score * weight for _, score, weight in scores)
        
        # Log detailed scores
        console.print("\n[cyan]Quality Analysis:[/cyan]")
        for category, score, weight in scores:
            console.print(f"  {category.capitalize()}: {score:.2f} (weight: {weight})")
        console.print(f"  [bold]Total Score: {total_score:.2f}[/bold]")
        
        return total_score
    
    def _evaluate_energy_flow(self, features: AdvancedTrackFeatures, 
                             start: float, end: float) -> float:
        """Evaluate energy continuity during transition"""
        # This is simplified - real implementation would analyze energy curve
        expected_change = features.track2_features.energy - features.track1_features.energy
        if abs(expected_change) < 0.1:
            return 0.9  # Smooth energy transition
        elif abs(expected_change) < 0.3:
            return 0.7  # Moderate energy change
        else:
            return 0.5  # Large energy change (could be good or bad)
    
    def _evaluate_rhythmic_alignment(self, features: AdvancedTrackFeatures,
                                   start: float, end: float) -> float:
        """Evaluate rhythmic alignment quality"""
        # Simplified - would analyze beat alignment in transition region
        return 0.8  # Placeholder
    
    def _evaluate_harmonic_compatibility(self, plan: AdvancedTransitionPlan) -> float:
        """Evaluate harmonic compatibility between tracks"""
        key1 = plan.track1_features.key
        key2 = plan.track2_features.key
        
        if not key1 or not key2:
            return 0.6  # Unknown keys
        
        # Simple harmonic compatibility rules
        if key1 == key2:
            return 1.0  # Perfect match
        
        # Check for compatible keys (simplified)
        major_keys = ["C", "G", "D", "A", "E", "B", "F#", "F", "Bb", "Eb", "Ab", "Db"]
        minor_keys = ["Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "Dm", "Gm", "Cm", "Fm", "Bbm"]
        
        # Adjacent keys in circle of fifths
        try:
            if key1 in major_keys and key2 in major_keys:
                idx1, idx2 = major_keys.index(key1), major_keys.index(key2)
                distance = min(abs(idx1 - idx2), 12 - abs(idx1 - idx2))
                return max(0.4, 1.0 - distance * 0.15)
            elif key1 in minor_keys and key2 in minor_keys:
                idx1, idx2 = minor_keys.index(key1), minor_keys.index(key2)
                distance = min(abs(idx1 - idx2), 12 - abs(idx1 - idx2))
                return max(0.4, 1.0 - distance * 0.15)
            else:
                return 0.6  # Mixed major/minor
        except ValueError:
            return 0.5  # Unknown key format
    
    def _evaluate_structural_appropriateness(self, plan: AdvancedTransitionPlan) -> float:
        """Evaluate if transition timing respects song structure"""
        # Check if transition uses appropriate sections
        t1_features = plan.track1_features
        t2_features = plan.track2_features
        
        score = 0.7  # Base score
        
        # Bonus for using outro/intro sections
        if plan.transition_start_time > (t1_features.duration - t1_features.outro_duration - 5):
            score += 0.2  # Started in outro region
        
        if plan.track2_fadein_time < t2_features.intro_duration:
            score += 0.1  # Fade in during intro
        
        return min(1.0, score)
    
    def _evaluate_creativity(self, plan: AdvancedTransitionPlan) -> float:
        """Evaluate creative aspects of the transition"""
        creativity_score = 0.5  # Base score
        
        # Bonus for interesting effects
        if len(plan.secondary_effects) > 0:
            creativity_score += 0.2
        
        # Bonus for non-standard primary effects
        if plan.primary_effect.name not in ["crossfade_linear", "crossfade_exponential"]:
            creativity_score += 0.3
        
        return min(1.0, creativity_score)
    
    def _generate_feedback(self, quality_score: float, plan: AdvancedTransitionPlan) -> str:
        """Generate descriptive feedback about the transition"""
        if quality_score >= 0.8:
            feedback = "🎵 Excellent transition! "
        elif quality_score >= 0.6:
            feedback = "🎶 Good transition. "
        elif quality_score >= 0.4:
            feedback = "🎵 Decent transition with room for improvement. "
        else:
            feedback = "🎶 Challenging transition - consider alternative approach. "
        
        # Add specific observations
        bpm_diff = abs(plan.track1_features.bpm - plan.track2_features.bpm)
        if bpm_diff > 6:
            feedback += f"Large BPM difference ({bpm_diff:.1f}) handled well. "
        
        energy_diff = plan.track2_features.energy - plan.track1_features.energy
        if energy_diff > 0.3:
            feedback += "Good energy buildup. "
        elif energy_diff < -0.3:
            feedback += "Effective energy breakdown. "
        
        if plan.primary_effect.name != "crossfade_linear":
            feedback += f"Creative use of {plan.primary_effect.name}. "
        
        return feedback

class LLMDJSystem:
    """Main system that orchestrates all components"""
    
    def __init__(self):
        self.analyzer = AdvancedAudioAnalyzer()
        self.reasoner = LLMTransitionReasoner()
        self.evaluator = TransitionEvaluator()
        self.effects = TransitionEffects()
    
    def create_transition(self, track1_path: str, track2_path: str,
                         output_path: str, user_prompt: Optional[str] = None) -> AdvancedTransitionPlan:
        """Main method to create an intelligent transition between two tracks"""
        
        console.print(Panel.fit(
            "[bold blue]🎧 LLM DJ System - Intelligent Transition Creation[/bold blue]",
            style="blue"
        ))
        
        with Progress() as progress:
            # Step 1: Analyze both tracks
            task1 = progress.add_task("Analyzing tracks...", total=100)
            
            track1_features = self.analyzer.extract_comprehensive_features(track1_path)
            progress.update(task1, advance=50)
            
            track2_features = self.analyzer.extract_comprehensive_features(track2_path)
            progress.update(task1, advance=50)
            
            # Step 2: LLM reasoning
            task2 = progress.add_task("AI reasoning...", total=100)
            transition_plan = self.reasoner.analyze_transition(
                track1_features, track2_features, user_prompt
            )
            progress.update(task2, advance=100)
            
            # Step 3: Apply transition
            task3 = progress.add_task("Applying transition...", total=100)
            mixed_audio = self._apply_transition_plan(track1_path, track2_path, transition_plan)
            progress.update(task3, advance=80)
            
            # Step 4: Export result
            mixed_audio.export(output_path, format="wav")
            progress.update(task3, advance=20)
        
        # Step 5: Evaluate result
        quality_score, feedback = self.evaluator.evaluate_transition(output_path, transition_plan)
        transition_plan.actual_quality = quality_score
        transition_plan.evaluation_feedback = feedback
        
        # Display results
        self._display_results(transition_plan)
        
        return transition_plan
    
    def _apply_transition_plan(self, track1_path: str, track2_path: str,
                              plan: AdvancedTransitionPlan) -> AudioSegment:
        """Apply the transition plan to create mixed audio"""
        
        # Load audio segments
        track1 = AudioSegment.from_file(track1_path)
        track2 = AudioSegment.from_file(track2_path)
        
        # Apply primary transition effect
        effect_name = plan.primary_effect.name
        duration_ms = plan.primary_effect.duration_ms
        
        if effect_name.startswith("crossfade"):
            curve = effect_name.split("_")[1] if "_" in effect_name else "linear"
            mixed = self.effects.crossfade(track1, track2, duration_ms, curve)
            
        elif effect_name == "beatmatch_cut":
            cut_point1 = plan.transition_start_time
            cut_point2 = plan.track2_fadein_time
            mixed = self.effects.beatmatch_cut(track1, track2, cut_point1, cut_point2)
            
        elif effect_name.startswith("filter_sweep"):
            # Apply filter to transition region
            direction = effect_name.split("_")[2] if len(effect_name.split("_")) > 2 else "high"
            if direction == "high":
                filtered_track1 = self.effects.filter_sweep(track1, 200, 8000, duration_ms)
            else:
                filtered_track1 = self.effects.filter_sweep(track1, 8000, 200, duration_ms)
            mixed = filtered_track1.append(track2, crossfade=duration_ms//2)
            
        elif effect_name == "echo_loop":
            echo_track1 = self.effects.echo_loop(track1)
            mixed = echo_track1.append(track2, crossfade=duration_ms)
            
        elif effect_name == "stutter_edit":
            stutter_track1 = self.effects.stutter_edit(track1)
            mixed = stutter_track1.append(track2, crossfade=duration_ms)
            
        elif effect_name == "reverse_buildup":
            reverse_track2 = self.effects.reverse_buildup(track2)
            mixed = track1.append(reverse_track2, crossfade=duration_ms)
            
        else:
            # Default to linear crossfade
            mixed = self.effects.crossfade(track1, track2, duration_ms, "linear")
        
        # Apply secondary effects
        for effect in plan.secondary_effects:
            # Apply additional processing
            if effect.name == "compress":
                mixed = mixed.compress_dynamic_range()
            elif effect.name == "normalize":
                mixed = normalize(mixed)
        
        return mixed
    
    def _display_results(self, plan: AdvancedTransitionPlan):
        """Display comprehensive results of the transition"""
        
        # Create results table
        table = Table(title="🎧 Transition Analysis Results")
        table.add_column("Aspect", style="cyan", no_wrap=True)
        table.add_column("Details", style="white")
        
        table.add_row("Tracks", f"{plan.track1_features.filename} → {plan.track2_features.filename}")
        table.add_row("BPM Change", f"{plan.track1_features.bpm:.1f} → {plan.track2_features.bpm:.1f}")
        table.add_row("Key Change", f"{plan.track1_features.key} → {plan.track2_features.key}")
        table.add_row("Primary Effect", plan.primary_effect.name)
        table.add_row("Secondary Effects", ", ".join([e.name for e in plan.secondary_effects]) or "None")
        table.add_row("Transition Duration", f"{plan.transition_duration:.1f}s")
        table.add_row("Predicted Quality", f"{plan.predicted_quality:.2f}/1.0")
        table.add_row("Actual Quality", f"{plan.actual_quality:.2f}/1.0" if plan.actual_quality else "Not evaluated")
        table.add_row("AI Confidence", f"{plan.confidence:.2f}/1.0")
        
        console.print(table)
        
        # Display AI reasoning
        console.print(Panel(
            plan.llm_analysis,
            title="🤖 AI Analysis",
            expand=False
        ))
        
        console.print(Panel(
            plan.transition_reasoning,
            title="🎛️ Transition Reasoning",
            expand=False
        ))
        
        if plan.evaluation_feedback:
            console.print(Panel(
                plan.evaluation_feedback,
                title="📊 Quality Evaluation",
                expand=False
            ))

# CLI Interface
@click.command()
@click.argument('track1', type=click.Path(exists=True))
@click.argument('track2', type=click.Path(exists=True))
@click.option('--output', '-o', default='transition_output.wav', help='Output file path')
@click.option('--prompt', '-p', help='Custom prompt for the AI DJ')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def main(track1, track2, output, prompt, verbose):
    """
    LLM-Powered DJ System
    
    Create intelligent transitions between two audio tracks using AI reasoning.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize system
    dj_system = LLMDJSystem()
    
    try:
        # Create transition
        transition_plan = dj_system.create_transition(track1, track2, output, prompt)
        
        console.print(f"\n✅ [green]Transition created successfully![/green]")
        console.print(f"📁 Output saved to: [bold]{output}[/bold]")
        console.print(f"⭐ Quality Score: [bold]{transition_plan.actual_quality:.2f}/1.0[/bold]")
        
    except Exception as e:
        console.print(f"❌ [red]Error creating transition: {e}[/red]")
        if verbose:
            console.print_exception()

if __name__ == "__main__":
    main() 