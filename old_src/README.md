# 🎧 LLM-Powered DJ System

An intelligent DJ system that uses Large Language Models to create sophisticated audio transitions between tracks. This system combines advanced audio analysis, AI reasoning, and professional DJ techniques to automatically generate high-quality transitions that respect musical theory, energy flow, and creative expression.

## 🚀 Features

### 🎵 Advanced Audio Analysis
- **Musical Features**: BPM, key detection, energy levels, danceability metrics
- **Structural Analysis**: Intro/outro detection, breakdown identification, drop points
- **Spectral Analysis**: MFCC, chroma features, spectral centroid analysis
- **Lyrical Analysis**: Speech-to-text processing and lyric density calculation

### 🤖 LLM-Powered Reasoning
- **GPT-4 Integration**: Uses OpenAI's GPT-4 for intelligent transition decisions
- **Context-Aware Planning**: Considers musical compatibility, energy flow, and user prompts
- **Natural Language Interface**: Accept custom instructions in plain English
- **Chain-of-Thought**: Transparent reasoning process for educational insights

### 🎛️ Professional Transition Effects
- **Crossfades**: Linear, exponential, and S-curve crossfades
- **Beat-matched Cuts**: Precise timing on beat boundaries
- **Filter Sweeps**: High-pass and low-pass filter transitions
- **Creative Effects**: Echo loops, stutter edits, reverse buildups
- **Dynamic Processing**: Compression and normalization

### 📊 Quality Evaluation
- **Automated Scoring**: Multi-factor quality assessment
- **Real-time Feedback**: Analysis of transition success
- **Detailed Metrics**: Energy flow, harmonic compatibility, structural appropriateness
- **Comparative Analysis**: Compare different transition approaches

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- OpenAI API key
- Audio files in WAV or MP3 format

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/your-username/llm-dj-system.git
cd llm-dj-system

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key-here"

# Optional: Set up Spotify API for enhanced metadata
export SPOTIPY_CLIENT_ID="your-spotify-client-id"
export SPOTIPY_CLIENT_SECRET="your-spotify-client-secret"
export ACOUSTID_KEY="your-acoustid-api-key"
```

### Dependencies

```
librosa>=0.9.0        # Audio analysis
pydub>=0.25.0         # Audio processing
soundfile>=0.10.0     # Audio I/O
openai>=1.0.0         # LLM integration
rich>=12.0.0          # Beautiful terminal output
click>=8.0.0          # CLI interface
matplotlib>=3.5.0     # Visualization
scipy>=1.8.0          # Signal processing
numpy>=1.21.0         # Numerical computing
```

## 🎯 Quick Start

### Command Line Interface

```bash
# Basic transition between two tracks
python llm_dj_system.py track1.wav track2.wav

# Custom output file
python llm_dj_system.py track1.wav track2.wav --output my_mix.wav

# Custom AI prompt
python llm_dj_system.py track1.wav track2.wav --prompt "Create a dramatic breakdown transition"

# Verbose output for debugging
python llm_dj_system.py track1.wav track2.wav --verbose
```

### Interactive Demo

```bash
# Launch the interactive demo
python demo_dj_system.py
```

The demo provides several options:
- **Scenario Demo**: Try predefined transition scenarios
- **Custom Transition**: Create transitions with custom prompts
- **Track Analysis**: Deep dive into individual track features
- **Comparison Mode**: Compare different transition approaches
- **System Info**: View capabilities and configuration

### Python API

```python
from llm_dj_system import LLMDJSystem

# Initialize the system
dj = LLMDJSystem()

# Create a transition
transition_plan = dj.create_transition(
    track1_path="song1.wav",
    track2_path="song2.wav", 
    output_path="transition.wav",
    user_prompt="Make it smooth and energetic"
)

# Access detailed analysis
print(f"AI Analysis: {transition_plan.llm_analysis}")
print(f"Quality Score: {transition_plan.actual_quality}")
print(f"Effect Used: {transition_plan.primary_effect.name}")
```

## 🎨 Usage Examples

### Scenario-Based Transitions

```python
# Energy buildup scenario
dj.create_transition(
    "mellow_track.wav", 
    "high_energy_track.wav",
    "buildup_mix.wav",
    "Build energy gradually from the breakdown section"
)

# Harmonic mixing
dj.create_transition(
    "track_in_C.wav",
    "track_in_G.wav", 
    "harmonic_mix.wav",
    "Focus on the harmonic relationship - use the circle of fifths"
)

# Creative experimentation
dj.create_transition(
    "track1.wav",
    "track2.wav",
    "creative_mix.wav", 
    "Be experimental! Try some glitch effects and reverse elements"
)
```

### Track Analysis

```python
# Analyze individual tracks
analyzer = AdvancedAudioAnalyzer()
features = analyzer.extract_comprehensive_features("my_track.wav")

print(f"BPM: {features.bpm}")
print(f"Key: {features.key}")
print(f"Energy: {features.energy}")
print(f"Breakdown points: {features.breakdown_points}")
```

### Transition Comparison

```python
# Compare different approaches
approaches = [
    ("Conservative", "Focus on technical precision and safety"),
    ("Creative", "Be experimental with interesting effects"),
    ("Energy-focused", "Maintain high energy throughout")
]

results = []
for name, prompt in approaches:
    plan = dj.create_transition("track1.wav", "track2.wav", f"{name.lower()}_mix.wav", prompt)
    results.append((name, plan))

# Find best approach
best = max(results, key=lambda x: x[1].actual_quality)
print(f"Best approach: {best[0]} (Quality: {best[1].actual_quality:.2f})")
```

## 🎛️ Transition Effects Library

### Crossfades
- **Linear**: Standard linear volume crossfade
- **Exponential**: Exponential curve for more natural fade
- **S-Curve**: Smooth S-shaped curve for professional results

### Beat-Matched Transitions
- **Beat Cut**: Precise cut on beat boundaries
- **BPM Sync**: Automatic tempo adjustment
- **Rhythmic Alignment**: Beat-grid synchronization

### Creative Effects
- **Filter Sweeps**: High-pass/low-pass frequency sweeps
- **Echo Loops**: Rhythmic echo and delay effects
- **Stutter Edits**: Glitch-style repetition effects
- **Reverse Buildup**: Dramatic reverse elements

### Dynamic Processing
- **Compression**: Dynamic range control
- **Normalization**: Level matching
- **EQ Adjustments**: Frequency balance

## 🔬 Technical Architecture

### Core Components

1. **AdvancedAudioAnalyzer**: Deep audio feature extraction
2. **LLMTransitionReasoner**: AI-powered decision making
3. **TransitionEffects**: Professional effect library
4. **TransitionEvaluator**: Quality assessment system
5. **LLMDJSystem**: Main orchestration class

### Audio Processing Pipeline

```
Audio Input → Feature Extraction → LLM Analysis → Effect Selection → Audio Processing → Quality Evaluation → Output
```

### LLM Integration

The system uses OpenAI's GPT-4 with a specialized prompt that includes:
- Musical theory knowledge
- DJ technique expertise
- Transition effect vocabulary
- Context-aware reasoning
- Creative and technical balance

## 📊 Quality Evaluation Metrics

### Objective Measures
- **Energy Flow Continuity** (30%): Smooth energy transitions
- **Rhythmic Alignment** (25%): Beat synchronization quality
- **Harmonic Compatibility** (20%): Key relationship adherence
- **Structural Appropriateness** (15%): Timing with song structure
- **Creative Factor** (10%): Innovation and interest level

### Feedback Categories
- 🎵 Excellent (0.8-1.0): Professional quality
- 🎶 Good (0.6-0.8): Solid performance
- 🎵 Decent (0.4-0.6): Acceptable with improvements
- 🎶 Challenging (<0.4): Needs different approach

## 🎯 Advanced Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional - Enhanced metadata
export SPOTIPY_CLIENT_ID="your-spotify-id"
export SPOTIPY_CLIENT_SECRET="your-spotify-secret"
export ACOUSTID_KEY="your-acoustid-key"

# Optional - Model selection
export LLM_MODEL="gpt-4"  # or "gpt-3.5-turbo"
```

### Custom LLM Prompts

You can customize the system's behavior by modifying prompts:

```python
reasoner = LLMTransitionReasoner(model="gpt-4")
# Modify reasoner._get_system_prompt() for custom behavior
```

### Audio Processing Settings

```python
analyzer = AdvancedAudioAnalyzer()
analyzer.sr = 44100  # Sample rate
# Customize feature extraction parameters
```

## 🚫 Limitations

- **Audio Quality**: Output quality depends on input file quality
- **Genre Specificity**: Optimized for electronic, hip-hop, and dance music
- **Processing Time**: Complex analysis can take 30-60 seconds per track
- **API Costs**: OpenAI API usage costs apply
- **File Formats**: Best results with high-quality WAV files

## 🔮 Future Enhancements

### Planned Features
- **Real-time Processing**: Live DJ performance capability
- **MIDI Integration**: Hardware controller support
- **Advanced Effects**: More sophisticated audio processing
- **Genre Expansion**: Classical, jazz, and world music support
- **Multi-track Mixing**: Handle more than two tracks simultaneously

### Research Directions
- **Crowd Response Modeling**: Audience feedback integration
- **Style Transfer**: Genre-crossing transitions
- **Emotional Arc Planning**: Long-form set construction
- **Collaborative AI**: Human-AI DJ partnerships

## 🤝 Contributing

We welcome contributions! Areas of interest:

### Development
- Audio processing improvements
- New transition effects
- LLM prompt optimization
- Performance enhancements

### Research
- Music information retrieval
- AI reasoning for creative tasks
- Audio quality metrics
- User experience studies

### Testing
- Track library expansion
- Edge case identification
- Quality assessment
- Cross-platform testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **librosa**: Comprehensive audio analysis library
- **OpenAI**: GPT-4 language model capabilities
- **pydub**: Simple audio processing in Python
- **Spotify**: Music metadata and audio features API
- **Rich**: Beautiful terminal formatting

## 📞 Support

- **Documentation**: Check this README and code comments
- **Issues**: Use GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Email**: [maintainer@example.com](mailto:maintainer@example.com)

---

*Built with ❤️ for the DJ and AI communities*

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Python**: 3.9+  
**Status**: Active Development