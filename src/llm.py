import os
import json
import argparse
from openai import OpenAI
from typing import Dict, List, Any

def summarize_track_data(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a comprehensive summary of track data for the LLM prompt."""
    file_name = os.path.basename(analysis_data.get('file_path', 'N/A'))
    
    # Basic beat insight (trim beat grid for brevity)
    beat_grid = analysis_data.get('beat_grid_seconds', [])
    beat_info = {
        "first_beat": round(beat_grid[0], 3) if beat_grid else 0,
        "beat_count": len(beat_grid),
        "average_interval": round(((beat_grid[-1] - beat_grid[0]) / len(beat_grid)), 3) if len(beat_grid) > 1 else 0,
        "preview_grid": [round(b, 3) for b in beat_grid[:32]]  # first 32 beats
    }

    # Section summaries (label + start)
    sections_raw = analysis_data.get('structural_analysis', [])
    section_summaries = [
        {
            "label": s.get('label'),
            "start": round(s.get('start', 0), 2)
        } for s in sections_raw
    ]

    # Lyrics
    lyrics_text_full = (analysis_data.get('lyrics_text') or "").strip()
    lyrics_excerpt = lyrics_text_full[:300] + ("…" if len(lyrics_text_full) > 300 else "")

    # Loudness / dynamics
    dynamics = {
        "loudness": analysis_data.get('spectral_features', {}).get('loudness', analysis_data.get('track', {}).get('loudness', -20)),
        "peak_loudness": analysis_data.get('peak_loudness'),
        "loudness_range": analysis_data.get('loudness_range')
    }

    summary = {
        "track_name": file_name,
        "bpm": analysis_data.get('bpm'),
        "tempo_confidence": analysis_data.get('track', {}).get('tempo_confidence'),
        "key_standard": analysis_data.get('key_standard'),
        "key_camelot": analysis_data.get('key_camelot'),
        "key_confidence": analysis_data.get('track', {}).get('key_confidence'),
        "mode": analysis_data.get('track', {}).get('mode'),
        "time_signature": analysis_data.get('track', {}).get('time_signature'),
        "energy_normalized": analysis_data.get('energy_normalized'),
        "dynamics": dynamics,
        "sections": section_summaries,
        "beat_info": beat_info,
        "lyrics_excerpt": lyrics_excerpt,
        "lyrics_available": bool(lyrics_text_full),
        "duration": analysis_data.get('track', {}).get('duration', 0),
        "stems_available": ["full", "vocals", "beat"],
    }

    return summary

def generate_llm_prompt(summary_a: Dict[str, Any], summary_b: Dict[str, Any], 
                       user_preference: str = "A professional club-style mix with creative elements.",
                       mix_style: str = "blend") -> str:
    """Constructs the comprehensive prompt for advanced DJ mixing."""
    
    prompt = f"""You are RAMZI, an expert virtual DJ with access to a professional DDJ-FLX4 controller. Create an innovative and technically impressive mix transition from Song A to Song B.

## YOUR DJ CONTROLLER CAPABILITIES:

### DECK CONTROLS:
- `load_track`: Load track with optional BPM adjustment - params: deck, file_path, target_bpm
- `play`, `pause`, `stop`: Basic transport - params: deck
- `cue_set`, `cue_play`, `back_cue`: Cue point control - params: deck
- `seek_to_time`: Jump to specific time - params: deck, time_in_seconds

### HOT CUES (8 per deck):
- `set_hot_cue`: Set hot cue at current position - params: deck, cue_idx (0-7)
- `jump_hot_cue`: Jump to hot cue - params: deck, cue_idx
- `delete_hot_cue`: Remove hot cue - params: deck, cue_idx

### LOOP CONTROLS:
- `loop_in`, `loop_out`: Set manual loop points - params: deck
- `loop_exit`: Exit active loop - params: deck
- `loop_half`, `loop_double`: Adjust loop size - params: deck
- `auto_loop`: Auto loop with beat length - params: deck, beats (0.25 to 32)

### BEAT CONTROLS:
- `beat_jump_forward`, `beat_jump_backward`: Jump by beats - params: deck, beats
- `beat_sync`: Sync to master tempo - params: deck
- `set_master_deck`: Set tempo master - params: deck
- `set_tempo`: Adjust tempo ±6% - params: deck, value (-1.0 to 1.0)

### PERFORMANCE MODES:
- `set_performance_mode`: Change pad mode - params: deck, mode (hot_cue/pad_fx1/pad_fx2/beat_jump/beat_loop/sampler/key_shift)
- `key_shift`: Pitch shift - params: deck, semitones (-7 to +7)
- `toggle_pad_fx`: Activate pad effects - params: deck, effect_id (1-32)

### STEM SELECTION:
- `set_segment`: Instantly switch the playing audio on a deck between "full", "vocals" (harmonic), or "beat" (percussive) stems. Params: deck, segment ('full'|'vocals'|'beat')

### MIXER CONTROLS:
- `set_parameter`: Control mixer knobs/faders - params: deck, parameter, value, fade_duration
  - Parameters: volume, trim, channel_fader, eq_high, eq_mid, eq_low, filter_cutoff_hz
  - Values: 0.0-1.0 (0.5 = neutral for EQ), filter in Hz (20-20000)
- `set_crossfader`: Crossfader position - params: position (0.0-1.0), fade_duration, curve (linear/smooth/cut)
- `set_smart_fader`: Enable smart fader - params: enabled, type (smooth/cut/scratch)

### EFFECTS:
- `set_beat_fx`: Apply beat effects - params: fx_type, channel, beats, depth
  - Types: echo, reverb, phaser, flanger, pitch, roll, transform
- `beat_fx_on`, `beat_fx_off`: Toggle beat FX
- `set_color_fx`: Color/Smart CFX - params: deck, fx_type (filter/noise/pitch/none), params
- `beat_repeat`: Loop roll effect - params: deck, activate (true/false), beats

### CREATIVE TOOLS:
- `jog_wheel`: Scratch/pitch bend - params: deck, scratch (bool), delta/bend
- `sampler_load`, `sampler_play`, `sampler_stop`: Sample control - params: slot (0-15)

### PAD FX EFFECTS (1-32):
1-16: Reverb, Delay, Phaser, Chorus, Distortion, Compressor, Bitcrush, HPF, LPF, etc.
17-32: Additional reverbs, delays, filters, pitch shifts, special effects

## TRACK DATA:

**SONG A (Outgoing):**
```json
{json.dumps(summary_a, indent=2)}
```

**SONG B (Incoming):**
```json
{json.dumps(summary_b, indent=2)}
```

## MIX REQUIREMENTS:

1. **Mix Style**: {mix_style}
2. **User Preference**: {user_preference}
3. **Duration**: Create a mix that showcases both tracks with a memorable transition
4. **Creativity**: Use at least 3 different controller features creatively
5. **Technical Excellence**: Demonstrate professional mixing techniques

## YOUR TASK:

1. **Analyze**: Study BPMs, keys, energy levels, and song structures
2. **Plan**: Design a creative transition strategy using the controller features
3. **Execute**: Write a detailed mix script with precise timing

Consider:
- Harmonic mixing (use Camelot wheel compatibility)
- Energy flow and dynamics
- Creative use of loops, hot cues, and effects
- EQ techniques for smooth frequency transitions
- Beat matching and phrase alignment
- Memorable moments using pad FX or beat repeat

## OUTPUT FORMAT:

Generate a JSON object with these exact keys:
{{
  "description": "Your mixing strategy explanation (2-3 sentences)",
  "technique_highlights": ["List", "of", "key", "techniques", "used"],
  "total_duration": 240.0,  // Total mix duration in seconds
  "script": [
    // Chronologically ordered commands
    {{"time": 0.0, "command": "load_track", "params": {{"deck": "A", "file_path": "{summary_a['track_name']}", "target_bpm": 128}}}},
    // ... more commands
  ]
}}

The output must be ONLY the JSON object. No explanations, no markdown, just pure JSON.
"""
    return prompt

def get_mix_style_prompt(style: str) -> str:
    """Returns specific instructions based on mix style."""
    styles = {
        "blend": "Create a long, smooth blend where both tracks play together for extended periods",
        "cut": "Use quick cuts and transforms for an energetic, hip-hop style mix",
        "creative": "Showcase advanced techniques like beat juggling, loops, and effects",
        "harmonic": "Focus on key-compatible mixing with smooth harmonic transitions",
        "buildup": "Create tension and release using filters, effects, and energy management",
        "mashup": "Layer elements creatively, using loops and hot cues to create a unique blend"
    }
    return styles.get(style, styles["blend"])

def validate_mix_script(script_data: Dict[str, Any]) -> bool:
    """Validates that the generated script follows correct format."""
    required_keys = ["description", "total_duration", "script"]
    if not all(key in script_data for key in required_keys):
        return False
    
    if not isinstance(script_data["script"], list):
        return False
    
    # Check that tracks are loaded
    load_commands = [cmd for cmd in script_data["script"] if cmd.get("command") == "load_track"]
    if len(load_commands) < 2:
        print("Warning: Script should load both tracks")
        return False
    
    # Validate command structure
    for cmd in script_data["script"]:
        if "time" not in cmd or "command" not in cmd:
            return False
        if cmd["command"] != "load_track" and "params" not in cmd:
            return False
    
    return True

def enhance_script_with_defaults(script_data: Dict[str, Any], 
                               analysis_a: Dict[str, Any], 
                               analysis_b: Dict[str, Any],
                               analysis_file_a: str = None,
                               analysis_file_b: str = None) -> Dict[str, Any]:
    """Adds helpful defaults and fixes common issues in generated scripts."""
    
    # Ensure proper file paths in load commands (use analysis file paths instead of song paths)
    for cmd in script_data["script"]:
        if cmd["command"] == "load_track":
            if cmd["params"]["deck"] == "A":
                cmd["params"]["file_path"] = analysis_file_a if analysis_file_a else analysis_a["file_path"]
            elif cmd["params"]["deck"] == "B":
                cmd["params"]["file_path"] = analysis_file_b if analysis_file_b else analysis_b["file_path"]
    
    # Sort commands by time
    script_data["script"].sort(key=lambda x: x["time"])
    
    # Add technique highlights if missing
    if "technique_highlights" not in script_data:
        # Analyze script to identify techniques used
        techniques = set()
        for cmd in script_data["script"]:
            if cmd["command"] in ["set_hot_cue", "jump_hot_cue"]:
                techniques.add("Hot cues")
            elif cmd["command"] in ["auto_loop", "loop_in", "loop_out"]:
                techniques.add("Looping")
            elif cmd["command"] in ["beat_jump_forward", "beat_jump_backward"]:
                techniques.add("Beat jumping")
            elif cmd["command"] == "toggle_pad_fx":
                techniques.add("Pad FX")
            elif cmd["command"] == "set_beat_fx":
                techniques.add("Beat FX")
            elif cmd["command"] == "beat_repeat":
                techniques.add("Beat repeat/Roll")
            elif "eq_" in cmd.get("params", {}).get("parameter", ""):
                techniques.add("EQ mixing")
            elif cmd.get("params", {}).get("parameter") == "filter_cutoff_hz":
                techniques.add("Filter sweeps")
        
        script_data["technique_highlights"] = list(techniques)
    
    return script_data

def generate_mix_script(analysis_file_a: str, analysis_file_b: str, 
                       user_prompt: str, output_file_path: str,
                       mix_style: str = "blend",
                       model: str = "gpt-4-turbo-preview") -> None:
    """Main function to generate advanced DJ mix scripts."""
    
    print("🎧 RAMZI Virtual DJ - Advanced Mix Generator")
    print("=" * 50)
    
    print("\n📁 Loading analysis files...")
    with open(analysis_file_a, 'r') as f:
        data_a = json.load(f)
    with open(analysis_file_b, 'r') as f:
        data_b = json.load(f)

    summary_a = summarize_track_data(data_a)
    summary_b = summarize_track_data(data_b)
    
    print(f"\n🎵 Track A: {summary_a['track_name']} ({summary_a['bpm']} BPM, {summary_a['key_camelot']})")
    print(f"🎵 Track B: {summary_b['track_name']} ({summary_b['bpm']} BPM, {summary_b['key_camelot']})")
    
    # Check harmonic compatibility
    camelot_a = summary_a.get('key_camelot', '')
    camelot_b = summary_b.get('key_camelot', '')
    if camelot_a and camelot_b:
        # Simple compatibility check (same number or adjacent)
        num_a = int(camelot_a[:-1]) if camelot_a[:-1].isdigit() else 0
        num_b = int(camelot_b[:-1]) if camelot_b[:-1].isdigit() else 0
        letter_a = camelot_a[-1]
        letter_b = camelot_b[-1]
        
        if (num_a == num_b and letter_a == letter_b) or \
           (abs(num_a - num_b) == 1 and letter_a == letter_b) or \
           (num_a == num_b and letter_a != letter_b):
            print("✅ Keys are harmonically compatible!")
        else:
            print("⚠️  Keys may require creative mixing or key shifting")
    
    print(f"\n🎚️  Mix style: {mix_style}")
    print(f"💭 User preference: {user_prompt}")
    
    # Add style-specific instructions
    full_preference = f"{user_prompt} {get_mix_style_prompt(mix_style)}"
    
    print("\n🤖 Generating LLM prompt...")
    prompt = generate_llm_prompt(summary_a, summary_b, full_preference, mix_style)
    
    try:
        print(f"\n📡 Sending to OpenAI API (using {model})...")
        client = OpenAI()  # Assumes OPENAI_API_KEY is set
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are RAMZI, an expert virtual DJ. Output only valid JSON for DJ mix scripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # Balanced creativity
            max_tokens=4000   # Ensure we can handle complex scripts
        )
        
        raw_response = response.choices[0].message.content
        
        # Clean and parse response
        json_response = raw_response.strip()
        if json_response.startswith('```'):
            # Remove markdown code blocks
            json_response = json_response.split('```')[1]
            if json_response.startswith('json'):
                json_response = json_response[4:]
        
        print("\n🔍 Parsing and validating mix script...")
        mix_script_data = json.loads(json_response)
        
        if not validate_mix_script(mix_script_data):
            raise ValueError("Generated script failed validation")
        
        # Enhance with defaults and fixes
        mix_script_data = enhance_script_with_defaults(mix_script_data, data_a, data_b, analysis_file_a, analysis_file_b)
        
        # Save the script
        with open(output_file_path, 'w') as f:
            json.dump(mix_script_data, f, indent=2)
            
        print(f"\n✅ Successfully generated mix script: {output_file_path}")
        print(f"\n🎯 Mix Strategy: {mix_script_data.get('description', 'N/A')}")
        
        if 'technique_highlights' in mix_script_data:
            print(f"\n🎛️  Techniques used:")
            for technique in mix_script_data['technique_highlights']:
                print(f"   • {technique}")
        
        print(f"\n⏱️  Total duration: {mix_script_data['total_duration']} seconds")
        print(f"📊 Commands in script: {len(mix_script_data['script'])}")
        
        # Show command breakdown
        command_counts = {}
        for cmd in mix_script_data['script']:
            command_counts[cmd['command']] = command_counts.get(cmd['command'], 0) + 1
        
        print("\n📋 Command breakdown:")
        for cmd, count in sorted(command_counts.items()):
            print(f"   • {cmd}: {count}")
        
    except json.JSONDecodeError as e:
        print(f"\n❌ Failed to parse JSON response: {e}")
        print("\n--- Raw AI Response ---")
        print(raw_response)
        raise
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your OpenAI API key is set correctly")
        print("2. Ensure you have API credits available")
        print("3. Try using a different model (gpt-4, gpt-3.5-turbo)")
        raise

def create_example_scripts():
    """Generate example mix scripts for different styles."""
    examples = {
        "smooth_blend": {
            "description": "Classic long blend with EQ swap and filter sweeps",
            "technique_highlights": ["EQ mixing", "Filter sweeps", "Harmonic mixing"],
            "total_duration": 300.0,
            "script": [
                {"time": 0.0, "command": "load_track", "params": {"deck": "A", "file_path": "track1.json", "target_bpm": 128}},
                {"time": 0.0, "command": "load_track", "params": {"deck": "B", "file_path": "track2.json", "target_bpm": 128}},
                {"time": 0.1, "command": "play", "params": {"deck": "A"}},
                {"time": 0.1, "command": "set_crossfader", "params": {"position": 0.0}},
                {"time": 120.0, "command": "play", "params": {"deck": "B"}},
                {"time": 120.0, "command": "set_parameter", "params": {"deck": "B", "parameter": "eq_low", "value": 0.0}},
                {"time": 128.0, "command": "set_parameter", "params": {"deck": "B", "parameter": "eq_low", "value": 0.5, "fade_duration": 16.0}},
                {"time": 128.0, "command": "set_parameter", "params": {"deck": "A", "parameter": "eq_low", "value": 0.0, "fade_duration": 16.0}},
                {"time": 120.0, "command": "set_crossfader", "params": {"position": 1.0, "fade_duration": 32.0}}
            ]
        },
        "creative_scratch": {
            "description": "Hip-hop style mix with scratching, beat juggling, and transforms",
            "technique_highlights": ["Scratching", "Beat juggling", "Hot cues", "Transform"],
            "total_duration": 180.0,
            "script": [
                {"time": 0.0, "command": "load_track", "params": {"deck": "A", "file_path": "track1.json"}},
                {"time": 0.0, "command": "load_track", "params": {"deck": "B", "file_path": "track2.json"}},
                {"time": 0.1, "command": "set_smart_fader", "params": {"enabled": true, "type": "scratch"}},
                {"time": 0.1, "command": "play", "params": {"deck": "A"}},
                {"time": 10.0, "command": "set_hot_cue", "params": {"deck": "A", "cue_idx": 0}},
                {"time": 12.0, "command": "set_hot_cue", "params": {"deck": "A", "cue_idx": 1}},
                {"time": 20.0, "command": "jog_wheel", "params": {"deck": "A", "scratch": true, "delta": -0.5}},
                {"time": 20.2, "command": "jog_wheel", "params": {"deck": "A", "scratch": true, "delta": 0.5}}
            ]
        }
    }
    
    for name, script in examples.items():
        with open(f"example_{name}.json", 'w') as f:
            json.dump(script, f, indent=2)
        print(f"Created example: example_{name}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate advanced DJ mix scripts using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm_dj.py track1_analysis.json track2_analysis.json -o mix.json
  python llm_dj.py track1_analysis.json track2_analysis.json -o mix.json -s creative -p "Festival main stage energy"
  python llm_dj.py --create-examples
        """
    )
    
    parser.add_argument("analysis_file_a", type=str, nargs='?', help="Path to analysis JSON for outgoing track")
    parser.add_argument("analysis_file_b", type=str, nargs='?', help="Path to analysis JSON for incoming track")
    parser.add_argument("-o", "--output_file", type=str, help="Output path for mix script JSON")
    parser.add_argument("-p", "--prompt", type=str, 
                       default="Create a professional mix that showcases both tracks effectively",
                       help="User preference for mixing style")
    parser.add_argument("-s", "--style", type=str, 
                       choices=["blend", "cut", "creative", "harmonic", "buildup", "mashup"],
                       default="blend",
                       help="Mix style preset")
    parser.add_argument("-m", "--model", type=str,
                       default="gpt-4o",
                       help="OpenAI model to use")
    parser.add_argument("--create-examples", action="store_true",
                       help="Create example mix scripts")
    
    args = parser.parse_args()
    
    if args.create_examples:
        create_example_scripts()
    elif args.analysis_file_a and args.analysis_file_b and args.output_file:
        generate_mix_script(
            args.analysis_file_a, 
            args.analysis_file_b, 
            args.prompt, 
            args.output_file,
            args.style,
            args.model
        )
    else:
        parser.print_help()