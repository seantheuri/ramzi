#!/usr/bin/env python3
"""
LLM DJ System - Example Usage
=============================

This script demonstrates how to use the LLM-powered DJ system
with the existing audio files in the project directory.
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from llm_dj_system import LLMDJSystem, console

def check_requirements():
    """Check if all requirements are met"""
    console.print(Panel.fit("🔍 Checking Requirements", style="blue"))
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]❌ OpenAI API key not found![/red]")
        console.print("[yellow]Please set your OpenAI API key:[/yellow]")
        console.print("export OPENAI_API_KEY='your-api-key-here'")
        return False
    else:
        console.print("[green]✅ OpenAI API key found[/green]")
    
    # Check for audio files
    audio_files = list(Path('.').glob('*.wav')) + list(Path('.').glob('*.mp3'))
    if len(audio_files) < 2:
        console.print("[red]❌ Need at least 2 audio files in current directory[/red]")
        console.print("[yellow]Available files:[/yellow]")
        for file in audio_files:
            console.print(f"  - {file.name}")
        return False
    else:
        console.print(f"[green]✅ Found {len(audio_files)} audio files[/green]")
        for file in audio_files[:5]:  # Show first 5
            console.print(f"  - {file.name}")
        if len(audio_files) > 5:
            console.print(f"  ... and {len(audio_files) - 5} more")
    
    return True

def basic_transition_example():
    """Demonstrate basic transition creation"""
    console.print(Panel.fit("🎵 Basic Transition Example", style="green"))
    
    # Find available audio files
    audio_files = list(Path('.').glob('*.wav')) + list(Path('.').glob('*.mp3'))
    
    if len(audio_files) < 2:
        console.print("[red]Not enough audio files for demo[/red]")
        return
    
    # Use first two files
    track1 = str(audio_files[0])
    track2 = str(audio_files[1])
    output = "example_basic_transition.wav"
    
    console.print(f"[cyan]Creating transition:[/cyan]")
    console.print(f"  Track 1: {Path(track1).name}")
    console.print(f"  Track 2: {Path(track2).name}")
    console.print(f"  Output: {output}")
    
    # Initialize DJ system
    dj = LLMDJSystem()
    
    try:
        # Create transition with basic prompt
        plan = dj.create_transition(
            track1, 
            track2, 
            output,
            "Create a smooth, professional transition that maintains energy"
        )
        
        console.print(f"\n[green]✅ Transition created successfully![/green]")
        console.print(f"Quality Score: {plan.actual_quality:.2f}/1.0")
        console.print(f"AI Confidence: {plan.confidence:.2f}/1.0")
        console.print(f"Effect Used: {plan.primary_effect.name}")
        
        return plan
        
    except Exception as e:
        console.print(f"[red]❌ Error creating transition: {e}[/red]")
        return None

def creative_transition_example():
    """Demonstrate creative transition with custom prompt"""
    console.print(Panel.fit("🎨 Creative Transition Example", style="magenta"))
    
    audio_files = list(Path('.').glob('*.wav')) + list(Path('.').glob('*.mp3'))
    
    if len(audio_files) < 2:
        console.print("[red]Not enough audio files for demo[/red]")
        return
    
    # Use different files if available
    track1 = str(audio_files[0])
    track2 = str(audio_files[-1])  # Use last file
    output = "example_creative_transition.wav"
    
    console.print(f"[cyan]Creating creative transition:[/cyan]")
    console.print(f"  Track 1: {Path(track1).name}")
    console.print(f"  Track 2: {Path(track2).name}")
    console.print(f"  Output: {output}")
    
    dj = LLMDJSystem()
    
    try:
        # Create transition with creative prompt
        plan = dj.create_transition(
            track1,
            track2,
            output,
            "Be experimental! Use interesting effects like stutters or reverse elements. "
            "Make this transition memorable and unique while respecting the music."
        )
        
        console.print(f"\n[green]✅ Creative transition created![/green]")
        console.print(f"Quality Score: {plan.actual_quality:.2f}/1.0")
        console.print(f"Primary Effect: {plan.primary_effect.name}")
        console.print(f"Secondary Effects: {[e.name for e in plan.secondary_effects]}")
        
        return plan
        
    except Exception as e:
        console.print(f"[red]❌ Error creating creative transition: {e}[/red]")
        return None

def analyze_track_example():
    """Demonstrate individual track analysis"""
    console.print(Panel.fit("🔍 Track Analysis Example", style="yellow"))
    
    audio_files = list(Path('.').glob('*.wav')) + list(Path('.').glob('*.mp3'))
    
    if not audio_files:
        console.print("[red]No audio files found for analysis[/red]")
        return
    
    track_file = str(audio_files[0])
    console.print(f"[cyan]Analyzing: {Path(track_file).name}[/cyan]")
    
    dj = LLMDJSystem()
    
    try:
        # Extract comprehensive features
        features = dj.analyzer.extract_comprehensive_features(track_file)
        
        # Display key information
        console.print(f"\n[green]📊 Analysis Results:[/green]")
        console.print(f"  BPM: {features.bpm:.1f}")
        console.print(f"  Key: {features.key or 'Unknown'}")
        console.print(f"  Energy: {features.energy:.2f}")
        console.print(f"  Duration: {features.duration:.1f}s")
        console.print(f"  Intro: {features.intro_duration:.1f}s")
        console.print(f"  Outro: {features.outro_duration:.1f}s")
        console.print(f"  Breakdowns: {len(features.breakdown_points)}")
        console.print(f"  Buildups: {len(features.buildup_points)}")
        console.print(f"  Drops: {len(features.drop_points)}")
        
        return features
        
    except Exception as e:
        console.print(f"[red]❌ Error analyzing track: {e}[/red]")
        return None

def comparison_example():
    """Demonstrate comparing different transition approaches"""
    console.print(Panel.fit("⚖️ Transition Comparison Example", style="cyan"))
    
    audio_files = list(Path('.').glob('*.wav')) + list(Path('.').glob('*.mp3'))
    
    if len(audio_files) < 2:
        console.print("[red]Not enough audio files for comparison[/red]")
        return
    
    track1 = str(audio_files[0])
    track2 = str(audio_files[1])
    
    console.print(f"[cyan]Comparing transition approaches:[/cyan]")
    console.print(f"  Track 1: {Path(track1).name}")
    console.print(f"  Track 2: {Path(track2).name}")
    
    dj = LLMDJSystem()
    
    approaches = [
        ("Safe", "Create a safe, professional transition focusing on technical precision"),
        ("Creative", "Be experimental with interesting effects and creative techniques"),
        ("Energy", "Focus on building energy and excitement throughout the transition")
    ]
    
    results = []
    
    for approach_name, prompt in approaches:
        console.print(f"\n[yellow]Creating {approach_name} approach...[/yellow]")
        output = f"example_comparison_{approach_name.lower()}.wav"
        
        try:
            plan = dj.create_transition(track1, track2, output, prompt)
            results.append((approach_name, plan))
            console.print(f"  Quality: {plan.actual_quality:.2f}, Effect: {plan.primary_effect.name}")
            
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
    
    if results:
        # Find best approach
        best = max(results, key=lambda x: x[1].actual_quality)
        console.print(f"\n[green]🏆 Best approach: {best[0]} "
                     f"(Quality: {best[1].actual_quality:.2f})[/green]")
    
    return results

def main():
    """Main example function"""
    console.print(Panel.fit(
        "[bold blue]🎧 LLM DJ System - Example Usage[/bold blue]\n"
        "[green]This script demonstrates the key features of the AI DJ system[/green]",
        style="blue"
    ))
    
    # Check requirements
    if not check_requirements():
        console.print("\n[red]Please fix the requirements above before running examples.[/red]")
        return
    
    console.print("\n[green]All requirements met! Running examples...[/green]")
    
    try:
        # Example 1: Basic transition
        console.print("\n" + "="*60)
        basic_plan = basic_transition_example()
        
        # Example 2: Track analysis
        console.print("\n" + "="*60)
        track_features = analyze_track_example()
        
        # Example 3: Creative transition
        console.print("\n" + "="*60)
        creative_plan = creative_transition_example()
        
        # Example 4: Comparison
        console.print("\n" + "="*60)
        comparison_results = comparison_example()
        
        # Summary
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "[bold green]🎉 Examples Complete![/bold green]\n"
            f"Generated {len([p for p in [basic_plan, creative_plan] if p])} transitions\n"
            "Check the output files in the current directory",
            style="green"
        ))
        
        # List output files
        output_files = list(Path('.').glob('example_*.wav'))
        if output_files:
            console.print("\n[cyan]Generated files:[/cyan]")
            for file in output_files:
                console.print(f"  🎵 {file.name}")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Examples interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error running examples: {e}[/red]")

if __name__ == "__main__":
    main() 