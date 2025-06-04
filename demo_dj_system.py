"""
LLM DJ System Demo
==================

Interactive demonstration of the LLM-powered DJ system with various transition scenarios.
This script showcases different capabilities and allows users to experiment with 
the AI DJ in different contexts.
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import print as rprint
from llm_dj_system import LLMDJSystem, console
import json

class DJDemo:
    """Interactive demo class for the LLM DJ System"""
    
    def __init__(self):
        self.dj_system = LLMDJSystem()
        self.demo_scenarios = {
            "1": {
                "name": "Classic Hip-Hop to Electronic",
                "description": "Transition from a classic hip-hop track to modern electronic",
                "prompt": "Create a smooth transition that maintains energy but shifts from hip-hop to electronic vibes"
            },
            "2": {
                "name": "BPM Challenge",
                "description": "Handle a large BPM difference with creative transition",
                "prompt": "Handle the BPM difference creatively, maybe use some effects to make it exciting"
            },
            "3": {
                "name": "Energy Buildup",
                "description": "Build energy from mellow to high-energy",
                "prompt": "Build energy gradually - start the transition when the first track is winding down and ramp up intensity"
            },
            "4": {
                "name": "Harmonic Mixing",
                "description": "Focus on harmonic compatibility and smooth mixing",
                "prompt": "Focus on harmonic mixing - use the key relationship to create a musically pleasing transition"
            },
            "5": {
                "name": "Creative Experiment",
                "description": "Let the AI get creative with unusual effects",
                "prompt": "Be creative! Try some interesting effects and techniques - surprise me with something unique"
            }
        }
    
    def run_demo(self):
        """Run the interactive demo"""
        console.print(Panel.fit(
            "[bold blue]🎧 LLM DJ System - Interactive Demo[/bold blue]\n"
            "[green]Welcome to the AI-powered DJ experience![/green]",
            style="blue"
        ))
        
        while True:
            try:
                choice = self.main_menu()
                
                if choice == "1":
                    self.run_scenario_demo()
                elif choice == "2":
                    self.run_custom_demo()
                elif choice == "3":
                    self.analyze_tracks()
                elif choice == "4":
                    self.compare_transitions()
                elif choice == "5":
                    self.show_system_info()
                elif choice == "6":
                    console.print("[yellow]Thanks for using the LLM DJ System! 🎵[/yellow]")
                    break
                else:
                    console.print("[red]Invalid choice. Please try again.[/red]")
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Demo interrupted. Goodbye! 👋[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]An error occurred: {e}[/red]")
                if Confirm.ask("Would you like to continue?"):
                    continue
                else:
                    break
    
    def main_menu(self) -> str:
        """Display main menu and get user choice"""
        console.print("\n" + "="*60)
        console.print("[bold cyan]🎛️ Main Menu[/bold cyan]")
        console.print("="*60)
        
        options = [
            ("1", "🎵 Run Scenario Demo", "Try predefined transition scenarios"),
            ("2", "🎨 Custom Transition", "Create your own transition with custom prompt"),
            ("3", "🔍 Analyze Tracks", "Deep analysis of track features"),
            ("4", "⚖️ Compare Transitions", "Compare different transition approaches"),
            ("5", "ℹ️ System Info", "View system capabilities and configuration"),
            ("6", "🚪 Exit", "Exit the demo")
        ]
        
        table = Table(show_header=False, box=None)
        table.add_column("Choice", style="bold cyan", width=5)
        table.add_column("Option", style="bold white", width=25)
        table.add_column("Description", style="dim white")
        
        for choice, option, desc in options:
            table.add_row(choice, option, desc)
        
        console.print(table)
        
        return Prompt.ask("\n[bold]Choose an option", choices=["1", "2", "3", "4", "5", "6"])
    
    def run_scenario_demo(self):
        """Run predefined scenario demonstrations"""
        console.print("\n[bold cyan]🎵 Scenario Demo[/bold cyan]")
        
        # Show available scenarios
        scenarios_table = Table(title="Available Scenarios")
        scenarios_table.add_column("ID", style="cyan", width=5)
        scenarios_table.add_column("Name", style="bold white", width=25)
        scenarios_table.add_column("Description", style="dim white")
        
        for scenario_id, scenario in self.demo_scenarios.items():
            scenarios_table.add_row(scenario_id, scenario["name"], scenario["description"])
        
        console.print(scenarios_table)
        
        # Get user choice
        scenario_choice = Prompt.ask(
            "\nSelect a scenario",
            choices=list(self.demo_scenarios.keys()),
            default="1"
        )
        
        scenario = self.demo_scenarios[scenario_choice]
        console.print(f"\n[green]Selected: {scenario['name']}[/green]")
        console.print(f"[dim]Prompt: {scenario['prompt']}[/dim]")
        
        # Get track files
        track1, track2 = self.get_track_files()
        if not track1 or not track2:
            return
            
        # Generate output filename
        output_file = f"demo_scenario_{scenario_choice}_{Path(track1).stem}_to_{Path(track2).stem}.wav"
        
        # Run transition
        self.execute_transition(track1, track2, output_file, scenario['prompt'])
    
    def run_custom_demo(self):
        """Run custom transition with user-provided prompt"""
        console.print("\n[bold cyan]🎨 Custom Transition Demo[/bold cyan]")
        
        # Get track files
        track1, track2 = self.get_track_files()
        if not track1 or not track2:
            return
            
        # Get custom prompt
        console.print("\n[yellow]Describe how you want the AI DJ to handle this transition:[/yellow]")
        console.print("[dim]Examples:")
        console.print("- 'Make it dramatic with some echo effects'")
        console.print("- 'Keep it smooth and professional'")
        console.print("- 'Use the breakdown section for a creative mix'[/dim]")
        
        custom_prompt = Prompt.ask("\nYour custom prompt", default="Create the best transition possible")
        
        # Generate output filename
        output_file = f"demo_custom_{Path(track1).stem}_to_{Path(track2).stem}.wav"
        
        # Run transition
        self.execute_transition(track1, track2, output_file, custom_prompt)
    
    def analyze_tracks(self):
        """Analyze individual tracks in detail"""
        console.print("\n[bold cyan]🔍 Track Analysis[/bold cyan]")
        
        track_file = self.get_single_track_file()
        if not track_file:
            return
            
        console.print(f"[yellow]Analyzing {track_file}...[/yellow]")
        
        # Extract features
        features = self.dj_system.analyzer.extract_comprehensive_features(track_file)
        
        # Display comprehensive analysis
        self.display_track_analysis(features)
    
    def compare_transitions(self):
        """Compare different transition approaches for the same track pair"""
        console.print("\n[bold cyan]⚖️ Transition Comparison[/bold cyan]")
        
        # Get track files
        track1, track2 = self.get_track_files()
        if not track1 or not track2:
            return
            
        console.print("\n[yellow]Creating multiple transitions with different approaches...[/yellow]")
        
        comparison_prompts = [
            ("Conservative", "Create a safe, professional transition that focuses on technical precision"),
            ("Creative", "Be experimental and creative - try interesting effects and techniques"),
            ("Energy-focused", "Focus on maintaining and building energy throughout the transition"),
            ("Harmonic", "Prioritize harmonic compatibility and musical theory"),
        ]
        
        results = []
        
        for approach_name, prompt in comparison_prompts:
            console.print(f"\n[cyan]Creating {approach_name} transition...[/cyan]")
            output_file = f"comparison_{approach_name.lower()}_{Path(track1).stem}_to_{Path(track2).stem}.wav"
            
            try:
                plan = self.dj_system.create_transition(track1, track2, output_file, prompt)
                results.append((approach_name, plan, output_file))
            except Exception as e:
                console.print(f"[red]Error creating {approach_name} transition: {e}[/red]")
        
        # Compare results
        self.display_comparison_results(results)
    
    def show_system_info(self):
        """Display system information and capabilities"""
        console.print("\n[bold cyan]ℹ️ System Information[/bold cyan]")
        
        # System capabilities
        capabilities = [
            ("Audio Analysis", "BPM, key, energy, structure, spectral features"),
            ("LLM Reasoning", "GPT-4 powered transition decision making"),
            ("Transition Effects", "Crossfades, cuts, filters, loops, stutters"),
            ("Quality Evaluation", "Automated feedback and scoring"),
            ("Supported Formats", "WAV, MP3, FLAC (via pydub)"),
            ("API Integration", "Spotify metadata, OpenAI GPT models")
        ]
        
        info_table = Table(title="🎧 LLM DJ System Capabilities")
        info_table.add_column("Feature", style="bold cyan", width=20)
        info_table.add_column("Details", style="white")
        
        for feature, details in capabilities:
            info_table.add_row(feature, details)
        
        console.print(info_table)
        
        # Transition vocabulary
        effects_table = Table(title="🎛️ Available Transition Effects")
        effects_table.add_column("Effect", style="bold yellow", width=20)
        effects_table.add_column("Description", style="white")
        
        effect_descriptions = {
            "crossfade_linear": "Standard linear crossfade",
            "crossfade_exponential": "Exponential curve crossfade",
            "crossfade_s_curve": "S-curve crossfade (smooth)",
            "beatmatch_cut": "Cut transition on beat boundaries",
            "filter_sweep_high": "High-pass filter sweep",
            "filter_sweep_low": "Low-pass filter sweep",
            "echo_loop": "Echo/delay loop effect",
            "stutter_edit": "Stutter/glitch effect",
            "reverse_buildup": "Reverse buildup effect",
            "quick_cut": "Quick cut transition",
            "long_blend": "Extended blend transition",
            "breakdown_mix": "Mix during breakdown section"
        }
        
        for effect, desc in effect_descriptions.items():
            effects_table.add_row(effect, desc)
        
        console.print(effects_table)
    
    def get_track_files(self) -> tuple:
        """Get two track files from user"""
        console.print("\n[yellow]Track Selection[/yellow]")
        
        # Check for demo files
        available_files = list(Path('.').glob('*.wav')) + list(Path('.').glob('*.mp3'))
        
        if available_files:
            console.print("[green]Available audio files in current directory:[/green]")
            for i, file in enumerate(available_files, 1):
                console.print(f"  {i}. {file.name}")
            
            if Confirm.ask("\nUse files from current directory?", default=True):
                if len(available_files) >= 2:
                    console.print("\nSelect first track:")
                    for i, file in enumerate(available_files, 1):
                        console.print(f"  {i}. {file.name}")
                    
                    choice1 = int(Prompt.ask("Track 1", choices=[str(i) for i in range(1, len(available_files)+1)])) - 1
                    track1 = str(available_files[choice1])
                    
                    remaining_files = [f for i, f in enumerate(available_files) if i != choice1]
                    console.print("\nSelect second track:")
                    for i, file in enumerate(remaining_files, 1):
                        console.print(f"  {i}. {file.name}")
                    
                    choice2 = int(Prompt.ask("Track 2", choices=[str(i) for i in range(1, len(remaining_files)+1)])) - 1
                    track2 = str(remaining_files[choice2])
                    
                    return track1, track2
        
        # Manual file entry
        track1 = Prompt.ask("Enter path to first track")
        track2 = Prompt.ask("Enter path to second track")
        
        # Validate files exist
        if not Path(track1).exists():
            console.print(f"[red]File not found: {track1}[/red]")
            return None, None
        if not Path(track2).exists():
            console.print(f"[red]File not found: {track2}[/red]")
            return None, None
            
        return track1, track2
    
    def get_single_track_file(self) -> str:
        """Get a single track file from user"""
        available_files = list(Path('.').glob('*.wav')) + list(Path('.').glob('*.mp3'))
        
        if available_files:
            console.print("[green]Available audio files:[/green]")
            for i, file in enumerate(available_files, 1):
                console.print(f"  {i}. {file.name}")
            
            if Confirm.ask("\nSelect from available files?", default=True):
                choice = int(Prompt.ask("Select file", choices=[str(i) for i in range(1, len(available_files)+1)])) - 1
                return str(available_files[choice])
        
        # Manual entry
        track_file = Prompt.ask("Enter path to track")
        if not Path(track_file).exists():
            console.print(f"[red]File not found: {track_file}[/red]")
            return None
            
        return track_file
    
    def execute_transition(self, track1: str, track2: str, output_file: str, prompt: str):
        """Execute a transition and display results"""
        try:
            console.print(f"\n[yellow]Creating transition: {Path(track1).name} → {Path(track2).name}[/yellow]")
            console.print(f"[dim]Output: {output_file}[/dim]")
            console.print(f"[dim]Prompt: {prompt}[/dim]")
            
            # Create transition
            plan = self.dj_system.create_transition(track1, track2, output_file, prompt)
            
            # Offer to save analysis
            if Confirm.ask("\nSave detailed analysis to JSON file?"):
                json_file = output_file.replace('.wav', '_analysis.json')
                self.save_analysis_to_json(plan, json_file)
                console.print(f"[green]Analysis saved to {json_file}[/green]")
            
            # Offer to play result (if system supports it)
            if Confirm.ask("Open output file location?"):
                import subprocess
                import platform
                
                if platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", "-R", output_file])
                elif platform.system() == "Windows":
                    subprocess.run(["explorer", "/select,", output_file])
                else:  # Linux
                    subprocess.run(["xdg-open", Path(output_file).parent])
            
        except Exception as e:
            console.print(f"[red]Error creating transition: {e}[/red]")
    
    def display_track_analysis(self, features):
        """Display comprehensive track analysis"""
        # Basic info table
        basic_table = Table(title=f"📊 Analysis: {features.filename}")
        basic_table.add_column("Property", style="cyan", width=20)
        basic_table.add_column("Value", style="white", width=30)
        basic_table.add_column("Interpretation", style="dim white")
        
        # Add rows with interpretations
        basic_table.add_row(
            "Duration", 
            f"{features.duration:.1f} seconds",
            f"{features.duration//60:.0f}:{features.duration%60:02.0f}"
        )
        basic_table.add_row(
            "BPM", 
            f"{features.bpm:.1f}",
            "Fast" if features.bpm > 140 else "Medium" if features.bpm > 100 else "Slow"
        )
        basic_table.add_row(
            "Key", 
            features.key or "Unknown",
            "Major" if features.key and 'm' not in features.key else "Minor" if features.key else "N/A"
        )
        basic_table.add_row(
            "Energy", 
            f"{features.energy:.2f}",
            "High" if features.energy > 0.7 else "Medium" if features.energy > 0.4 else "Low"
        )
        basic_table.add_row(
            "Danceability", 
            f"{features.danceability:.2f}",
            "Very danceable" if features.danceability > 0.8 else "Danceable" if features.danceability > 0.5 else "Less danceable"
        )
        basic_table.add_row(
            "Speechiness", 
            f"{features.speechiness:.2f}",
            "Vocal heavy" if features.speechiness > 0.5 else "Some vocals" if features.speechiness > 0.2 else "Mostly instrumental"
        )
        
        console.print(basic_table)
        
        # Structure table
        structure_table = Table(title="🏗️ Track Structure")
        structure_table.add_column("Element", style="cyan", width=20)
        structure_table.add_column("Value", style="white")
        
        structure_table.add_row("Intro Duration", f"{features.intro_duration:.1f}s")
        structure_table.add_row("Outro Duration", f"{features.outro_duration:.1f}s")
        structure_table.add_row("Breakdown Points", str(len(features.breakdown_points)))
        structure_table.add_row("Buildup Points", str(len(features.buildup_points)))
        structure_table.add_row("Drop Points", str(len(features.drop_points)))
        structure_table.add_row("Segments Detected", str(len(features.segments)))
        
        console.print(structure_table)
        
        # Audio characteristics
        audio_table = Table(title="🔊 Audio Characteristics")
        audio_table.add_column("Feature", style="cyan", width=20)
        audio_table.add_column("Value", style="white", width=15)
        audio_table.add_column("Description", style="dim white")
        
        audio_table.add_row(
            "Spectral Centroid", 
            f"{features.spectral_centroid:.0f} Hz",
            "Brightness/timbre characteristic"
        )
        audio_table.add_row(
            "Zero Crossing Rate", 
            f"{features.zero_crossing_rate:.3f}",
            "Texture indicator"
        )
        audio_table.add_row(
            "Lyric Density", 
            f"{features.lyric_density:.2f} w/s",
            "Words per second"
        )
        
        console.print(audio_table)
    
    def display_comparison_results(self, results):
        """Display comparison of different transition approaches"""
        if not results:
            console.print("[red]No successful transitions to compare[/red]")
            return
        
        comparison_table = Table(title="⚖️ Transition Comparison Results")
        comparison_table.add_column("Approach", style="bold cyan", width=15)
        comparison_table.add_column("Primary Effect", style="yellow", width=20)
        comparison_table.add_column("Duration", style="white", width=10)
        comparison_table.add_column("Quality Score", style="green", width=12)
        comparison_table.add_column("AI Confidence", style="blue", width=12)
        comparison_table.add_column("File", style="dim white")
        
        for approach_name, plan, output_file in results:
            comparison_table.add_row(
                approach_name,
                plan.primary_effect.name,
                f"{plan.transition_duration:.1f}s",
                f"{plan.actual_quality:.2f}/1.0" if plan.actual_quality else "N/A",
                f"{plan.confidence:.2f}/1.0",
                Path(output_file).name
            )
        
        console.print(comparison_table)
        
        # Show best performing approach
        if all(plan.actual_quality for _, plan, _ in results):
            best_approach = max(results, key=lambda x: x[1].actual_quality)
            console.print(f"\n[green]🏆 Best performing approach: {best_approach[0]} "
                         f"(Quality: {best_approach[1].actual_quality:.2f})[/green]")
    
    def save_analysis_to_json(self, plan, filename):
        """Save transition analysis to JSON file"""
        # Convert plan to JSON-serializable format
        analysis_data = {
            "track1": {
                "filename": plan.track1_features.filename,
                "bpm": plan.track1_features.bpm,
                "key": plan.track1_features.key,
                "energy": plan.track1_features.energy,
                "duration": plan.track1_features.duration
            },
            "track2": {
                "filename": plan.track2_features.filename,
                "bpm": plan.track2_features.bpm,
                "key": plan.track2_features.key,
                "energy": plan.track2_features.energy,
                "duration": plan.track2_features.duration
            },
            "transition": {
                "primary_effect": plan.primary_effect.name,
                "secondary_effects": [e.name for e in plan.secondary_effects],
                "duration": plan.transition_duration,
                "start_time": plan.transition_start_time,
                "predicted_quality": plan.predicted_quality,
                "actual_quality": plan.actual_quality,
                "confidence": plan.confidence
            },
            "llm_analysis": plan.llm_analysis,
            "transition_reasoning": plan.transition_reasoning,
            "evaluation_feedback": plan.evaluation_feedback
        }
        
        with open(filename, 'w') as f:
            json.dump(analysis_data, f, indent=2)

def main():
    """Main entry point for the demo"""
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable not set![/red]")
        console.print("[yellow]Please set your OpenAI API key:[/yellow]")
        console.print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Initialize and run demo
    demo = DJDemo()
    demo.run_demo()

if __name__ == "__main__":
    main() 