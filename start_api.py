#!/usr/bin/env python3
"""
RAMZI DJ API Launcher
===================

Convenient script to start the RAMZI DJ API server with proper configuration.
"""

import os
import sys
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['flask', 'flask_cors', 'librosa', 'openai', 'numpy', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\nInstall them with:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_environment():
    """Check environment variables and configuration"""
    warnings = []
    
    # Check OpenAI API key
    if not os.environ.get('OPENAI_API_KEY'):
        warnings.append("⚠️  OPENAI_API_KEY not set - lyrics transcription and mix generation will fail")
    
    # Check secret key for production
    if not os.environ.get('SECRET_KEY'):
        warnings.append("⚠️  SECRET_KEY not set - using development key (not secure for production)")
    
    if warnings:
        print("Environment warnings:")
        for warning in warnings:
            print(f"   {warning}")
        print()

def setup_directories():
    """Create necessary directories"""
    directories = ['uploads', 'analysis', 'mix_scripts', 'rendered_mixes']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directory ready: {directory}/")

def print_api_endpoints():
    """Print available API endpoints"""
    print("\n🔗 Available API Endpoints:")
    print("\n📋 Core Endpoints:")
    print("   GET  /api/health                      - Health check")
    print("   GET  /api/config                      - API configuration")
    print("   GET  /api/tracks                      - List all tracks")
    print("   POST /api/tracks/upload               - Upload audio file")
    print("   POST /api/tracks/{id}/analyze         - Start track analysis")
    print("   GET  /api/tracks/{id}/analysis        - Get analysis results")
    print("\n🔬 Advanced Analysis:")
    print("   GET  /api/tracks/{id}/features        - Detailed musical features")
    print("   GET  /api/tracks/{id}/segments        - Spotify-like segments with pitch/timbre")
    print("   GET  /api/tracks/{id}/beats           - Rhythmic analysis (bars, beats, tatums)")
    print("   GET  /api/tracks/{id}/sections        - Structural sections analysis")
    print("   POST /api/tracks/compare              - Compare two tracks for compatibility")
    print("   POST /api/tracks/batch-analyze        - Batch analyze multiple tracks")
    print("   GET  /api/analysis/search             - Search tracks by musical characteristics")
    print("\n🎛️  Mix Generation:")
    print("   POST /api/mix/generate                - Generate AI mix script")
    print("   GET  /api/mix/scripts                 - List mix scripts")
    print("   GET  /api/mix/scripts/{id}            - Get specific mix script")
    print("   POST /api/create-mix-simple           - One-call mix creation")
    print("   POST /api/render-mix                  - Render mix script to audio")
    print("\n⚙️  Background Tasks:")
    print("   GET  /api/tasks                       - List active tasks")
    print("   GET  /api/tasks/{id}                  - Get task status")
    print("\n📁 File Serving:")
    print("   GET  /api/files/{filename}            - Serve uploaded/generated files")

def print_search_examples():
    """Print example search queries"""
    print("\n🔍 Search API Examples:")
    print("   /api/analysis/search?min_bpm=120&max_bpm=130")
    print("   /api/analysis/search?key=C&mode=major")
    print("   /api/analysis/search?camelot=8A")
    print("   /api/analysis/search?min_energy=0.7&has_section=chorus")
    print("   /api/analysis/search?mode=minor&has_section=breakdown")

def main():
    parser = argparse.ArgumentParser(
        description="Start the RAMZI DJ API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_api.py                    # Start with default settings
  python start_api.py --port 8000        # Start on different port
  python start_api.py --host 0.0.0.0     # Accept external connections
  python start_api.py --debug             # Enable debug mode
        """
    )
    
    parser.add_argument('--host', default=None,
                       help='Host to bind to (default: 127.0.0.1, 0.0.0.0 in production)')
    parser.add_argument('--port', type=int, default=None,
                       help='Port to bind to (default: 5000, $PORT in production)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (forced off in production)')
    parser.add_argument('--no-checks', action='store_true',
                       help='Skip dependency and environment checks')
    parser.add_argument('--show-endpoints', action='store_true',
                       help='Show all available API endpoints')
    
    args = parser.parse_args()
    
    print("🎧 RAMZI DJ API Launcher v2.0")
    print("=" * 50)
    
    if args.show_endpoints:
        print_api_endpoints()
        print_search_examples()
        print("\n📖 Full documentation: API_README.md")
        return

    if not args.no_checks:
        print("\n📦 Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        print("✅ All required packages are installed")
        
        print("\n🔧 Checking environment...")
        check_environment()
    
    print("📁 Setting up directories...")
    setup_directories()
    
    # --- Production vs. Development Configuration ---
    is_production = 'RENDER' in os.environ
    
    host = args.host or ('0.0.0.0' if is_production else '127.0.0.1')
    port = args.port or (int(os.environ.get('PORT', 5000)) if is_production else 5000)
    debug_mode = args.debug and not is_production

    print(f"\n🚀 Starting API server...")
    print(f"   Mode: {'Production' if is_production else 'Development'}")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Debug: {debug_mode}")
    print(f"   URL: http://{host}:{port}")
    
    # Import and run the Flask app
    try:
        # Add src to path so we can import the modules
        src_path = Path(__file__).parent / 'src'
        sys.path.insert(0, str(src_path))
        
        from api import app
        
        print("\n✅ API server starting...")
        print_api_endpoints()
        print("\n📖 Documentation:")
        print("   API Reference: API_README.md")
        print("   Search Examples: Use --show-endpoints flag")
        
        if is_production:
            from waitress import serve
            print("\n🏭 Running with Waitress production server.")
            serve(app, host=host, port=port)
        else:
            print("\n🔧 Running with Flask development server.")
            app.run(
                host=host,
                port=port,
                debug=debug_mode
            )
        
    except ImportError as e:
        print(f"\n❌ Failed to import API modules: {e}")
        print("Make sure you're running from the project root directory")
        print("pip install flask flask-cors librosa openai numpy scipy")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 