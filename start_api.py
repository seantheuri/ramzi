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
    required_packages = ['flask', 'flask_cors', 'librosa', 'openai']
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
    directories = ['uploads', 'analysis', 'mix_scripts']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directory ready: {directory}/")

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
    
    parser.add_argument('--host', default='127.0.0.1',
                       help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--no-checks', action='store_true',
                       help='Skip dependency and environment checks')
    
    args = parser.parse_args()
    
    print("🎧 RAMZI DJ API Launcher")
    print("=" * 50)
    
    if not args.no_checks:
        print("\n📦 Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        print("✅ All required packages are installed")
        
        print("\n🔧 Checking environment...")
        check_environment()
    
    print("📁 Setting up directories...")
    setup_directories()
    
    print(f"\n🚀 Starting API server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Debug: {args.debug}")
    print(f"   URL: http://{args.host}:{args.port}")
    
    # Import and run the Flask app
    try:
        # Add src to path so we can import the modules
        src_path = Path(__file__).parent / 'src'
        sys.path.insert(0, str(src_path))
        
        from api import app
        
        print("\n✅ API server starting...")
        print("\nEndpoints available:")
        print("   GET  /api/health              - Health check")
        print("   GET  /api/tracks              - List tracks")
        print("   POST /api/tracks/upload       - Upload track")
        print("   POST /api/tracks/{id}/analyze - Analyze track")
        print("   POST /api/mix/generate        - Generate mix")
        print("   GET  /api/config              - API configuration")
        print("\n📖 Full API documentation: API_README.md")
        print("\n" + "=" * 50)
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        
    except ImportError as e:
        print(f"\n❌ Failed to import API modules: {e}")
        print("Make sure you're running from the project root directory")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 