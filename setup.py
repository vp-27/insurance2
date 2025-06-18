#!/usr/bin/env python3
"""
Installation and setup script for Live Insurance Risk & Quote Co-Pilot
"""

import os
import subprocess
import sys
import json
from pathlib import Path

class SetupManager:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.venv_dir = self.project_dir / "venv"
        self.data_dir = self.project_dir / "live_data_feed"
        
    def print_step(self, step, description):
        print(f"\n{'='*60}")
        print(f"Step {step}: {description}")
        print('='*60)
    
    def run_command(self, command, description=""):
        """Run a shell command and handle errors"""
        print(f"🔄 {description}" if description else f"🔄 Running: {command}")
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout.strip())
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
            return False
    
    def check_python(self):
        """Check Python version"""
        self.print_step(1, "Checking Python Installation")
        
        if sys.version_info < (3, 8):
            print("❌ Python 3.8 or higher is required")
            print(f"Current version: {sys.version}")
            return False
        
        print(f"✅ Python {sys.version.split()[0]} detected")
        return True
    
    def create_virtual_environment(self):
        """Create and activate virtual environment"""
        self.print_step(2, "Setting Up Virtual Environment")
        
        if self.venv_dir.exists():
            print("📁 Virtual environment already exists")
            return True
        
        if not self.run_command(f"python3 -m venv {self.venv_dir}", "Creating virtual environment"):
            return False
        
        print("✅ Virtual environment created successfully")
        return True
    
    def install_dependencies(self):
        """Install Python dependencies"""
        self.print_step(3, "Installing Dependencies")
        
        # Determine pip command based on OS
        if os.name == 'nt':  # Windows
            pip_cmd = f"{self.venv_dir}/Scripts/pip"
        else:  # Unix/Linux/macOS
            pip_cmd = f"{self.venv_dir}/bin/pip"
        
        # Upgrade pip first
        if not self.run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
            return False
        
        # Install requirements
        requirements_file = self.project_dir / "requirements.txt"
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        if not self.run_command(f"{pip_cmd} install -r {requirements_file}", "Installing Python packages"):
            return False
        
        print("✅ Dependencies installed successfully")
        return True
    
    def setup_directories(self):
        """Create necessary directories"""
        self.print_step(4, "Setting Up Directories")
        
        # Create data directory
        self.data_dir.mkdir(exist_ok=True)
        print(f"📁 Data directory: {self.data_dir}")
        
        # Create sample data file
        sample_data = {
            "source": "setup",
            "timestamp": "2025-06-18T00:00:00Z",
            "location": "New York, NY",
            "content": "System initialization - no current alerts",
            "type": "system"
        }
        
        sample_file = self.data_dir / "initial_data.json"
        with open(sample_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print("✅ Directories and sample data created")
        return True
    
    def setup_environment_file(self):
        """Set up environment configuration"""
        self.print_step(5, "Environment Configuration")
        
        env_file = self.project_dir / ".env"
        
        if env_file.exists():
            print("📄 .env file already exists")
            return True
        
        env_content = """# API Keys (Optional - system works without them)
OPENROUTER_API_KEY=your_openrouter_api_key_here
NEWS_API_KEY=your_newsdata_io_api_key_here

# OpenRouter Configuration
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=mistralai/mixtral-8x7b-instruct

# Application Configuration
REFRESH_INTERVAL=30
WEATHER_FETCH_INTERVAL=60
NEWS_FETCH_INTERVAL=300

# Base Insurance Configuration
BASE_INSURANCE_COST=500
RISK_MULTIPLIER=0.1
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print("✅ Environment file created")
        print("💡 You can add API keys to .env for enhanced functionality")
        return True
    
    def test_installation(self):
        """Test the installation"""
        self.print_step(6, "Testing Installation")
        
        # Test imports
        test_script = """
import sys
sys.path.insert(0, '.')

try:
    # Test core imports
    from config import config
    from data_fetcher import DataFetcher
    from pipeline import LiveRAGPipeline
    print("✅ Core modules imported successfully")
    
    # Test data fetcher
    fetcher = DataFetcher()
    print("✅ Data fetcher initialized")
    
    # Test pipeline (basic init)
    pipeline = LiveRAGPipeline()
    print("✅ RAG pipeline initialized")
    
    print("🎉 Installation test passed!")
    
except Exception as e:
    print(f"❌ Installation test failed: {e}")
    sys.exit(1)
"""
        
        test_file = self.project_dir / "test_install.py"
        with open(test_file, 'w') as f:
            f.write(test_script)
        
        # Run test with virtual environment
        if os.name == 'nt':  # Windows
            python_cmd = f"{self.venv_dir}/Scripts/python"
        else:  # Unix/Linux/macOS
            python_cmd = f"{self.venv_dir}/bin/python"
        
        success = self.run_command(f"{python_cmd} test_install.py", "Running installation test")
        
        # Clean up test file
        test_file.unlink()
        
        return success
    
    def print_final_instructions(self):
        """Print final setup instructions"""
        print("\n" + "🎉" * 20)
        print("SETUP COMPLETE!")
        print("🎉" * 20)
        
        print("\n📋 Next Steps:")
        print("1. Activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print(f"   {self.venv_dir}\\Scripts\\activate")
        else:  # Unix/Linux/macOS
            print(f"   source {self.venv_dir}/bin/activate")
        
        print("\n2. Start the application:")
        print("   python app.py")
        
        print("\n3. Open your browser:")
        print("   http://localhost:8000")
        
        print("\n4. Optional - Add API keys to .env file:")
        print("   - OpenRouter API key for LLM functionality")
        print("   - NewsData.io API key for live news feeds")
        
        print("\n📚 Documentation:")
        print("   - README.md: Complete setup and usage guide")
        print("   - demo.py: Automated demo script")
        
        print("\n🛠 Troubleshooting:")
        print("   - Check requirements.txt for dependency issues")
        print("   - Ensure Python 3.8+ is installed")
        print("   - System works in simulation mode without API keys")
    
    def run_setup(self):
        """Run the complete setup process"""
        print("🏢 Live Insurance Risk & Quote Co-Pilot")
        print("Setting up your real-time insurance underwriting system...")
        
        steps = [
            self.check_python,
            self.create_virtual_environment,
            self.install_dependencies,
            self.setup_directories,
            self.setup_environment_file,
            self.test_installation
        ]
        
        for i, step in enumerate(steps, 1):
            if not step():
                print(f"\n❌ Setup failed at step {i}")
                return False
        
        self.print_final_instructions()
        return True

if __name__ == "__main__":
    setup = SetupManager()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick setup without prompts
        setup.run_setup()
    else:
        # Interactive setup
        print("🚀 Welcome to the Live Insurance Risk & Quote Co-Pilot setup!")
        print("\nThis will install and configure the system for you.")
        
        response = input("\nProceed with setup? (y/N): ").lower().strip()
        
        if response in ['y', 'yes']:
            setup.run_setup()
        else:
            print("Setup cancelled.")
