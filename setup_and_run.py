import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        'streamlit>=1.28.0',
        'pandas>=1.5.0', 
        'torch>=2.0.0',
        'transformers>=4.21.0',
        'plotly>=5.15.0',
        'requests>=2.28.0'
    ]
    
    print("🔍 Checking dependencies...")
    
    for package in required_packages:
        try:
            package_name = package.split('>=')[0].split('==')[0]
            __import__(package_name)
            print(f"✅ {package_name} - OK")
        except ImportError:
            print(f"❌ {package_name} - Installing...")
            install_package(package)
            print(f"✅ {package_name} - Installed")
    
    print("🎉 All dependencies ready!")

if __name__ == "__main__":
    check_and_install_dependencies()
    
    # Run streamlit app
    print("\n🚀 Starting Streamlit app...")
    os.system("streamlit run main.py")
