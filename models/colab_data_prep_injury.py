#!/usr/bin/env python3
"""
Enhanced Colab Data Preparation with Injury Features
====================================================

This script creates the injury-enhanced colab_data_with_injury.zip package
containing all necessary files for injury-aware MCTS in Google Colab.
"""

import os
import shutil
import zipfile
from pathlib import Path
import json

def create_injury_enhanced_colab_package():
    """Create comprehensive Colab package with injury features"""
    
    print("🏥 Creating Injury-Enhanced Colab Package")
    print("=" * 50)
    
    # Define source and target directories
    base_dir = Path(__file__).parent.parent
    colab_source_dir = base_dir / "colab_upload_injury"
    models_dir = base_dir / "models"
    
    # Create temporary staging directory
    staging_dir = base_dir / "colab_staging_injury"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()
    
    print(f"📁 Staging directory: {staging_dir}")
    
    # File mapping: source -> target_name
    files_to_include = {
        # Core data files
        "draft_board.csv": "draft_board.csv",
        "FantasyPros_2025_Overall_ADP_Rankings.csv": "FantasyPros_2025_Overall_ADP_Rankings.csv", 
        "rookie_data_clean.csv": "rookie_data_clean.csv",
        "model_weights/rookie_regressor.pkl": "rookie_regressor.pkl",
        
        # Enhanced injury data
        "injury_enhanced_demo.csv": "injury_enhanced_demo.csv",
        
        # Injury enhancement modules
        "models/mcts_injury_wrapper.py": "mcts_injury_wrapper.py",
        "models/injury_aware_mcts.py": "injury_aware_mcts.py", 
        "models/injury_enhancement.py": "injury_enhancement.py",
        "models/injury_aware_model.py": "injury_aware_model.py",
        "injury_demo.py": "injury_demo.py",
        
        # Colab-specific files
        "colab_upload_injury/metadata.json": "metadata.json",
        "colab_upload_injury/colab_loader.py": "colab_loader.py"
    }
    
    # Copy files to staging directory
    copied_files = []
    missing_files = []
    
    for source_path, target_name in files_to_include.items():
        source_file = base_dir / source_path
        target_file = staging_dir / target_name
        
        if source_file.exists():
            try:
                shutil.copy2(source_file, target_file)
                file_size = target_file.stat().st_size
                copied_files.append((target_name, file_size))
                print(f"✅ {target_name:30s} ({file_size:,} bytes)")
            except Exception as e:
                print(f"❌ Failed to copy {source_path}: {e}")
                missing_files.append(source_path)
        else:
            print(f"⚠️  Missing: {source_path}")
            missing_files.append(source_path)
    
    print(f"\n📊 Copy Summary:")
    print(f"   ✅ Copied: {len(copied_files)} files")
    print(f"   ❌ Missing: {len(missing_files)} files")
    
    if missing_files:
        print(f"\n⚠️  Missing files:")
        for file in missing_files:
            print(f"   • {file}")
    
    # Create the ZIP package
    zip_path = base_dir / "colab_data_with_injury.zip"
    
    print(f"\n📦 Creating ZIP package: {zip_path.name}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_name, file_size in copied_files:
            file_path = staging_dir / file_name
            zipf.write(file_path, file_name)
            print(f"   📄 Added: {file_name}")
    
    # Calculate package size
    package_size = zip_path.stat().st_size
    
    # Clean up staging directory
    shutil.rmtree(staging_dir)
    
    # Verify package contents
    print(f"\n🔍 Verifying package contents...")
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        package_files = zipf.namelist()
        print(f"   📋 Total files in package: {len(package_files)}")
        
        # Check for key files
        key_files = ['metadata.json', 'colab_loader.py', 'mcts_injury_wrapper.py', 'injury_enhanced_demo.csv']
        for key_file in key_files:
            if key_file in package_files:
                print(f"   ✅ {key_file}")
            else:
                print(f"   ❌ Missing: {key_file}")
    
    print(f"\n✅ Injury-Enhanced Colab Package Created!")
    print(f"   📦 Package: {zip_path.name}")
    print(f"   📏 Size: {package_size:,} bytes ({package_size/1024/1024:.2f} MB)")
    print(f"   📁 Files: {len(copied_files)}")
    
    # Show usage instructions
    print(f"\n🚀 Usage Instructions:")
    print(f"   1. Upload {zip_path.name} to Google Colab")
    print(f"   2. Extract with: !unzip {zip_path.name}")
    print(f"   3. Load data: from colab_loader import load_colab_data; data = load_colab_data()")
    print(f"   4. Choose integration method (see options in colab_loader.py)")
    print(f"   5. Run demos: !python injury_demo.py")
    
    return zip_path, copied_files, missing_files

def show_package_comparison():
    """Compare original vs injury-enhanced packages"""
    
    base_dir = Path(__file__).parent.parent
    
    # Check if both packages exist
    original_zip = base_dir / "colab_data.zip"
    injury_zip = base_dir / "colab_data_with_injury.zip"
    
    print(f"\n📊 Package Comparison:")
    print("=" * 30)
    
    if original_zip.exists():
        original_size = original_zip.stat().st_size
        with zipfile.ZipFile(original_zip, 'r') as zipf:
            original_files = len(zipf.namelist())
        print(f"📦 Original Package:")
        print(f"   📁 Files: {original_files}")
        print(f"   📏 Size: {original_size:,} bytes ({original_size/1024/1024:.2f} MB)")
    else:
        print(f"❌ Original package not found")
        original_size = 0
        original_files = 0
    
    if injury_zip.exists():
        injury_size = injury_zip.stat().st_size
        with zipfile.ZipFile(injury_zip, 'r') as zipf:
            injury_files = len(zipf.namelist())
        print(f"🏥 Injury-Enhanced Package:")
        print(f"   📁 Files: {injury_files}")
        print(f"   📏 Size: {injury_size:,} bytes ({injury_size/1024/1024:.2f} MB)")
        
        if original_size > 0:
            size_increase = injury_size - original_size
            file_increase = injury_files - original_files
            print(f"\n📈 Enhancement Impact:")
            print(f"   📁 Additional files: +{file_increase}")
            print(f"   📏 Size increase: +{size_increase:,} bytes (+{size_increase/1024/1024:.2f} MB)")
            print(f"   📊 Size increase: +{(size_increase/original_size)*100:.1f}%")
    else:
        print(f"❌ Injury-enhanced package not found")

def verify_injury_package():
    """Verify the injury-enhanced package works correctly"""
    
    print(f"\n🧪 Testing Injury-Enhanced Package:")
    print("=" * 40)
    
    base_dir = Path(__file__).parent.parent
    zip_path = base_dir / "colab_data_with_injury.zip"
    
    if not zip_path.exists():
        print(f"❌ Package not found: {zip_path}")
        return False
    
    # Extract to temporary directory for testing
    test_dir = base_dir / "test_injury_package"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    try:
        # Extract package
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(test_dir)
        
        print(f"✅ Package extracted successfully")
        
        # Test key files exist and are valid
        tests = [
            ("metadata.json", "JSON metadata"),
            ("colab_loader.py", "Python loader"),
            ("mcts_injury_wrapper.py", "Injury wrapper"),
            ("injury_enhanced_demo.csv", "Enhanced data"),
            ("draft_board.csv", "Draft board")
        ]
        
        all_tests_passed = True
        for filename, description in tests:
            file_path = test_dir / filename
            if file_path.exists():
                if filename.endswith('.json'):
                    try:
                        with open(file_path, 'r') as f:
                            json.load(f)
                        print(f"✅ {description}: Valid JSON")
                    except json.JSONDecodeError:
                        print(f"❌ {description}: Invalid JSON")
                        all_tests_passed = False
                elif filename.endswith('.py'):
                    # Basic Python syntax check
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        compile(content, filename, 'exec')
                        print(f"✅ {description}: Valid Python")
                    except SyntaxError:
                        print(f"❌ {description}: Syntax error")
                        all_tests_passed = False
                elif filename.endswith('.csv'):
                    try:
                        import pandas as pd
                        df = pd.read_csv(file_path)
                        print(f"✅ {description}: Valid CSV ({len(df)} rows)")
                    except Exception:
                        print(f"❌ {description}: Invalid CSV")
                        all_tests_passed = False
                else:
                    print(f"✅ {description}: File exists")
            else:
                print(f"❌ {description}: Missing file")
                all_tests_passed = False
        
        # Clean up test directory
        shutil.rmtree(test_dir)
        
        if all_tests_passed:
            print(f"\n🎉 All tests passed! Package is ready for use.")
        else:
            print(f"\n⚠️  Some tests failed. Please check the package.")
        
        return all_tests_passed
        
    except Exception as e:
        print(f"❌ Package testing failed: {e}")
        if test_dir.exists():
            shutil.rmtree(test_dir)
        return False

if __name__ == "__main__":
    # Create the injury-enhanced package
    zip_path, copied_files, missing_files = create_injury_enhanced_colab_package()
    
    # Show comparison with original package
    show_package_comparison()
    
    # Verify the package works
    verify_injury_package()
    
    print(f"\n🎯 Summary:")
    print(f"   ✅ Created: colab_data_with_injury.zip")
    print(f"   📁 Contains: {len(copied_files)} files")
    print(f"   🏥 Features: Full injury-aware MCTS")
    print(f"   🚀 Ready: Upload to Google Colab and start drafting!")
    
    if missing_files:
        print(f"   ⚠️  Note: {len(missing_files)} files were missing during creation")
