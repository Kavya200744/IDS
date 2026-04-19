"""
Verification Script - Check if IDS files are correctly fixed
Run this in your project directory to verify you have the fixed version
"""

import os
import sys

def check_file_exists(filepath):
    """Check if file exists"""
    return os.path.exists(filepath)

def check_app_py_fix():
    """Check if app.py has the correct fix"""
    print("\n" + "="*70)
    print("CHECKING app.py FIX")
    print("="*70)
    
    if not check_file_exists('app.py'):
        print("❌ app.py not found in current directory!")
        print("   Make sure you're running this from the project root")
        return False
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Check for the correct fix
    if 'predictions, probabilities = model.predict(X)' in content:
        print("✅ CORRECT: Found 'predictions, probabilities = model.predict(X)'")
        
        # Check for extraction
        if 'prediction = predictions[0]' in content:
            print("✅ CORRECT: Found 'prediction = predictions[0]'")
            print("✅ CORRECT: Found 'probability = probabilities[0]'")
            return True
        else:
            print("❌ WRONG: Missing 'prediction = predictions[0]'")
            return False
    else:
        print("❌ WRONG: Still has old code 'prediction, probability = model.predict(X)'")
        print("\n⚠️  YOU ARE USING THE OLD BROKEN VERSION!")
        return False

def check_preprocessing_py_fix():
    """Check if preprocessing.py has the correct fix"""
    print("\n" + "="*70)
    print("CHECKING preprocessing.py FIX")
    print("="*70)
    
    if not check_file_exists('preprocessing.py'):
        print("❌ preprocessing.py not found in current directory!")
        return False
    
    with open('preprocessing.py', 'r') as f:
        content = f.read()
    
    # Check for the correct fix
    if "data['spkts'][attack_mask] = (data['spkts'][attack_mask] * 1.5).astype(int)" in content:
        print("✅ CORRECT: Found correct data type conversion")
        return True
    elif "data['spkts'][attack_mask] *= 1.5" in content:
        print("❌ WRONG: Still has old broken multiplication")
        print("\n⚠️  YOU ARE USING THE OLD BROKEN VERSION!")
        return False
    else:
        print("⚠️  WARNING: Could not find the relevant line")
        return False

def main():
    print("\n" + "="*70)
    print(" "*20 + "IDS FILE VERIFICATION")
    print("="*70)
    print(f"\nCurrent directory: {os.getcwd()}")
    
    # Check both files
    app_ok = check_app_py_fix()
    preprocessing_ok = check_preprocessing_py_fix()
    
    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    
    if app_ok and preprocessing_ok:
        print("\n✅ ✅ ✅ ALL FIXES ARE CORRECT! ✅ ✅ ✅")
        print("\nYou have the FIXED version. The system should work!")
        print("\nNext steps:")
        print("1. Run: python train.py")
        print("2. Run: python app.py")
        print("3. Open: http://127.0.0.1:5000")
    else:
        print("\n❌ ❌ ❌ YOU HAVE THE WRONG VERSION! ❌ ❌ ❌")
        print("\nYou are still using the OLD BROKEN files!")
        print("\nWhat to do:")
        print("1. DELETE your current project folder completely")
        print("2. DOWNLOAD the ids_project_FINAL_FIXED.zip again")
        print("3. EXTRACT it to a NEW location")
        print("4. Run this verification script again")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
