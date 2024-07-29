import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline

def main():
    run_pipeline()

if __name__ == "__main__":
    main()