#!/bin/bash
#SBATCH --job-name=llm_run
#SBATCH --output=out/llm_run_%j.out
#SBATCH --error=out/llm_run_%j.err
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# הפעלת הסביבה הווירטואלית
source /home/dsi/baruchm9/LLMs/.venv/bin/activate

# מעבר לתיקיית הפרויקט
cd /home/dsi/baruchm9/LLMs/

# התקנת כל החבילות הדרושות
pip install -q -r requirements.txt

# אימון מחדש ואז הערכה
python finetune.py
python eval.py
