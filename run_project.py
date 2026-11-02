"""Run the complete, sequential project from the ten original CSV files."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
REQUIRED=[
    'application_train.csv','application_test.csv','bureau.csv','bureau_balance.csv',
    'previous_application.csv','installments_payments.csv','credit_card_balance.csv',
    'POS_CASH_balance.csv','HomeCredit_columns_description.csv','sample_submission.csv',
]

def main():
    missing=[name for name in REQUIRED if not (ROOT/'data'/name).is_file()]
    if missing:
        raise SystemExit('Missing files in data/: '+', '.join(missing))
    stages=[('main.py',['--folds','3']),('enhanced.py',[]),('enhanced.py',['--extra']),('finalize_report.py',[]),('verify_outputs.py',[])]
    for name,args in stages:
        print('Running:',name,*args,flush=True)
        subprocess.run([sys.executable,str(ROOT/name),*args],cwd=ROOT,check=True)
    print('Complete. Read RESULTS_ZH.md; predictions: output/submission.csv',flush=True)

if __name__=='__main__':main()
