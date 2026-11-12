# Code

`src/` contains the experiment scripts and `M1/` contains the earlier prototype.

From this Code directory, run the following in PowerShell:

    Set-Location .\src
    Copy-Item ..\..\Results\data_out . -Recurse
    python run_all.py --dry-run

Use a fresh working copy for the copy command. Install dependencies from
`requirements-m4.txt` if available, otherwise consult `requirements.txt`.
Open `M4_Verification.ipynb` if present to check the archived M4 results.
