# Data_Mining_HW1_G02

## Links

- Document: [Google Docs](https://docs.google.com/document/d/17EDonnW9b7wmKxxSu9W65ORkYWyFXTFzqIjIo54BFNc/edit?usp=sharing)
- Report: [Overleaf](https://www.overleaf.com/5219884191pcbkyktfpgxp#06cc55)

## How to Use

### 1. Download the Repository

Open a terminal in the folder where you want to save the project, then run:

```bash
git clone https://github.com/yfchinn/Data_Mining_HW1_G02.git
cd Data_Mining_HW1_G02
```

Cloning downloads the project and sets up its connection to GitHub. Run this step only once. For later sessions, open a terminal inside the cloned folder.

### 2. Set Your Commit Identity

Inside the cloned folder, run these commands once, replacing the examples with your own name and email:

```bash
git config user.name "Your Name"
git config user.email "your-email@example.com"
```

### 3. Get the Latest Changes

Before you start editing, make sure your previous work is committed, then run:

```bash
git switch main
git pull --rebase origin main
```

### 4. Commit and Push Your Changes

After editing, check which files changed:

```bash
git status
git diff
```

Stage the files you want to share, replacing `path/to/your-file` with an actual file or folder path. Use quotes around paths containing spaces:

```bash
git add "path/to/your-file"
git commit -m "Describe your changes"
git pull --rebase origin main
git push origin main
```

Repeat `git add` for additional files before committing. The final pull includes any updates your teammates pushed while you were working.
