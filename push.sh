#!/bin/bash

# --- CONFIGURATION ---
REPO_URL="https://github.com/ssurface3/taxonomyenrichment.git"
BRANCH="master"
# ---------------------

echo "🚀 Starting Safe Push Process..."

# 1. Setup .gitignore to prevent uploading junk/secrets
# We add connect.sh to gitignore to prevent the secret leak issue you had before
echo "Creating .gitignore to keep repo clean..."
echo ".virtual_documents" > .gitignore
echo "__pycache__" >> .gitignore
echo "*.tar.gz" >> .gitignore
echo "connect.sh" >> .gitignore
echo ".ipynb_checkpoints" >> .gitignore

# 2. Initialize Git (Safe to run even if already initialized)
if [ ! -d ".git" ]; then
    git init
    git branch -M $BRANCH
fi

# 3. Add the remote repository
# (It might say 'remote origin already exists', which is fine)
git remote add origin $REPO_URL 2>/dev/null || git remote set-url origin $REPO_URL

# 4. Stage all local files
echo "📦 Staging files..."
git add .

# 5. Commit local changes
# We check if there are changes to commit to avoid empty commit errors
if git diff-index --quiet HEAD --; then
    echo "No new changes to commit."
else
    git commit -m "Merging local Kaggle work with GitHub repo"
fi

# 6. THE CRITICAL STEP: Pull remote files without deleting local ones
# --allow-unrelated-histories tells Git: "I know these look like different projects, merge them anyway."
# --no-edit accepts the default merge message
echo "⬇️  Pulling existing GitHub files (Merging)..."
git pull origin $BRANCH --allow-unrelated-histories --no-edit

# 7. Push everything back up
echo "⬆️  Pushing to GitHub..."
git push --set-upstream origin $BRANCH

echo "✅ Done! Local files and GitHub files are now merged."