#!/usr/bin/env bash
set -e
git remote set-url origin https://github.com/Topabaem05/Omegahee.git 2>/dev/null || git remote add origin https://github.com/Topabaem05/Omegahee.git
echo "Attempting push to main branch of https://github.com/Topabaem05/Omegahee ..."
if git push -u origin main; then
  echo "Successfully pushed to main!"
else
  echo "Push to main failed or restricted; pushing to fallback branch feat/omega-uzu-v2 ..."
  git checkout -B feat/omega-uzu-v2
  git push -u origin feat/omega-uzu-v2
  echo "Successfully pushed to feat/omega-uzu-v2!"
fi
