param(
    [string]$CommitMessage = "Update Skein"
)

$ErrorActionPreference = 'Stop'

# Confirm that this script is being run from inside a Git repository.
$repoRoot = git rev-parse --show-toplevel 2>$null
if ($LASTEXITCODE -ne 0 -or -not $repoRoot) {
    Write-Host "ERROR: Run this script from inside your Skein Git repository." -ForegroundColor Red
    exit 1
}
Set-Location $repoRoot

$branch = git branch --show-current
if ($LASTEXITCODE -ne 0 -or -not $branch) {
    Write-Host "ERROR: Could not determine the current Git branch." -ForegroundColor Red
    exit 1
}

$remote = git remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0 -or -not $remote) {
    Write-Host "ERROR: Git remote 'origin' is not configured." -ForegroundColor Red
    exit 1
}

Write-Host "Repository: $repoRoot" -ForegroundColor Cyan
Write-Host "Branch:     $branch"
Write-Host "Remote:     $remote"
Write-Host ""

# Stage modifications/deletions to already tracked files only.
# New files must be added explicitly with `git add path\to\file` first.
git add -u
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: git add failed." -ForegroundColor Red
    exit 1
}

$hasStagedChanges = $false
git diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    $hasStagedChanges = $true
} elseif ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Could not inspect staged changes." -ForegroundColor Red
    exit 1
}

if (-not $hasStagedChanges) {
    Write-Host "No tracked-file changes to commit. Nothing was pushed." -ForegroundColor Yellow
    Write-Host "Untracked files are deliberately not auto-added. Add new files explicitly when you intend to publish them."
    exit 0
}

Write-Host "Changes to be published:" -ForegroundColor Cyan
git diff --cached --stat
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
git diff --cached --check
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Whitespace/conflict-marker check failed. Fix it before syncing." -ForegroundColor Red
    exit 1
}

$confirm = Read-Host "Commit and push these changes to origin/$branch? Type Y to continue"
if ($confirm -cne 'Y') {
    Write-Host "Cancelled. Changes remain staged locally; nothing was pushed." -ForegroundColor Yellow
    exit 0
}

if ([string]::IsNullOrWhiteSpace($CommitMessage)) {
    $CommitMessage = Read-Host "Enter a commit message"
}
if ([string]::IsNullOrWhiteSpace($CommitMessage)) {
    Write-Host "ERROR: Commit message cannot be empty." -ForegroundColor Red
    exit 1
}

git commit -m $CommitMessage
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Commit failed. Review the message above; nothing was pushed." -ForegroundColor Red
    exit 1
}

git push -u origin $branch
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Push failed. Your commit is saved locally; resolve the GitHub connection and run git push again." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Sync complete: committed and pushed to GitHub." -ForegroundColor Green
git status --short
