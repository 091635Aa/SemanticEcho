# ════════════════════════════════════════════════════════
# 语义回响 (Semantic Echo) - GitHub 推送脚本
# ════════════════════════════════════════════════════════
# 用法：在 PowerShell 中运行此脚本
# 注意：请先设置环境变量 GITHUB_TOKEN，或在下面填入你的 token
# ════════════════════════════════════════════════════════

# 从环境变量读取 token（优先），或直接修改下面这行
$token = $env:GITHUB_TOKEN
if (-not $token) {
    Write-Host "请设置 GITHUB_TOKEN 环境变量后再运行此脚本" -ForegroundColor Red
    exit 1
}

$repoName = "SemanticEcho"
$repoDescription = "语义回响：通过回收被丢弃Token嵌入增强语言模型表达能力"

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  语义回响 (Semantic Echo) - GitHub 推送脚本" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# 步骤 1: 设置 Git 远程仓库
Write-Host "`n[1/3] 设置 Git 远程仓库..." -ForegroundColor Yellow
$userResult = Invoke-WebRequest -Uri "https://api.github.com/user" `
    -Headers @{ Authorization = "Bearer $token"; "User-Agent" = "PowerShell" } `
    -UseBasicParsing
$userInfo = $userResult.Content | ConvertFrom-Json
$fullName = "$($userInfo.login)/$repoName"
git remote remove origin 2>$null
git remote add origin "https://$token@github.com/$fullName.git"
Write-Host "  ✓ 远程仓库已设置: $fullName" -ForegroundColor Green

# 步骤 2: 暂存并提交文件
Write-Host "`n[2/3] 暂存并提交文件..." -ForegroundColor Yellow
git add -A
git reset -- .trae/ 2>$null

$commitMessage = "初始提交：语义回响 (Semantic Echo) 完整项目

- 论文 (LaTeX + PDF)
- 核心代码（回响池、采样处理器、评估器、实验运行器等）
- 实验数据（13组对照实验）
- 可视化图表"

git commit -m "$commitMessage" 2>$null
if ($LASTEXITCODE -ne 0) {
    git add -A
    git commit --allow-empty -m "$commitMessage"
}

# 步骤 3: 推送至 GitHub
Write-Host "`n[3/3] 推送至 GitHub..." -ForegroundColor Yellow
git push -u origin master:main 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n" + "=" * 60 -ForegroundColor Cyan
    Write-Host "  ✓ 推送成功！" -ForegroundColor Green
    Write-Host "  仓库地址: https://github.com/$fullName" -ForegroundColor Green
    Write-Host "=" * 60 -ForegroundColor Cyan
}
else {
    Write-Host "  ✗ 推送失败，请检查网络连接和 token 权限" -ForegroundColor Red
}
