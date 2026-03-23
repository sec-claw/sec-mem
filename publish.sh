#!/bin/bash
set -e

echo "🚀 Publishing @sec-claw/sec-mem..."

# 检查是否登录
echo "🔍 检查 npm 登录状态..."
npm whoami || (echo "❌ 请先运行: npm login" && exit 1)

# 安装依赖
echo "📦 安装依赖..."
npm install

# 构建
echo "🔨 构建 TypeScript..."
npm run build

# 测试 Python 后端
echo "🐍 检查 Python 依赖..."
python3 -c "import faiss; import numpy; import pydantic; print('✓ Python 依赖 OK')" || (echo "⚠️ 警告: Python 依赖可能未安装" && echo "  运行: pip install faiss-cpu numpy pydantic")

# 检查将发布的文件
echo "📋 检查包内容..."
npm pack --dry-run | head -20

echo ""
echo "📦 包信息:"
echo "  名称: @sec-claw/sec-mem"
echo "  版本: $(node -p "require('./package.json').version")"
echo ""

# 确认
read -p "🚀 确认发布到 npm? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    npm publish --access public
    echo ""
    echo "✅ 发布成功!"
    echo ""
    echo "📎 链接:"
    echo "  https://www.npmjs.com/package/@sec-claw/sec-mem"
    echo ""
    echo "📥 安装命令:"
    echo "  npm install @sec-claw/sec-mem"
else
    echo "❌ 取消发布"
    exit 1
fi
