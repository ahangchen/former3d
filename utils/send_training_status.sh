#!/bin/bash
# 发送训练状态到QQ Bot

STATUS_FILE="/tmp/training_status.txt"

if [ -f "$STATUS_FILE" ]; then
    # 读取状态消息
    MESSAGE=$(cat "$STATUS_FILE")

    # 使用OpenClaw的message工具发送
    # 注意：这需要在OpenClaw环境中执行
    # 当前会话conversation_label是 qqbot:c2c:C95F43638870FC087DDB8AC7664CA602

    # 直接输出消息，OpenClaw会处理发送
    echo "$MESSAGE"
else
    echo "⚠️ 未找到训练状态文件"
fi
