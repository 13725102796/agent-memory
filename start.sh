#!/bin/bash
PORT=8899
cd "$(dirname "$0")"

# 杀掉占用端口的旧进程
PID=$(lsof -ti :$PORT 2>/dev/null)
if [ -n "$PID" ]; then
    echo "端口 $PORT 被占用 (PID: $PID)，正在关闭..."
    kill $PID 2>/dev/null
    sleep 1
fi

echo "启动聊天服务 http://localhost:$PORT"
python -m memory_agent.web.server
