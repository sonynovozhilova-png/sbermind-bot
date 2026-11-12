@echo off
echo Запускаю нейросеть Ollama...
start /b docker compose up -d ollama

echo Запускаю сайт...
cd webserver
start /b ..\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 5000

echo Бот запущен! Не закрывайте это окно.
pause