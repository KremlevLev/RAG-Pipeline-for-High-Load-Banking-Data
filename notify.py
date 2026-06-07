import os
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Notifier")

@mcp.tool()
def send_telegram_notification(message: str) -> str:
    """
    Отправляет уведомление в Telegram пользователю.
    """
    # Теперь скрипт берет токены из вашего JSON конфига (блок env)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        return "Ошибка: Токен или Chat ID не найдены в переменных окружения."

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": f"🤖 Cline Alert:\n\n{message}",
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return "Уведомление успешно отправлено в Telegram."
        return f"Ошибка Telegram API: {response.text}"
    except Exception as e:
        return f"Не удалось отправить уведомление: {str(e)}"

if __name__ == "__main__":
    mcp.run()