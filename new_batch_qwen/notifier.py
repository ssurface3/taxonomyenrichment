import requests
import traceback

BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
CHAT_ID = "YOUR_CHAT_ID_HERE"

def send_msg(text):
    if not BOT_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        data = {"chat_id": CHAT_ID, "text": text}
        requests.post(url, data=data)
    except:
        pass 

class TelegramLogger:
    def __enter__(self):
        send_msg("Script Started: Prediction Loop")
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type:
            error_msg = "".join(traceback.format_exception(exc_type, exc_value, tb))
            
            send_msg(f"CRITICAL ERROR:\n\n{error_msg[-1000:]}")
        else:
            send_msg("Script Finished Successfully!")