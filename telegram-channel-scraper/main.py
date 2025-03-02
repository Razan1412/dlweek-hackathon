# Imports
import os
import asyncio
from dotenv import load_dotenv
from TelegramChannelScraper import TelegramChannelScraper

# Load environment variables from .env
load_dotenv()

# Get API credentials from environment variables
api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")
phone = os.getenv("TELEGRAM_PHONE")

# Ensure credentials are loaded
if not all([api_id, api_hash, phone]):
    raise ValueError("Missing API credentials. Check your .env file.")

# Create a scraper object and pass in API credentials
scraper = TelegramChannelScraper(
    credentials={
        "api_id": api_id, 
        "api_hash": api_hash, 
        "phone": phone,
    }
)

# Option 1: Provide a channel and pull all messages
asyncio.run(
    scraper.get_messages(
        channel_name="news_finance",
    )
)

# Option 2: Provide a channel and pull from a specific message ID onwards
# asyncio.run(
#     scraper.get_messages(
#         channel_name="<channel name>",
#         start_message_id=<message ID>,
#     )
# )
