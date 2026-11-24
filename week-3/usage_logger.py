
import time
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv
import os

load_dotenv(override=True)


class UsageLogger:
    def __init__(self):
        uri = os.getenv("MONGODB_URI")

        self.client = MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
            retryWrites=True
        )

        # Check MongoDB connection
        self.client.admin.command("ping")

        self.db = self.client["llm_monitoring"]
        self.collection = self.db["api_usage"]

        print("MongoDB connected successfully!")

    def log_call(self, record):
        record["timestamp"] = datetime.now(timezone.utc)

        for attempt in range(3):
            try:
                self.collection.insert_one(record)
                print("Usage logged successfully!")
                return True

            except PyMongoError as error:
                print(f"Logging attempt {attempt + 1} failed: {error}")

                if attempt < 2:
                    time.sleep(2 ** attempt)

        print("Could not log usage after 3 attempts.")
        return False

    def close(self):
        self.client.close()
        print("MongoDB connection closed.")
