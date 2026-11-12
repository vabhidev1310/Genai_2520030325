
from datetime import datetime, timezone
from pymongo import MongoClient


class MongoLogger:
    def __init__(self, connection_string="mongodb://localhost:27017/"):
        self.client = MongoClient(
            connection_string,
            serverSelectionTimeoutMS=5000
        )

        self.db = self.client["llm_monitoring"]
        self.collection = self.db["call_logs"]

        # Verify that MongoDB is reachable.
        self.client.admin.command("ping")
        print("Connected to MongoDB successfully.")

    def log_call(self, log_data):
        log_data["timestamp"] = datetime.now(timezone.utc)
        self.collection.insert_one(log_data)
        print("LLM call logged successfully.")

    def close(self):
        self.client.close()
        print("MongoDB connection closed.")
