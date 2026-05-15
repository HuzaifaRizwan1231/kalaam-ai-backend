import asyncio, logging
from typing import Dict, Any

class ProgressService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProgressService, cls).__new__(cls)
            cls._instance.progress_data = {}
            cls._instance.queues = {}
        return cls._instance

    def update_progress(self, tracking_id: str, progress: float, stage: str):
        logging.info(f"Progress update for {tracking_id}: {progress}% ({stage})")
        self.progress_data[tracking_id] = {"progress": progress, "stage": stage}
        
        # Notify all active listeners for this tracking_id
        if tracking_id in self.queues:
            for queue in self.queues[tracking_id]:
                queue.put_nowait({"progress": progress, "stage": stage})

    async def subscribe(self, tracking_id: str):
        logging.info(f"New SSE subscription for tracking_id: {tracking_id}")
        if tracking_id not in self.queues:
            self.queues[tracking_id] = []
        
        queue = asyncio.Queue()
        self.queues[tracking_id].append(queue)
        
        try:
            # Yield initial progress if available
            if tracking_id in self.progress_data:
                yield self.progress_data[tracking_id]
                
            while True:
                data = await queue.get()
                yield data
        finally:
            logging.info(f"SSE subscription closed for tracking_id: {tracking_id}")
            self.queues[tracking_id].remove(queue)
            if not self.queues[tracking_id]:
                del self.queues[tracking_id]

    def remove_progress(self, tracking_id: str):
        logging.info(f"Removing progress data for {tracking_id}")
        if tracking_id in self.progress_data:
            del self.progress_data[tracking_id]
