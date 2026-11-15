"""
Asynchronous Distributed Job Queue and Worker Manager.

Processes bulk customer feedback batches concurrently with:
- Semaphore-based concurrency throttling
- Non-blocking background execution
- Real-time progress updates in the database
"""
import asyncio
from typing import List, Optional
from datetime import datetime, timezone

from core.config import settings, logger
from models.schemas import AnalysisResult, JobStatus
from services.analyzer import analyzer
from services.database import DatabaseRepository


class JobQueueManager:
    """Manages asynchronous batch processing workers."""

    def __init__(self, max_concurrency: int = settings.MAX_CONCURRENT_WORKERS):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.active_tasks = {}

    async def _process_single_item(self, review_text: str, job_id: str) -> Optional[AnalysisResult]:
        async with self.semaphore:
            try:
                # Run CPU/network analysis asynchronously
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, analyzer.analyze_single, review_text)
                
                # Persist to database
                await DatabaseRepository.save_review(result, job_id=job_id)
                await DatabaseRepository.update_job_progress(job_id, completed_inc=1)
                return result
            except Exception as e:
                logger.error(f"Error analyzing review for job {job_id}: {e}")
                await DatabaseRepository.update_job_progress(job_id, completed_inc=0, failed_inc=1)
                return None

    async def _execute_batch_job(self, job_id: str, reviews: List[str]):
        logger.info(f"Starting execution of batch Job {job_id} ({len(reviews)} reviews)")
        await DatabaseRepository.update_job_progress(job_id, completed_inc=0, status=JobStatus.PROCESSING.value)

        tasks = [self._process_single_item(rev, job_id) for rev in reviews]
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Finished batch Job {job_id}")

    def enqueue_job(self, job_id: str, reviews: List[str]):
        """Dispatches batch processing in background without blocking caller."""
        task = asyncio.create_task(self._execute_batch_job(job_id, reviews))
        self.active_tasks[job_id] = task

        def _cleanup(fut):
            self.active_tasks.pop(job_id, None)

        task.add_done_callback(_cleanup)


job_manager = JobQueueManager()

