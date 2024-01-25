from types import FunctionType
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from typing import List

from pytz import utc

scheduler = BackgroundScheduler()
def startSchedule():
    jobstores = {
        'default': MemoryJobStore()
    }
    executors = {
        'default': AsyncIOExecutor()
    }
    job_defaults = {
        'coalesce': False,
        'max_instances': 3
    }
    scheduler = AsyncIOScheduler(jobstores=jobstores, executors=executors, job_defaults=job_defaults, timezone=utc)
    scheduler.start()

def setSchedule(times: List[str], function: FunctionType):
    for job in scheduler.get_jobs():
        print(f'removing job {job}')
        job.remove()
    for time in times:
        print(f'adding schedule {time}')
        hour_min = time.split(":")
        job = scheduler.add_job(function, 'cron', hour=int(hour_min[0]), minute=int(hour_min[1]))