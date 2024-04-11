from types import FunctionType
from typing import Callable
import inspect
import logging
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from typing import List
from pytz import utc

#logging.basicConfig()
#logging.getLogger('apscheduler').setLevel(logging.DEBUG)

# def startSchedule():
#     jobstores = {
#         'default': MemoryJobStore()
#     }
#     executors = {
#         'default': AsyncIOExecutor()
#     }
#     job_defaults = {
#         'coalesce': False,
#         'max_instances': 3
#     }
#     scheduler = AsyncIOScheduler(jobstores=jobstores, executors=executors, job_defaults=job_defaults, timezone=utc)
#     scheduler.start()
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


def setSchedule(times: List[str], function: Callable):
    getNextTime(times)
    for job in scheduler.get_jobs():
        print(f'removing job {job}')
        job.remove()
    for time in times:
        print(f'adding schedule {time}')
        hour_min = time.split(":")
        job = scheduler.add_job(function, 'cron', hour=int(hour_min[0]), minute=int(hour_min[1]))




def getNextTime(times: List[str]) -> (int, datetime.datetime):
    dts = []
    now = datetime.datetime.utcnow()
    print(f"UTX TIME {now}")
    for time in times:
        hour_min = time.split(":")
        c = datetime.datetime(now.year, now.month, now.day, hour=int(hour_min[0]), minute=int(hour_min[1]))
        cts = c.timestamp()
        if cts - now.timestamp() < 0:
            c = c + datetime.timedelta(days=1)
        dts.append(c)
        print(c)
        
    # get all differences with date as values 
    cloz_dict = { 
      abs(now.timestamp() - date.timestamp()) : date 
      for date in dts}
     
    # extracting minimum key using min()
    res = cloz_dict[min(cloz_dict.keys())]
     
    # printing result
    print("Nearest date from list : " + str(res))
    print("Seconds " + str(res.timestamp()-now.timestamp()))
    seconds = res.timestamp() - now.timestamp()
    return int(seconds), res

