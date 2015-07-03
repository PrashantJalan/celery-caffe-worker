from __future__ import absolute_import
from celery import Celery

app = Celery('cloudcv',
             broker='amqp://guest:guest@172.17.0.185:5672//',
             backend='redis://172.17.0.187:6379',
             include=['executable.web_tasks'])

# Optional configuration, see the application user guide.
app.conf.update(
    CELERY_ENABLE_UTC=True,
    CELERY_TASK_RESULT_EXPIRES=3600,
)

if __name__ == '__main__':
    app.start()

