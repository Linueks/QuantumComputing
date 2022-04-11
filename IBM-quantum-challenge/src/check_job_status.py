import qiskit as qk
from qiskit.providers.jobstatus import JobStatus

provider = qk.IBMQ.load_account()

provider = qk.IBMQ.get_provider(
    hub='ibm-q-community',
    group='ibmquantumawards',
    project='open-science-22'
)
jakarta_backend = provider.get_backend('ibmq_jakarta')

job_id = '624d6f6aa5d4eeee6477c33b'
job = jakarta_backend.retrieve_job(job_id)

try:
    job_status = job.status()  # Query the backend server for job status.

    if job_status is JobStatus.RUNNING:
        print("The job is still running")

    elif job_status is JobStatus.QUEUED:
        queue_position = job.queue_info().position
        estimated_complete_time = job.queue_info().estimated_complete_time

        print(f'The job is in queue with position: {queue_position}')
        print(f'with estimated complete time {estimated_complete_time}')


except IBMQJobApiError as ex:
    print("Something wrong happened!: {}".format(ex))
