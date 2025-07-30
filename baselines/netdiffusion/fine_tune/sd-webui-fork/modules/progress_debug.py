import base64
import io
import time
import logging

import gradio as gr
from pydantic import BaseModel, Field

from modules.shared import opts

import modules.shared as shared


current_task = None
pending_tasks = {}
finished_tasks = []
recorded_results = []
recorded_results_limit = 2


def start_task(id_task):
    global current_task

    current_task = id_task
    pending_tasks.pop(id_task, None)


def finish_task(id_task):
    global current_task

    if current_task == id_task:
        current_task = None

    finished_tasks.append(id_task)
    if len(finished_tasks) > 16:
        finished_tasks.pop(0)


def record_results(id_task, res):
    recorded_results.append((id_task, res))
    if len(recorded_results) > recorded_results_limit:
        recorded_results.pop(0)


def add_task_to_queue(id_job):
    pending_tasks[id_job] = time.time()


class ProgressRequest(BaseModel):
    id_task: str = Field(default=None, title="Task ID", description="id of the task to get progress for")
    id_live_preview: int = Field(default=-1, title="Live preview image ID", description="id of last received last preview image")
    live_preview: bool = Field(default=True, title="Include live preview", description="boolean flag indicating whether to include the live preview image")


class ProgressResponse(BaseModel):
    active: bool = Field(title="Whether the task is being worked on right now")
    queued: bool = Field(title="Whether the task is in queue")
    completed: bool = Field(title="Whether the task has already finished")
    progress: float = Field(default=None, title="Progress", description="The progress with a range of 0 to 1")
    eta: float = Field(default=None, title="ETA in secs")
    live_preview: str = Field(default=None, title="Live preview image", description="Current live preview; a data: uri")
    id_live_preview: int = Field(default=None, title="Live preview image ID", description="Send this together with next request to prevent receiving same image")
    textinfo: str = Field(default=None, title="Info text", description="Info text used by WebUI.")


def setup_progress_api(app):
    return app.add_api_route("/internal/progress", progressapi, methods=["POST"], response_model=ProgressResponse)


# Configure logging
log_file_path = '/home/rachel/Loras/lora1/preprocessed_data/progressapi_errors.txt'


def log_error(message):
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')

# Initial log entry to verify logging works
log_error("Logging initialized successfully.")

def progressapi(req: ProgressRequest):
    log_error(f"ProgressRequest received: {req}")
    
    active = req.id_task == current_task
    queued = req.id_task in pending_tasks
    completed = req.id_task in finished_tasks
    
    log_error(f"Task status - Active: {active}, Queued: {queued}, Completed: {completed}")

    if not active:
        textinfo = "Waiting..."
        if queued:
            sorted_queued = sorted(pending_tasks.keys(), key=lambda x: pending_tasks[x])
            queue_index = sorted_queued.index(req.id_task)
            textinfo = "In queue: {}/{}".format(queue_index + 1, len(sorted_queued))
            log_error(f"Queue position: {textinfo}")
        return ProgressResponse(active=active, queued=queued, completed=completed, id_live_preview=-1, textinfo=textinfo)

    progress = 0

    job_count, job_no = shared.state.job_count, shared.state.job_no
    sampling_steps, sampling_step = shared.state.sampling_steps, shared.state.sampling_step

    log_error(f"Job count: {job_count}, Job no: {job_no}")
    log_error(f"Sampling steps: {sampling_steps}, Sampling step: {sampling_step}")

    if job_count > 0:
        progress += job_no / job_count
    if sampling_steps > 0 and job_count > 0:
        progress += 1 / job_count * sampling_step / sampling_steps

    progress = min(progress, 1)
    log_error(f"Progress: {progress}")

    elapsed_since_start = time.time() - shared.state.time_start
    predicted_duration = elapsed_since_start / progress if progress > 0 else None
    eta = predicted_duration - elapsed_since_start if predicted_duration is not None else None

    log_error(f"Elapsed time: {elapsed_since_start}, Predicted duration: {predicted_duration}, ETA: {eta}")

    live_preview = None
    id_live_preview = req.id_live_preview

    if opts.live_previews_enable and req.live_preview:
        shared.state.set_current_image()
        if shared.state.id_live_preview != req.id_live_preview:
            image = shared.state.current_image
            image_path = getattr(shared.state, 'current_image_path', 'Unknown')

            log_error(f"Current image: {image}, Path: {image_path}")

            if image is not None:
                buffered = io.BytesIO()
                save_kwargs = {}

                if opts.live_previews_image_format == "png":
                    if max(*image.size) <= 256:
                        save_kwargs = {"optimize": True}
                    else:
                        save_kwargs = {"optimize": False, "compress_level": 1}

                try:
                    # Ensure the image is loaded properly
                    if image.fp is None:
                        raise ValueError("Image file pointer is None, possibly due to a corrupt image or incorrect file path.")
                    
                    image.save(buffered, format=opts.live_previews_image_format, **save_kwargs)
                    base64_image = base64.b64encode(buffered.getvalue()).decode('ascii')
                    live_preview = f"data:image/{opts.live_previews_image_format};base64,{base64_image}"
                    id_live_preview = shared.state.id_live_preview
                    log_error("Live preview generated successfully.")
                except Exception as e:
                    error_message = f"Error saving image {image} from {image_path}: {e}"
                    log_error(error_message)
                    print(error_message)
            else:
                log_error("No current image available for live preview.")
    
    return ProgressResponse(
        active=active, 
        queued=queued, 
        completed=completed, 
        progress=progress, 
        eta=eta, 
        live_preview=live_preview, 
        id_live_preview=id_live_preview, 
        textinfo=shared.state.textinfo
    )

def restore_progress(id_task):
    while id_task == current_task or id_task in pending_tasks:
        time.sleep(0.1)

    res = next(iter([x[1] for x in recorded_results if id_task == x[0]]), None)
    if res is not None:
        return res

    return gr.update(), gr.update(), gr.update(), f"Couldn't restore progress for {id_task}: results either have been discarded or never were obtained"
