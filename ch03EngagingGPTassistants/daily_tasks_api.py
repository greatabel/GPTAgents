from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# uvicorn daily_tasks_api:app --reload --port 5001

# http://127.0.0.1:5001/tasks
# http://127.0.0.1:5001/docs

app = FastAPI()

# app = FastAPI(
#     docs_url=None,          # 关闭 Swagger UI
#     redoc_url=None,         # 关闭 ReDoc
#     openapi_url=None        # 不提供 /openapi.json
# )

# Define a Pydantic model for a task
class Task(BaseModel):
    id: int
    description: str
    completed: bool

# Mock data: List of tasks
tasks = [
    Task(id=1, description="Buy groceries", completed=False),
    Task(id=2, description="Read a book", completed=True),
    Task(id=3, description="Complete FastAPI project", completed=False),
]

@app.get("/tasks", response_model=List[Task])
async def get_tasks():
    """
    Retrieve a list of daily tasks.
    """
    return tasks
