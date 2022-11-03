import os
from typing import Tuple
from llm_orchestration import *
from nebula3_database.movie.movie_db import MOVIE_DB
from experts.pipeline.api import PipelineApi, PipelineTask

def test_pipeline_task(pipeline_id):
    class LlmTask(PipelineTask):
        def __init__(self):
            self.llm_task = LlmTaskInternal()
            print("LlmTask Initialized successfully.")

        def process_movie(self, movie_id: str) -> Tuple[bool, str]:
            print (f'LlmTask: handling movie: {movie_id}')

            output = self.llm_task.process_movie(movie_id)

            print("LlmTask: Finished handling movie.")
            print(output)
            return output
        def get_name(self) -> str:
            return "llm"

    pipeline = PipelineApi(None)
    task = LlmTask()
    pipeline.handle_pipeline_task(task, pipeline_id, stop_on_failure=True)


def test():
    pipeline_id = os.environ.get('PIPELINE_ID')
    # print(pipeline_id)
    # pipeline_id='12345678'
    test_pipeline_task(pipeline_id)

if __name__ == '__main__':
    test()