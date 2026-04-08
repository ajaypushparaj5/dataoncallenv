"""DataOnCallEnv Client.

Provides a client for connecting to a DataOnCallEnv server.
DataOnCallEnvClient extends EnvClient to interact with the environment
via HTTP endpoints (reset, step, state).
"""
from openenv.core.client import EnvClient


class DataOnCallEnvClient(EnvClient):
    """
    Client for the DataOnCallEnv environment.

    Example:
        >>> with DataOnCallEnvClient(base_url="http://localhost:7860") as env:
        ...     obs = env.reset(task_id=1)
        ...     result = env.step({"tool": "list_tables", "query": ""})
    """

    pass  # EnvClient provides all needed functionality
