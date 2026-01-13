from enum import Enum


class ToolType(str, Enum):
    CUSTOM = "custom"
    MMA_CORE = "mma_core"
    MMA_CODER_CORE = "mma_coder_core"
    MMA_MEMORY_CORE = "mma_memory_core"
    MMA_MULTI_AGENT_CORE = "mma_multi_agent_core"


class JobType(str, Enum):
    JOB = "job"
    RUN = "run"


class ToolSourceType(str, Enum):
    """Defines what a tool was derived from"""

    python = "python"
    json = "json"
