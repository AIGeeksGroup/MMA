import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MMABenchConfig:
    # --- API Settings ---
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    TARGET_API_KEY = os.getenv("TARGET_API_KEY")
    TARGET_BASE_URL = os.getenv("TARGET_BASE_URL", "https://api.openai.com/v1")

    JUDGE_API_KEY = os.getenv("JUDGE_API_KEY")
    JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1")
    
    # Models
    GENERATOR_MODEL = "qwen3-max"
    IMAGE_MODEL = "qwen-image-plus"
    EVALUATOR_MODEL = "qwen3-max"
    
    # --- Benchmark Structure ---
    NUM_CASES = 30
    SESSIONS_PER_CASE = 10
    EVENTS_PER_CASE = 3 # X, Y, Z
    
    # --- Output Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    IMAGE_DIR = os.path.join(DATA_DIR, "images")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    # --- Logic Matrix (Forced Distribution) ---
    # We will enforce this distribution across the 3 events in each case
    # Priority: Inversion > Ambiguity > Standard > Unknowable
    VISUAL_PRIORITY = {
        "Type_B_Inversion": 10,   # Highest priority for image injection
        "Type_C_Ambiguity": 8,
        "Type_A_Standard": 5,
        "Type_D_Unknowable": 0    # Usually no image or irrelevant
    }
    
    # --- Narrative Phases ---
    PHASES = {
        "Calibration": range(1, 5),   # S1-S4
        "Noise_Injection": range(5, 8), # S5-S7
        "The_Trap": range(8, 9),      # S8 (Visual Injection)
        "Conflict_Continuum": range(9, 11) # S9-S10 (No easy resolution)
    }

    @staticmethod
    def ensure_dirs():
        os.makedirs(MMABenchConfig.DATA_DIR, exist_ok=True)
        os.makedirs(MMABenchConfig.IMAGE_DIR, exist_ok=True)
        os.makedirs(MMABenchConfig.RESULTS_DIR, exist_ok=True)
