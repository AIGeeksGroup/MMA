import os
import json
import time
import base64
import mimetypes
import requests
from openai import OpenAI
from .config import MMABenchConfig

class MMABenchClient:
    def __init__(self):
        """
        Initializes clients for the Generator (Dataset Creation), Target Agent (Evaluation Candidate), 
        and Judge (Automated Scorer).
        """
        # 1. Generator Client (e.g., High-performance LLM for creating cases)
        self.gen_client = OpenAI(
            api_key=MMABenchConfig.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        ) if MMABenchConfig.DASHSCOPE_API_KEY else None

        # 2. Target Agent Client (The model being evaluated)
        # Prioritizes TARGET_API_KEY, falls back to OPENAI_API_KEY or DASHSCOPE_API_KEY.
        
        target_api_key = os.getenv("TARGET_API_KEY", os.getenv("OPENAI_API_KEY"))
        target_base_url = os.getenv("TARGET_BASE_URL", "https://api.openai.com/v1")
        
        if not target_api_key and MMABenchConfig.DASHSCOPE_API_KEY:
            print("[INFO] TARGET_API_KEY not found. Falling back to DASHSCOPE_API_KEY for Target Client.")
            target_api_key = MMABenchConfig.DASHSCOPE_API_KEY
            if target_base_url == "https://api.openai.com/v1":
                target_base_url = MMABenchConfig.DASHSCOPE_BASE_URL
                print(f"[INFO] Switched Target Base URL to {target_base_url}")

        self.target_client = OpenAI(
            api_key=target_api_key,
            base_url=target_base_url
        )

        # 3. Judge Client (The model grading the reasoning)
        # Fallback to TARGET_API_KEY if JUDGE_API_KEY is not set
        judge_api_key = MMABenchConfig.JUDGE_API_KEY
        if not judge_api_key:
             judge_api_key = target_api_key
             print("[INFO] JUDGE_API_KEY not found. Fallback to TARGET_API_KEY.")
             
        self.judge_client = OpenAI(
            api_key=judge_api_key,
            base_url=MMABenchConfig.JUDGE_BASE_URL
        ) if judge_api_key else None

    def _encode_image(self, image_path):
        """Encodes an image file to base64 string."""
        if not os.path.exists(image_path):
            return None
        
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/png" # Default
            
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        return f"data:{mime_type};base64,{encoded_string}"

    def generate_json(self, system_prompt, user_prompt, model=MMABenchConfig.GENERATOR_MODEL, temperature=0.7, client_type="generator", images=None):
        """
        Generates structured JSON.
        :param images: List of absolute file paths to images (for VLM support).
        """
        # Select client
        if client_type == "target":
            client = self.target_client
        elif client_type == "judge":
            client = self.judge_client
        else:
            client = self.gen_client
            # Generator usually uses the config model (qwen-max)
        
        if not client:
            print(f"Error: {client_type} client not configured.")
            return {}

        try:
            messages = [{"role": "system", "content": system_prompt}]
            
            # --- Qwen-VL Format Fix ---
            if "qwen" in model.lower() and images and len(images) > 0:
                # Qwen-VL specific format: content list inside user message
                user_content = []
                
                # 1. Add Images
                for img_path in images:
                    # Qwen-VL via OpenAI-compatible API often supports direct URLs or Base64
                    # Standard OpenAI Vision format:
                    base64_img = self._encode_image(img_path)
                    if base64_img:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": base64_img}
                        })
                    else:
                        print(f"Warning: Could not load image {img_path}")
                
                # 2. Add Text Prompt
                user_content.append({"type": "text", "text": user_prompt})
                
                messages.append({"role": "user", "content": user_content})

            elif images and len(images) > 0:
                # Standard OpenAI Vision Format (GPT-4o, etc.)
                user_content = [{"type": "text", "text": user_prompt}]
                
                for img_path in images:
                    base64_img = self._encode_image(img_path)
                    if base64_img:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": base64_img}
                        })
                    else:
                        print(f"Warning: Could not load image {img_path}")
                
                messages.append({"role": "user", "content": user_content})
            else:
                # Text-only
                messages.append({"role": "user", "content": user_prompt})

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            
            # --- Robust JSON Parsing ---
            # Qwen sometimes outputs incomplete JSON or comments.
            # Attempt to fix common issues or just parse.
            
            # 1. Strip whitespace
            content = content.strip()
            
            # 2. Extract JSON block if wrapped in markdown
            if "```" in content:
                import re
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
                if match:
                    content = match.group(1)
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # 3. Try ast.literal_eval for Python-style dicts (single quotes)
                try:
                    import ast
                    # Only safe if content looks like a dict
                    if content.strip().startswith("{") and content.strip().endswith("}"):
                         return ast.literal_eval(content)
                except:
                    pass

                # 4. Last ditch: try to parse the first valid JSON object with regex
                try:
                    import re
                    # Look for { ... } structure
                    match = re.search(r"\{.*\}", content, re.DOTALL)
                    if match:
                         found_json = match.group(0)
                         try:
                            return json.loads(found_json)
                         except:
                            # Try ast on the regex match
                            return ast.literal_eval(found_json)
                except:
                    pass
                    
                print(f"JSON Parse Failed. Raw content:\n{content[:200]}...")
                return {}
                    
        except Exception as e:
            print(f"JSON Gen Error ({model}): {e}")
            # Fallback for models that don't support JSON mode strictly or failed
            return {}

    def generate_image(self, prompt):
        """Generates image using qwen-image-plus via DashScope API."""
        api_key = MMABenchConfig.DASHSCOPE_API_KEY
        if not api_key:
            print("Error: DASHSCOPE_API_KEY not set.")
            return None
            
        url = MMABenchConfig.DASHSCOPE_BASE_URL.replace("compatible-mode/v1", "api/v1/services/aigc/multimodal-generation/generation")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Structure for qwen-image-plus
        payload = {
            "model": MMABenchConfig.IMAGE_MODEL,
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [{"text": prompt}]
                }]
            },
            "parameters": {
                "size": "1328*1328",
                "n": 1
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                res_json = response.json()
                # Parse response: output -> choices -> message -> content -> [ {image: ...} ]
                try:
                    content = res_json["output"]["choices"][0]["message"]["content"]
                    for item in content:
                        if "image" in item:
                            return item["image"]
                except (KeyError, IndexError, TypeError):
                    print(f"Image Parse Error. Raw: {res_json}")
            else:
                print(f"Image Gen Failed ({response.status_code}): {response.text}")
                
        except Exception as e:
            print(f"Image Gen Exception: {e}")
            
        return None
