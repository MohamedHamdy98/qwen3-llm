# app_router.py
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
import subprocess
import psutil
import torch, logging
from utils.model_loader import load_model_and_tokenizer

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model & tokenizer
print("🔄 Loading Qwen3-4B model and tokenizer...")
model, tokenizer = load_model_and_tokenizer()
print("✅ Model loaded to device:", model.device)

@router.post("/generate")
def generate_text(
    prompt: str = Form(...),
    max_tokens: int = Form(512),
    thinking: bool = Form(False)
):
    try:
        # Format as chat messages
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking  # Optional, supported in Qwen
        )

        # Tokenize input
        model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

        # Generate output
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens
        )[0][model_inputs.input_ids.shape[1]:]

        # Optional: parse thinking content if exists
        try:
            think_token_id = tokenizer.convert_tokens_to_ids("</think>")
            index = len(output_ids) - output_ids[::-1].index(think_token_id)
        except ValueError:
            index = 0

        thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
        response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

        return {
            "thinking": thinking,
            "response": response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"🔥 Inference error: {str(e)}")

@router.post("/extract_controls")
def extract_controls(
    prompt: str = Form(...),
    max_tokens: int = Form(512),
    thinking: bool = Form(False)
):
    try:
        # ✅ System message to instruct the LLM
        system_message = """
        You are a smart clause extraction assistant. Your task is to extract clear regulatory or contractual **instructions or obligations** from raw policy content written in **Arabic or English**, and return them in a structured JSON format.

        📌 Rules to follow:
        1. The input will be raw text extracted from a Word document.
        2. Completely **ignore** any content under headings titled “المقدمة” or “التعريفات” (Introduction / Definitions).
        3. Extract only actual **instructions, obligations, or restrictions**.

        🟢 Valid clauses typically begin with:
        - In Arabic: "يجب", "لا يجوز", "لا يجب", "يمكن", "يقتصر", "يُحظر", "يلتزم", "يتعين"
        - In English: "must", "must not", "shall", "shall not", "should", "may", "is required to", "is prohibited from"

        📦 Expected JSON format:
        Return the extracted clauses as an array of objects like:
        {
          "clauses": [
            { "title": "التعليمات: 1", "description": "..." },
            { "title": "التعليمات: 2", "description": "..." }
          ]
        }

        🚫 Do NOT include:
        - Any explanation or reasoning
        - Any markdown formatting (e.g., ```json)
        - Any extra tags (e.g., <think>)
        - Any text outside the JSON block

        ✅ Your output must begin with:
        {
        "clauses": [
        """

        # Format as chat messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking
        )

        model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

        # Generate output
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens
        )[0][model_inputs.input_ids.shape[1]:]

        # Decode full output
        full_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # ✅ Extract only the part after </think> if it exists
        if "</think>" in full_output:
            clean_output = full_output.split("</think>")[-1].strip()
        else:
            clean_output = full_output

        return {
            "response": clean_output  # 🔥 Just the JSON response only
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"🔥 Inference error: {str(e)}")


@router.get("/monitor_dashboard")
async def monitor_dashboard():
    try:
        # 1. GPU Info via PyTorch
        gpu_stats = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                torch.cuda.set_device(i)
                gpu_stats.append({
                    "device_id": i,
                    "device_name": props.name,
                    "total_memory_MB": round(props.total_memory / 1024**2, 2),
                    "memory_allocated_MB": round(torch.cuda.memory_allocated(i) / 1024**2, 2),
                    "memory_reserved_MB": round(torch.cuda.memory_reserved(i) / 1024**2, 2),
                    "capability": props.major,
                })
        else:
            gpu_stats = "CUDA not available"

        # 2. Check FlashAttention usage (only if manually enabled)
        flash_attention_used = "flash_attention_2" in str(getattr(model.config, "attn_implementation", "")).lower()

        # 3. Nvidia-smi raw output
        try:
            smi_output = subprocess.check_output(["nvidia-smi"], timeout=5).decode()
        except Exception as e:
            smi_output = f"nvidia-smi error: {str(e)}"

        # 4. CPU memory
        virtual_mem = psutil.virtual_memory()
        cpu_memory = {
            "total_MB": round(virtual_mem.total / 1024**2, 2),
            "used_MB": round(virtual_mem.used / 1024**2, 2),
            "percent": virtual_mem.percent
        }

        # 5. Model loaded device (optional)
        model_device = str(next(model.parameters()).device) if model else "Not loaded"

        return JSONResponse(content={
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "flash_attention_used": flash_attention_used,
            "gpu_stats": gpu_stats,
            "cpu_memory": cpu_memory,
            "model_device": model_device,
            "nvidia_smi": smi_output
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})
    
@router.get("/")
def home():
    return {"message": "Qwen Model is Working..🔥"}