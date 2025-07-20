# app_router.py
from fastapi import APIRouter, HTTPException, Form, Query, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from models.gpu_log import GPULog
import torch, logging
from utils.model_loader import load_model_and_tokenizer
from utils.helper_functions import get_gpu_info
import uuid
from database.session import SessionLocal
from utils.save_gpu_log import save_gpu_log

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model & tokenizer
print("🔄 Loading Qwen3-4B model and tokenizer...")
model, tokenizer = load_model_and_tokenizer()
print("✅ Model loaded to device:", model.device)

# @router.post("/generate")
# def generate_text(
#     prompt: str = Form(...),
#     max_tokens: int = Form(512),
#     thinking: bool = Form(False)
# ):
#     try:
#         # Format as chat messages
#         messages = [{"role": "user", "content": prompt}]
#         prompt_text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#             enable_thinking=thinking  # Optional, supported in Qwen
#         )

#         # Tokenize input
#         model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

#         # Generate output
#         output_ids = model.generate(
#             **model_inputs,
#             max_new_tokens=max_tokens
#         )[0][model_inputs.input_ids.shape[1]:]

#         # Optional: parse thinking content if exists
#         try:
#             think_token_id = tokenizer.convert_tokens_to_ids("</think>")
#             index = len(output_ids) - output_ids[::-1].index(think_token_id)
#         except ValueError:
#             index = 0

#         thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
#         response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

#         return {
#             "thinking": thinking,
#             "response": response
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"🔥 Inference error: {str(e)}")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/extract_controls")
def extract_controls(
    prompt: str = Form(...),
    max_tokens: int = Form(512),
    thinking: bool = Form(False)
):
    request_id = str(uuid.uuid4())
    try:
        logger.info(f"📥 Request {request_id} started for /extract_controls")
        gpu_before = get_gpu_info()
        logger.info(f"📊 GPU BEFORE: {gpu_before}")
        save_gpu_log("extract_controls", request_id, gpu_before, "before")
        logger.info("📥 Received request to /extract_controls endpoint.")
        logger.info(f"Prompt length: {len(prompt)} characters | max_tokens: {max_tokens} | thinking: {thinking}")

        # ✅ System message to instruct the LLM
        system_message = "\n".join([
            "You are an intelligent assistant designed to extract legal or regulatory clauses. Your task is to extract **explicit instructions, obligations, or restrictions** from texts in **Arabic or English** sourced from policies or contracts, and return them in a structured textual format.",
            "📌 Rules you must follow:",
            "1. The input will be raw text extracted from a Word document.",
            "2. Completely ignore any content under the headings “Introduction” or “Definitions”.",
            "3. Extract only sentences that contain **actual instructions, obligations, or restrictions**.",
            "🟢 Valid sentences typically begin with:",
            "- In Arabic: \"يجب\", \"لا يجوز\", \"لا يجب\", \"يمكن\", \"يقتصر\", \"يُحظر\", \"يلتزم\", \"يتعين\"",
            "- In English: \"must\", \"must not\", \"shall\", \"shall not\", \"should\", \"may\", \"is required to\", \"is prohibited from\"",
            "📦 Expected output format:",
            "- If the input is Arabic:",
            "    - Each clause must begin with: التعليمات: n, where `n` is a sequential number.",
            "    - On the next line: the Arabic clause text.",
            "- If the input is English:",
            "    - Each clause must begin with: Instruction: n",
            "    - On the next line: the English clause text.",
            "🚫 Do NOT include:",
            "- Any interpretations or explanations",
            "- Any Markdown formatting",
            "- Any JSON or code snippets",
            "✅ Examples:",
            "Arabic:",
            "التعليمات: 1 - يجب على المُستخدمين أن يلتزموا بالشروط والمنافع العامة والخاصة لتقديم خدمة الرسائل القصيرة (SMS).",
            "التعليمات: 2 - يُحظر على المُستخدمين استخدام الخدمة بشكل غير قانوني أو مخالف للشروط والمنافع العامة والخاصة.",
            "التعليمات: 3 - يجب على المصرح له الالتزام بما يصدر من الهيئة من تنظيمات وقرارات تتعلق بالرسائل القصيرة",
            "English:",
            "Instruction: 1",
            "The user must maintain confidentiality."
        ])


        logger.info("🧠 Preparing messages for the LLM...")

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

        logger.info("🔁 Tokenizing input and sending to model...")
        model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens
        )[0][model_inputs.input_ids.shape[1]:]

        full_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        logger.info("✅ Model generation complete.")

        if "</think>" in full_output:
            clean_output = full_output.split("</think>")[-1].strip()
            logger.info("🧽 Cleaned output after </think> tag.")
        else:
            clean_output = full_output

        logger.info("🎯 Returning extracted clauses JSON.")

        # 🚀 Log GPU state after processing
        gpu_after = get_gpu_info()
        logger.info(f"📊 GPU AFTER: {gpu_after}")
        save_gpu_log("extract_controls", request_id, gpu_after, "after")

        return {
            "response": clean_output
        }

    except Exception as e:
        logger.exception("🔥 Inference error occurred.")
        raise HTTPException(status_code=500, detail=f"🔥 Inference error: {str(e)}")



@router.get("/logs/gpu", summary="Query GPU usage logs", tags=["Admin"])
def get_gpu_logs(
    request_id: Optional[str] = Query(None, description="Filter by request ID"),
    endpoint: Optional[str] = Query(None, description="Filter by endpoint"),
    stage: Optional[str] = Query(None, regex="^(before|after)$", description="Filter by stage"),
    limit: int = Query(100, ge=1, le=1000, description="Max number of logs to return"),
    db: Session = Depends(get_db)
):
    query = db.query(GPULog)

    if request_id:
        query = query.filter(GPULog.request_id == request_id)
    if endpoint:
        query = query.filter(GPULog.endpoint == endpoint)
    if stage:
        query = query.filter(GPULog.stage == stage)

    logs = query.order_by(GPULog.timestamp.desc()).limit(limit).all()

    return [
        {
            "id": log.id,
            "timestamp": log.timestamp,
            "endpoint": log.endpoint,
            "request_id": log.request_id,
            "stage": log.stage,
            "gpu_index": log.gpu_index,
            "mem_used_MB": log.mem_used_MB,
            "mem_total_MB": log.mem_total_MB,
            "gpu_util_percent": log.gpu_util_percent,
            "mem_util_percent": log.mem_util_percent,
        }
        for log in logs
    ]
    
@router.get("/")
def home():
    return {"message": "Qwen 3 Model is Working..🔥"}