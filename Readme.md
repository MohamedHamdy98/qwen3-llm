
# 🧪 Qwen API Testing Guide

This guide provides `curl` examples for testing the Qwen API endpoints with text, images, and PDF files.

---

## 🔹 Generate with Text

Sends a user message along with an image file.

```bash

curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "prompt=اشرح لي الذكاء الاصطناعي&max_tokens=256"
```