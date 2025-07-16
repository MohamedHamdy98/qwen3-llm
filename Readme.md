
# 🧪 Qwen API Testing Guide

This guide provides `curl` examples for testing the Qwen API endpoints with text, images, and PDF files.

---

## 🔹 Generate with Text and Image

Sends a user message along with an image file.

```bash
curl -X 'POST' \
  'https://olive-rhubarb-0skssjp0m12oy2i1.salad.cloud/qwen/generate' \
  -H 'Content-Type: multipart/form-data' \
  -F 'messages="[{\"role\": \"user\", \"content\": \"Hello with an image!\"}]"' \
  -F 'max_new_tokens=128' \
  -F 'files=@/path/to/your/image.jpg'
```

---

## 🔹 Generate with Text Only

Sends just a text message without any files.

```bash
curl -X 'POST' \
  'https://olive-rhubarb-0skssjp0m12oy2i1.salad.cloud/qwen/generate' \
  -H 'Content-Type: multipart/form-data' \
  -F 'messages="[{\"role\": \"user\", \"content\": \"Hello!\"}]"' \
  -F 'max_new_tokens=128'
```

---

## 🔹 Generate with PDF (Text and Images Extracted)

Sends a plain message and a PDF file (which may contain text and images).

```bash
curl -X POST \
  'https://olive-rhubarb-0skssjp0m12oy2i1.salad.cloud/qwen/generate_pdf' \
  -F 'messages=["Hello, how are you?"]' \
  -F 'max_new_tokens=128' \
  -F 'files=@/path/to/your_file.pdf'
```

---

## 📝 Notes

- 📁 Replace `/path/to/your/image.jpg` or `/path/to/your_file.pdf` with real file paths ( if you will upload path! ).
- 🧾 The `messages` field is expected to be a stringified JSON array ( just in the generate endpoint ).
- 📚 The `/generate_pdf` endpoint automatically extracts **text and images** from PDFs and processes them.

---
