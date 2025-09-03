import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates)

def get_gpu_info(index=0):
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(index)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        utilization = nvmlDeviceGetUtilizationRates(handle)

        gpu_data = {
            "gpu_index": index,
            "mem_used_MB": mem_info.used // 1024**2,
            "mem_total_MB": mem_info.total // 1024**2,
            "gpu_util_percent": utilization.gpu,
            "mem_util_percent": utilization.memory
        }
        return gpu_data
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            nvmlShutdown()
        except:
            pass

from langdetect import detect

def build_system_message(text: str):
    try:
        lang = detect(text)
    except:
        lang = "ar"  # fallback

    if lang == "en":
        # النسخة الإنجليزية
        system_message = """
Extract regulatory clauses from the text. Any sentence that contains an obligation, restriction, or instruction = a separate clause. Output JSON only.

Example 1 ("must" patterns):
Text: "The licensee must obtain approval. The licensee must not resell. The subscriber shall verify the data."

Output:
{
  "clauses": [
    {
      "title": "Instruction: 1",
      "description": "The licensee must obtain approval.",
    },
    {
      "title": "Instruction: 2",
      "description": "The licensee must not resell SMS services to external parties.",
    },
    {
      "title": "Instruction: 3",
      "description": "The subscriber shall verify the data.",
    }
  ]
}

Example 2 (other patterns):
Text: "The service may not be used for illegal purposes. Publishing prohibited content is forbidden. No single user authorization."

Output:
{
  "clauses": [
    {
      "title": "Instruction: 1",
      "description": "The service may not be used for illegal purposes.",
    },
    {
      "title": "Instruction: 2",
      "description": "Publishing prohibited content is forbidden.",
    },
    {
      "title": "Instruction: 3",
      "description": "No single user authorization for message preparation and approval.",
    }
  ]
}

🔍 Look for:
- must, shall, required, mandatory
- may not, prohibited, forbidden, not allowed
- obligation, restriction, necessity
- "should be" as a condition

Now extract from the following text:
        """
    else:
        # النسخة العربية (زي اللي عندك)
        system_message = '''
استخرج البنود التنظيمية من النص. أي جملة تحتوي على التزام أو قيد أو تعليمة = بند منفصل. إخراج JSON فقط.

مثال 1 (أنماط "يجب"):
النص: "يجب على المصرح له الحصول على موافقة. يجب ألا يقوم بإعادة البيع. يتعين على المشترك التحقق من البيانات."

الإخراج:
{
  "clauses": [
    {
      "title": "التعليمات: 1",
      "description": "يجب على المصرح له الحصول على موافقة.",
    },
    {
      "title": "التعليمات: 2",
      "description": "يجب ألا يقوم المصرح له لتقديم خدمة الرسائل القصيرة بإعادة البيع لأي جهة خارج المملكة.",
    },
    {
      "title": "التعليمات: 3",
      "description": "يتعين على المشترك التحقق من البيانات.",
    }
  ]
}

مثال 2 (أنماط أخرى):
النص: "لا يجوز استخدام الخدمة لأغراض غير قانونية. يُحظر نشر المحتوى المخالف. عدم منح صلاحية للمستخدم الواحد."

الإخراج:
{
  "clauses": [
    {
      "title": "التعليمات: 1",
      "description": "لا يجوز استخدام الخدمة لأغراض غير قانونية.",
    },
    {
      "title": "التعليمات: 2",
      "description": "يُحظر نشر المحتوى المخالف.",
    },
    {
      "title": "التعليمات: 3",
      "description": "عدم منح صلاحية إعداد محتوى الرسالة واعتمادها لمستخدم واحد.",
    }
  ]
}

مثال 3 (جمل بدون كلمات إلزام واضحة لكن تحتوي قواعد):
النص: "الخادم الرئيسي داخل المملكة فقط. المستخدم سعودي الجنسية. تخزين البيانات داخلياً مطلوب."

الإخراج:
{
  "clauses": [
    {
      "title": "التعليمات: 1",
      "description": "يجب أن يكون الخادم الرئيسي لتقديم الخدمة داخل المملكة وتخزين البيانات داخلياً.",
    },
    {
      "title": "التعليمات: 2",
      "description": "أن يكون المستخدم لنظام إرسال الرسائل القصيرة سعودي الجنسية.",
    }
  ]
}

مثال 4 (خليط من كل الأنماط):
النص: "يجب الحصول على الترخيص. الشركة تقدم الخدمة بجودة عالية. لا يُسمح بالوصول غير المُصرح به. ضرورة تحديث البيانات شهرياً."

الإخراج:
{
  "clauses": [
    {
      "title": "التعليمات: 1",
      "description": "يجب الحصول على الترخيص.",
    },
    {
      "title": "التعليمات: 2",
      "description": "لا يُسمح بالوصول غير المُصرح به.",
    },
    {
      "title": "التعليمات: 3",
      "description": "ضرورة تحديث البيانات شهرياً.",
    }
  ]
}

🔍 ابحث عن:
- يجب، يجب على، يجب ألا، يجب أن
- لا يجوز، لا يُسمح، يُحظر، ممنوع
- يتعين، يلتزم، ضرورة، وجوب
- عدم، ألا يقوم، لا يحق
- "أن يكون" (كشرط أو متطلب)

الآن استخرج من النص التالي:
        '''
    
    return system_message
