"""
فایل تنظیمات برای AI Optimization

این فایل شامل تنظیمات مورد نیاز برای استفاده از AI optimization در NSGA-II است.
"""

# ==================== تنظیمات OpenRouter API ====================

# API Key از OpenRouter (از https://openrouter.ai/ دریافت کنید)
OPENROUTER_API_KEY = "sk-Wne1ay7BrN5P1wxjU40DRW0bRkQO8ocPS4EdwIjK6TMaPgPW"  # "your-api-key-here"
AI_MODEL = "deepseek-chat"
BASE_URL = "https://api.gapapi.com/v1"
# OPENROUTER_API_KEY = "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"  # "your-api-key-here"
# AI_MODEL = "openai/gpt-oss-20b:free"
# BASE_URL = "https://openrouter.ai/api/v1"
# AI_MODEL = "minimax/minimax-m2:free"
# BASE_URL = "https://api.gapapi.com/v1"

# ==================== تنظیمات AI Optimization ====================
# فعال/غیرفعال کردن AI optimization
USE_AI_OPTIMIZATION = True
# فاصله زمانی درخواست پیشنهادات از AI (هر چند نسل)
AI_UPDATE_INTERVAL = 5  # هر 5 نسل
# حداکثر تعداد تلاش مجدد برای درخواست API
MAX_API_RETRIES = 3
# تایم‌اوت درخواست API (ثانیه)
API_TIMEOUT = 30

# ==================== تنظیمات پیش‌فرض الگوریتم ====================

# نرخ‌های پیش‌فرض
DEFAULT_CROSSOVER_RATE = 0.9
DEFAULT_MUTATION_RATE = 0.1

# ==================== تنظیمات لاگ و نمایش ====================

# نمایش جزئیات AI optimization
VERBOSE_AI = True

# ذخیره تاریخچه پیشنهادات AI
SAVE_AI_HISTORY = True

# مسیر ذخیره فایل‌های AI
AI_OUTPUT_DIR = "ai_outputs"

# ==================== تنظیمات اعتبارسنجی ====================

# اعتبارسنجی پیشنهادات AI
VALIDATE_AI_RECOMMENDATIONS = True

# استفاده از متدهای پیش‌فرض در صورت نامعتبر بودن پیشنهادات AI
FALLBACK_TO_DEFAULT = True

# ==================== تابع کمکی برای تنظیمات ====================

def get_ai_config():
    """
    دریافت تنظیمات AI optimization
    
    Returns:
    --------
    Dict: تنظیمات AI
    """
    return {
        'api_key': OPENROUTER_API_KEY,
        'model': AI_MODEL,
        'base_url': BASE_URL,
        'use_optimization': USE_AI_OPTIMIZATION,
        'update_interval': AI_UPDATE_INTERVAL,
        'max_retries': MAX_API_RETRIES,
        'timeout': API_TIMEOUT,
        'verbose': VERBOSE_AI,
        'save_history': SAVE_AI_HISTORY,
        'output_dir': AI_OUTPUT_DIR,
        'validate': VALIDATE_AI_RECOMMENDATIONS,
        'fallback': FALLBACK_TO_DEFAULT
    }

def is_ai_ready():
    """
    بررسی آمادگی AI optimization
    
    Returns:
    --------
    bool: آیا AI optimization آماده است
    """
    return (OPENROUTER_API_KEY is not None and 
            OPENROUTER_API_KEY != "your-api-key-here" and
            USE_AI_OPTIMIZATION)

def print_config_status():
    """
    نمایش وضعیت تنظیمات
    """
    print("🔧 AI Optimization Configuration Status:")
    print(f"   API Key: {'✅ Set' if OPENROUTER_API_KEY and OPENROUTER_API_KEY != 'your-api-key-here' else '❌ Not Set'}")
    print(f"   AI Model: {AI_MODEL}")
    print(f"   AI Optimization: {'✅ Enabled' if USE_AI_OPTIMIZATION else '❌ Disabled'}")
    print(f"   Update Interval: Every {AI_UPDATE_INTERVAL} generations")
    print(f"   Verbose Mode: {'✅ Enabled' if VERBOSE_AI else '❌ Disabled'}")
    
    if is_ai_ready():
        print("\n🎉 AI Optimization is ready to use!")
    else:
        print("\n⚠️ To enable AI Optimization:")
        print("   1. Set your actual OpenRouter API key in OPENROUTER_API_KEY")
        print("   2. Set USE_AI_OPTIMIZATION to True")

if __name__ == "__main__":
    print_config_status()



