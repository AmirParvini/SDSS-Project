"""
فایل تنظیمات برای AI Optimization

این فایل شامل تنظیمات مورد نیاز برای استفاده از AI optimization در NSGA-II است.
"""

# ==================== تنظیمات OpenRouter API ====================

# API Key از OpenRouter (از https://openrouter.ai/ دریافت کنید)
OPENROUTER_API_KEY = "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"  # "your-api-key-here"

# مدل AI مورد استفاده (مدل‌های رایگان موجود)
AI_MODEL = "minimax/minimax-m2:free"  # مدل رایگان deepseek

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
    print("🔧 وضعیت تنظیمات AI Optimization:")
    print(f"   API Key: {'✅ تنظیم شده' if OPENROUTER_API_KEY and OPENROUTER_API_KEY != 'your-api-key-here' else '❌ تنظیم نشده'}")
    print(f"   AI Model: {AI_MODEL}")
    print(f"   AI Optimization: {'✅ فعال' if USE_AI_OPTIMIZATION else '❌ غیرفعال'}")
    print(f"   Update Interval: هر {AI_UPDATE_INTERVAL} نسل")
    print(f"   Verbose Mode: {'✅ فعال' if VERBOSE_AI else '❌ غیرفعال'}")
    
    if is_ai_ready():
        print("\n🎉 AI Optimization آماده استفاده است!")
    else:
        print("\n⚠️ برای فعال‌سازی AI Optimization:")
        print("   1. API key واقعی OpenRouter را در OPENROUTER_API_KEY قرار دهید")
        print("   2. USE_AI_OPTIMIZATION را True کنید")

if __name__ == "__main__":
    print_config_status()



