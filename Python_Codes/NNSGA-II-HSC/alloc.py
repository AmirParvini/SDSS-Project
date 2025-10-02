def allocate_population_under_capacity(total_population, capacity_a, capacity_b, capacity_c):
    """
    جمعیت را بین سه مکان تخصیص می دهد به شرطی که جمعیت کمتر یا مساوی با 
    مجموع کل ظرفیت ها باشد. تخصیص بر اساس نسبت ظرفیت ها انجام می شود.

    ورودی ها:
    - total_population (int/float): کل جمعیتی که باید تخصیص داده شود.
    - capacity_a (int/float): ظرفیت مکان A.
    - capacity_b (int/float): ظرفیت مکان B.
    - capacity_c (int/float): ظرفیت مکان C.

    خروجی:
    - allocation_a, allocation_b, allocation_c (tuple): تخصیص داده شده به هر مکان.
    """

    # 1. محاسبه مجموع کل ظرفیت ها
    total_capacity = capacity_a + capacity_b + capacity_c

    # بررسی شرط مسئله: جمعیت باید کمتر یا مساوی کل ظرفیت باشد
    if total_population > total_capacity:
        # در این حالت باید از روش دیگری (مانند تخصیص عادلانه یا بر اساس اولویت) استفاده شود.
        # اما طبق خواسته شما، این کد برای حالت "ظرفیت بیشتر یا مساوی" طراحی شده است.
        raise ValueError("جمعیت کل (Population: {}) از مجموع ظرفیت ها (Capacity: {}) بیشتر است. این کد برای این حالت طراحی نشده است.".format(total_population, total_capacity))

    # 2. محاسبه سهم نسبی هر مکان از کل ظرفیت
    # این نسبت، درصد تخصیص از کل جمعیت را مشخص می کند.
    ratio_a = capacity_a / total_capacity
    ratio_b = capacity_b / total_capacity
    # می توانیم ratio_c را هم محاسبه کنیم یا از (1 - ratio_a - ratio_b) استفاده کنیم
    ratio_c = capacity_c / total_capacity

    # 3. محاسبه تخصیص بر اساس سهم نسبی
    # از تابع round برای گرد کردن به نزدیکترین عدد صحیح استفاده می کنیم (اگر جمعیت ها باید عدد صحیح باشند)
    # توجه: گرد کردن ممکن است مجموع تخصیص ها را اندکی تغییر دهد.
    allocation_a = round(total_population * ratio_a)
    allocation_b = round(total_population * ratio_b)
    
    # برای مکان آخر، جهت تضمین اینکه مجموع تخصیص ها دقیقاً برابر با کل جمعیت شود،
    # آن را از باقیمانده کسر می کنیم تا از خطای ناشی از گرد کردن جلوگیری شود.
    allocation_c = total_population - allocation_a - allocation_b
    
    # 4. بررسی نهایی (اختیاری) - فقط برای اطمینان از عدم تجاوز تخصیص از ظرفیت
    # در این روش (توزیع بر اساس نسبت ظرفیت)، این شرط همواره برقرار است.
    if allocation_a > capacity_a or allocation_b > capacity_b or allocation_c > capacity_c:
        # این حالت نباید رخ دهد مگر به دلیل خطای محاسباتی/گرد کردن بسیار نادر
        print("هشدار: خطای گرد کردن باعث شده تخصیص از ظرفیت فراتر رود! (اگرچه در عمل نادر است)")


    return allocation_a, allocation_b, allocation_c

# --- مثال های کاربردی ---

# مثال 1: ظرفیت بسیار بیشتر از جمعیت
population_1 = 100
cap_a_1, cap_b_1, cap_c_1 = 50, 150, 200 # مجموع ظرفیت: 400
# نسبت ها: 12.5%، 37.5%، 50%

alloc_a_1, alloc_b_1, alloc_c_1 = allocate_population_under_capacity(population_1, cap_a_1, cap_b_1, cap_c_1)

print("--- مثال ۱ (ظرفیت >> جمعیت) ---")
print(f"جمعیت کل: {population_1} | کل ظرفیت: {cap_a_1 + cap_b_1 + cap_c_1}")
print(f"تخصیص A: {alloc_a_1} (ظرفیت: {cap_a_1})") # انتظار: 13
print(f"تخصیص B: {alloc_b_1} (ظرفیت: {cap_b_1})") # انتظار: 37
print(f"تخصیص C: {alloc_c_1} (ظرفیت: {cap_c_1})") # انتظار: 50 (باقیمانده: 100 - 13 - 37)
print(f"مجموع تخصیص: {alloc_a_1 + alloc_b_1 + alloc_c_1}")
print("---")

# ---
# مثال 2: ظرفیت برابر با جمعیت
population_2 = 300
cap_a_2, cap_b_2, cap_c_2 = 75, 150, 75 # مجموع ظرفیت: 300
# نسبت ها: 33.3%، 33.3%، 33.3%

alloc_a_2, alloc_b_2, alloc_c_2 = allocate_population_under_capacity(population_2, cap_a_2, cap_b_2, cap_c_2)

print("--- مثال ۲ (ظرفیت = جمعیت) ---")
print(f"جمعیت کل: {population_2} | کل ظرفیت: {cap_a_2 + cap_b_2 + cap_c_2}")
print(f"تخصیص A: {alloc_a_2} (ظرفیت: {cap_a_2})") # انتظار: 100
print(f"تخصیص B: {alloc_b_2} (ظرفیت: {cap_b_2})") # انتظار: 100
print(f"تخصیص C: {alloc_c_2} (ظرفیت: {cap_c_2})") # انتظار: 100
print(f"مجموع تخصیص: {alloc_a_2 + alloc_b_2 + alloc_c_2}")
print("---")
# ---
# مثال 3: توزیع نامتوازن نزدیک به ظرفیت
population_3 = 190
cap_a_3, cap_b_3, cap_c_3 = 40, 50, 150 # مجموع ظرفیت: 200
# نسبت ها: 20%، 25%، 55%

alloc_a_3, alloc_b_3, alloc_c_3 = allocate_population_under_capacity(population_3, cap_a_3, cap_b_3, cap_c_3)

print("--- مثال ۳ (نامتوازن نزدیک به ظرفیت) ---")
print(f"جمعیت کل: {population_3} | کل ظرفیت: {cap_a_3 + cap_b_3 + cap_c_3}")
print(f"تخصیص A: {alloc_a_3} (ظرفیت: {cap_a_3})") # انتظار: 38
print(f"تخصیص B: {alloc_b_3} (ظرفیت: {cap_b_3})") # انتظار: 48
print(f"تخصیص C: {alloc_c_3} (ظرفیت: {cap_c_3})") # انتظار: 104 (با باقیمانده)
print(f"مجموع تخصیص: {alloc_a_3 + alloc_b_3 + alloc_c_3}")
print("---")