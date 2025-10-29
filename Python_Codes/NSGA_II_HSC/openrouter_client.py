import requests
import json
import time
from typing import Dict, Any, Optional


class OpenRouterClient:
    """
    کلاینت برای ارتباط با OpenRouter API برای بهبود الگوریتم NSGA-II

    این کلاس مسئولیت‌های زیر را دارد:
    1. ارسال اطلاعات مسئله و کروموزوم به AI
    2. ارسال شاخص‌های عملکرد (HV, Spacing, etc.) هر 5 نسل
    3. دریافت پیشنهادات AI برای متدهای ترکیب، جهش و انتخاب
    4. پارس کردن پاسخ AI و تبدیل به فرمت قابل استفاده
    """

    def __init__(self, api_key: str, model: str):
        """
        Constructor برای OpenRouter Client

        Parameters:
        -----------
        api_key: کلید API از OpenRouter
        model: مدل AI مورد استفاده (پیش‌فرض: مدل رایگان Llama)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",  # اختیاری
            "X-Title": "NSGA-II Optimization",  # اختیاری
        }
        self._system_message = "You are an expert in genetic algorithms and multi-objective optimization. I have a humanitarian logistics optimization problem that is solved using the NSGA-II algorithm."

        # ذخیره اطلاعات مسئله برای استفاده در درخواست‌های بعدی
        self.problem_description = None
        self.chromosome_structure = None

    def set_problem_description(self, problem_info: Dict[str, Any]):
        """
        تنظیم اطلاعات مسئله برای ارسال به AI

        Parameters:
        -----------
        problem_info: دیکشنری شامل اطلاعات مسئله
        """
        self.problem_description = problem_info

    def _format_metrics_history(self, history: list) -> str:
        """
        فرمت کردن تاریخچه شاخص‌ها برای نمایش در prompt

        Parameters:
        -----------
        history: لیست شاخص‌های هر نسل

        Returns:
        --------
        str: رشته فرمت شده
        """
        if not history:
            return "No history available yet."

        formatted = ""
        for idx, metrics in enumerate(history):
            formatted += f"\n**Generation {idx}:**\n"
            formatted += f"  - Hypervolume (HV): {metrics.get('hypervolume', 0):.6f}\n"
            formatted += f"  - Spacing: {metrics.get('spacing', 0):.6f}\n"
            formatted += f"  - Spread: {metrics.get('spread', 0):.6f}\n"
            formatted += f"  - Pareto solutions: {metrics.get('pareto_count', 0)}\n"
            formatted += f"  - Avg crowding distance: {metrics.get('avg_crowding_distance', 0):.6f}\n"
            formatted += f"  - Diversity: {metrics.get('diversity', 0):.6f}\n"

        return formatted

    def create_prompt(
        self,
        generation: int,
        metrics: Dict[str, float],
        current_methods: Dict[str, Any],
    ) -> str:
        """
        ایجاد prompt یکپارچه شامل توضیحات مسئله و شاخص‌های عملکرد

        Parameters:
        -----------
        generation: نسل فعلی
        metrics: دیکشنری شاخص‌های عملکرد
        current_methods: متدهای فعلی استفاده شده

        Returns:
        --------
        str: متن prompt برای AI
        """
        prompt = f"""
**Problem Information:**
- Problem Type: Humanitarian Logistics Optimization
- Number of Objectives: 3 (F1: Total Distance, F2: Unmet Demand, F3: Death Probability)
- Chromosome Structure: 7 different parts
- Number of Generations: 40
- Population Size: 150
- Number of Distribution Centers: 3
- Number of Candidate Shelters: 11
- Number of Damage Points: 5
- Number of Hospitals: 4
- Number of Candidate Temporary Medical Centers: 10

**Chromosome Structure (7 parts):**

- **Note on part sizes:**
    1. The length of **Part 1**, **Part 2**, and **Part 3** is equal to the number of candidate shelters.
    2. The number of rows in **Part 4** and **Part 5** is equal to the number of damage points, and their number of columns is equal to the number of hospitals.
    3. The number of rows in **Part 6** and **Part 7** is equal to the number of damage points, and their number of columns is equal to (number of hospitals + number of candidate temporary medical centers).

1. **Part 1**: Distribution center assignment to shelters (integer list)
2. **Part 2**: Flow values from distribution centers (float list between 0-1)
3. **Part 3**: Damage point to shelter assignment (integer list)
   - **Part 3 consists of 2 sections:**
     - **Section 1** (indices 0 to number_of_damage_points-1): Permutation (no duplicates allowed)
     - **Section 2** (indices number_of_damage_points to number_of_shelters-1): Free assignment
4. **Part 4**: Severe injured to hospital assignment (float matrix)
5. **Part 5**: Percentage of severe injured transported by ambulance (float matrix)
6. **Part 6**: Minor injured to hospitals and temporary medical centers assignment (float matrix)
7. **Part 7**: Percentage of minor injured transported by ambulance (float matrix)

**Crossover and Mutation Recommendations Format:**
The chromosome has 7 distinct parts, each requiring different crossover and mutation methods:
- Parts 1: List (integer)
- Parts 2: List (float)
- Part 3: Has 2 sections requiring different methods:
  - **Part 3 Section 1** (indices 0 to number_of_damage_points-1): Permutation (no duplicates) - requires permutation-preserving methods
  - **Part 3 Section 2** (indices number_of_damage_points to number_of_shelters-1): Free assignment - can use any list method
- Parts 4 & 5: Matrices (float)
- Part 6 & 7: Matrix (float)

You must provide separate crossover and mutation methods for EACH part (including both sections of Part 3).

**Available Methods:**
- **Crossover**: 15 different methods:
    - one_point_crossover_list    (Suitable only for Part 1, 2, Part 3 Section 2)
    - two_point_crossover_list    (Suitable only for Part 1, 2, Part 3 Section 2)
    - multi_point_crossover_list  (Suitable only for Part 1, 2, Part 3 Section 2)
    - uniform_crossover_list      (Suitable only for Part 1, 2, Part 3 Section 2)
    - order_crossover_list        (Suitable only for Part 3 Section 1)
    - partially_mapped_crossover  (Suitable only for Part 3 Section 1)
    - arithmetic_crossover_list   (Suitable only for Part 2)
    - blend_crossover_list        (Suitable only for Part 2)
    - one_point_crossover_matrix  (Suitable only for Parts 4, 5, 6, 7)
    - two_point_crossover_matrix  (Suitable only for Parts 4, 5, 6, 7)
    - uniform_crossover_matrix    (Suitable only for Parts 4, 5, 6, 7)
    - block_crossover_matrix      (Suitable only for Parts 4, 5, 6, 7)
    - row_wise_crossover_matrix   (Suitable only for Parts 4, 5, 6, 7)
    - arithmetic_crossover_matrix (Suitable only for Parts 4, 5, 6, 7)
    - simulated_binary_crossover  (Suitable only for Part 2)

- **Mutation**: 16 different methods  
    - bit_flip_mutation_list         (Suitable only for Part 1, Part 3 Section 2)
    - swap_mutation_list             (Suitable only for Part 1, Part 3 Section 1, Part 3 Section 2)
    - inversion_mutation_list        (Suitable only for Part 1, Part 3 Section 1, Part 3 Section 2)
    - scramble_mutation_list         (Suitable only for Part 1, Part 3 Section 1, Part 3 Section 2)
    - insertion_mutation_list        (Suitable only for Part 1, Part 3 Section 1, Part 3 Section 2)
    - displacement_mutation_list     (Suitable only for Part 1, Part 3 Section 1, Part 3 Section 2)
    - gaussian_mutation_list         (Suitable only for Part 2)
    - uniform_mutation_list          (Suitable only for Part 2)
    - polynomial_mutation_list       (Suitable only for Part 2)
    - boundary_mutation_list         (Suitable only for Part 2)
    - random_element_mutation_matrix (Suitable only for Parts 4, 5, 6, 7)
    - gaussian_mutation_matrix       (Suitable only for Parts 4, 5, 6, 7)
    - row_mutation_matrix            (Suitable only for Parts 4, 5, 6, 7)
    - column_mutation_matrix         (Suitable only for Parts 4, 5, 6, 7)
    - block_mutation_matrix          (Suitable only for Parts 4, 5, 6, 7)
    - creep_mutation_matrix          (Suitable only for Parts 4, 5, 6, 7)

- **Selection**: 7 different methods  
    - crowded_binary_tournament  
    - adaptive_k_tournament  
    - feasibility_first_tournament  
    - rank_based_roulette  
    - reference_biased_tournament  
    - age_diversity_tournament  
    - epsilon_dominance_tournament  

Every 5 generations, I will send you the following performance metrics:
- Hypervolume (HV)
- Spacing
- Spread
- Number of Pareto solutions
- Average crowding distance
- Population diversity

**Performance Metrics History (from generation 0 to {generation}):**

Current generation: {generation}

**Metrics History (Metric history from the first run of the algorithm to the current run):**
{self._format_metrics_history(metrics.get("history", []))}

**Latest Generation Metrics:**
- Hypervolume (HV): {metrics.get("hypervolume", 0):.6f}
- Spacing: {metrics.get("spacing", 0):.6f}
- Spread: {metrics.get("spread", 0):.6f}
- Number of Pareto solutions: {metrics.get("pareto_count", 0)}
- Average crowding distance: {metrics.get("avg_crowding_distance", 0):.6f}
- Population diversity: {metrics.get("diversity", 0):.6f}

**Current Methods For the last 5 generations:**
- Crossover probability: {current_methods.get("crossover_probability", 0):.3f}
- Mutation probability: {current_methods.get("mutation_probability", 0):.3f}
- Inner mutation_rate (per-gene/element): {current_methods.get("mutation_inner_rate", 0.1):.3f}
- Selection: {current_methods.get("selection_method", "unknown")}
- Crossover methods per part: {current_methods.get("crossover_methods", {})}
- Mutation methods per part: {current_methods.get("mutation_methods", {})}

**Please suggest the best combination of methods and probabilities for the next 5 generations (generations {generation + 1} to {generation + 5}).**

        Provide your response in the following JSON format:
        ```json
        {{
            "analysis": "Brief analysis of current situation and recommendations",
            "recommendations": {{
                "crossover": {{
                    "part1": "method_name_for_part1",
                    "part2": "method_name_for_part2",
                    "part3_p1": "method_name_for_part3_section1_permutation",
                    "part3_p2": "method_name_for_part3_section2_free",
                    "part4": "method_name_for_part4",
                    "part5": "method_name_for_part5",
                    "part6": "method_name_for_part6",
                    "part7": "method_name_for_part7",
                    "probability": 0.9
                }},
                "mutation": {{
                    "part1": "method_name_for_part1",
                    "part2": "method_name_for_part2",
                    "part3_p1": "method_name_for_part3_section1_permutation",
                    "part3_p2": "method_name_for_part3_section2_free",
                    "part4": "method_name_for_part4",
                    "part5": "method_name_for_part5",
                    "part6": "method_name_for_part6",
                    "part7": "method_name_for_part7",
                    "probability": 0.1,
                    "mutation_rate": 0.10
                }},
                "selection": {{
                    "method": "suggested_selection_method_name"
                }}
            }}
        }}
        ```
        """
        return prompt

    def send_request(
        self, prompt: str, max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        ارسال درخواست به OpenRouter API

        Parameters:
        -----------
        prompt: متن prompt برای AI
        max_retries: حداکثر تعداد تلاش مجدد

        Returns:
        --------
        Dict: پاسخ AI یا None در صورت خطا
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": prompt},
            ],
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url, headers=self.headers, json=payload, timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    return data
                else:
                    print(
                        f"API request error (attempt {attempt + 1}): {response.status_code}"
                    )
                    print(f"Error text: {response.text}")

            except requests.exceptions.RequestException as e:
                print(f"API connection error (attempt {attempt + 1}): {e}")

            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff

        return None

    def parse_ai_response(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        پارس کردن پاسخ AI و استخراج پیشنهادات

        Parameters:
        -----------
        response: پاسخ API از OpenRouter

        Returns:
        --------
        Dict: پیشنهادات AI یا None در صورت خطا
        """
        try:
            content = response["choices"][0]["message"]["content"]

            # جستجوی JSON در پاسخ
            import re

            json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                recommendations = json.loads(json_str)
                return recommendations
            else:
                # اگر JSON پیدا نشد، سعی کن کل متن را پارس کن
                try:
                    recommendations = json.loads(content)
                    return recommendations
                except:  # noqa: E722
                    print("Could not extract JSON from AI response")
                    print(f"AI response: {content}")
                    return None

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Error parsing AI response: {e}")
            return None

    def get_recommendations(
        self,
        generation: int,
        metrics: Dict[str, float],
        current_methods: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        دریافت پیشنهادات AI برای بهبود الگوریتم

        Parameters:
        -----------
        generation: نسل فعلی
        metrics: شاخص‌های عملکرد
        current_methods: متدهای فعلی

        Returns:
        --------
        Dict: پیشنهادات AI یا None در صورت خطا
        """
        # ایجاد prompt
        prompt = self.create_prompt(generation, metrics, current_methods)

        # ارسال درخواست
        response = self.send_request(prompt)
        if response is None:
            return None

        # پارس کردن پاسخ
        recommendations = self.parse_ai_response(response)
        return recommendations

    def validate_recommendations(self, recommendations: Dict[str, Any]) -> bool:
        """
        اعتبارسنجی پیشنهادات AI

        Parameters:
        -----------
        recommendations: پیشنهادات AI

        Returns:
        --------
        bool: اعتبار پیشنهادات
        """
        try:
            # بررسی وجود کلیدهای ضروری
            required_keys = ["recommendations"]
            if not all(key in recommendations for key in required_keys):
                return False

            rec = recommendations["recommendations"]

            # بررسی متدهای پیشنهادی
            valid_crossover = [
                "one_point_crossover_list",
                "two_point_crossover_list",
                "uniform_crossover_list",
                "order_crossover_list",
                "arithmetic_crossover_list",
                "blend_crossover_list",
                "one_point_crossover_matrix",
                "two_point_crossover_matrix",
                "uniform_crossover_matrix",
                "block_crossover_matrix",
                "row_wise_crossover_matrix",
                "arithmetic_crossover_matrix",
                "multi_point_crossover_list",
                "partially_mapped_crossover",
                "simulated_binary_crossover",
            ]

            valid_mutation = [
                "bit_flip_mutation_list",
                "swap_mutation_list",
                "inversion_mutation_list",
                "scramble_mutation_list",
                "insertion_mutation_list",
                "displacement_mutation_list",
                "gaussian_mutation_list",
                "uniform_mutation_list",
                "polynomial_mutation_list",
                "boundary_mutation_list",
                "random_element_mutation_matrix",
                "gaussian_mutation_matrix",
                "row_mutation_matrix",
                "column_mutation_matrix",
                "block_mutation_matrix",
                "creep_mutation_matrix",
            ]

            valid_selection = [
                "crowded_binary_tournament",
                "adaptive_k_tournament",
                "feasibility_first_tournament",
                "rank_based_roulette",
                "reference_biased_tournament",
                "age_diversity_tournament",
                "epsilon_dominance_tournament",
            ]

            # بررسی متدهای پیشنهادی برای هر بخش
            if "crossover" in rec:
                for part in [
                    "part1",
                    "part2",
                    "part3_p1",
                    "part3_p2",
                    "part4",
                    "part5",
                    "part6",
                    "part7",
                ]:
                    if part in rec["crossover"]:
                        if rec["crossover"][part] not in valid_crossover:
                            print(
                                f"⚠️ Invalid crossover method for {part}: {rec['crossover'][part]}"
                            )
                            return False

            if "mutation" in rec:
                for part in [
                    "part1",
                    "part2",
                    "part3_p1",
                    "part3_p2",
                    "part4",
                    "part5",
                    "part6",
                    "part7",
                ]:
                    if part in rec["mutation"]:
                        if rec["mutation"][part] not in valid_mutation:
                            print(
                                f"⚠️ Invalid mutation method for {part}: {rec['mutation'][part]}"
                            )
                            return False

            if "selection" in rec:
                if rec["selection"]["method"] not in valid_selection:
                    print(f"⚠️ Invalid selection method: {rec['selection']['method']}")
                    return False

            # بررسی احتمال‌ها و نرخ داخلی جهش
            if "crossover" in rec and "probability" in rec["crossover"]:
                prob = rec["crossover"]["probability"]
                if not (0 <= prob <= 1):
                    print(f"⚠️ Invalid crossover probability: {prob}")
                    return False

            if "mutation" in rec and "probability" in rec["mutation"]:
                prob = rec["mutation"]["probability"]
                if not (0 <= prob <= 1):
                    print(f"⚠️ Invalid mutation probability: {prob}")
                    return False

            if "mutation" in rec and "mutation_rate" in rec["mutation"]:
                inner = rec["mutation"]["mutation_rate"]
                if not (0 <= inner <= 1):
                    print(f"⚠️ Invalid inner mutation_rate: {inner}")
                    return False

            return True

        except Exception as e:
            print(f"❌ Error validating recommendations: {e}")
            return False
