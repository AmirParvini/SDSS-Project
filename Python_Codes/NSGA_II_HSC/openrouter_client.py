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
            formatted += f"  - Elitism rate: {metrics.get('elitism_rate', 0):.4f}\n"
            # اضافه کردن آمار اهداف از کل جمعیت
            mean_obj = metrics.get('mean_objectives', [0, 0, 0])
            min_obj = metrics.get('min_objectives', [0, 0, 0])
            std_obj = metrics.get('std_objectives', [0, 0, 0])
            formatted += f"  - Mean objectives: F1={mean_obj[0]:.6f}, F2={mean_obj[1]:.6f}, F3={mean_obj[2]:.6f}\n"
            formatted += f"  - Min objectives: F1={min_obj[0]:.6f}, F2={min_obj[1]:.6f}, F3={min_obj[2]:.6f}\n"
            formatted += f"  - Std objectives: F1={std_obj[0]:.6f}, F2={std_obj[1]:.6f}, F3={std_obj[2]:.6f}\n"

        return formatted

    def create_prompt(
        self,
        generation: int,
        metrics: Dict[str, float],
        current_methods: Dict[str, Any],
        metrics_history: list[Dict[str, Any]],
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
<Context>
You are assisting an ongoing research project that applies NSGA-II to a multi-objective humanitarian logistics optimization problem.
The problem decisions are encoded in a 7-part chromosome representing distribution center assignments, shelter flows, damage-point assignments, and transport/triage matrices for severe and minor casualties.
The optimization aims to minimize three objectives: F1 (Total Distance), F2 (Unmet Demand), F3 (Death Probability).
The run configuration and problem scale (population size, number of generations, numbers of facilities/nodes) are provided below.
<context>

<Role>
You are an expert advisor (algorithm designer / adaptive operator selector) for multi-objective evolutionary algorithms, specialized in NSGA-II and adaptive operator control.
Your job is to analyze periodic performance and section-wise diversity diagnostics and recommend **per-section** crossover, mutation, and a selection method — plus tuned probabilities — for the next 5 generations.
Your recommendations should be actionable (method names are drawn from the available-method lists) and justified by the diagnostics (e.g., low variance or entropy → increase exploration on that section).
<Role>

<Instruction>
Using the supplied problem information, method catalogs, and periodic metrics, produce an adaptive operator plan for the next five generations.
Your response must be a single JSON object (see the "Output Example" below).
Focus on:
- Per-part crossover method
- Per-part mutation method
- Global crossover probability and mutation probability
- Per-gene (inner) mutation rate if applicable
- A single selection method
- Elitism rate (elitism_rate): percentage of best individuals to preserve (0.05 to 0.3, typically 0.1-0.2)
- Short analysis text (why you chose these operators, and what triggers should change them in the future)
 - Per-part probabilities (optional): crossover_part_probability, mutation_part_probability
Be concise but include a technical justification (1–3 sentences) for each major choice.
When reasoning about which operator to prefer for a part, use the part data type: discrete/permutation (Parts 1 & Part 3 section 1), continuous (Part 2), matrix-continuous (Parts 4–7), and the special two-section layout of Part 3.
<Instruction>

<Specification>
Problem Information:
- Problem Type: Humanitarian Logistics Optimization
- Number of Objectives: 3 (F1: Total Distance, F2: Unmet Demand, F3: Death Probability)
- Chromosome Structure: 7 parts
- Number of Generations: 40
- Population Size: 150
- Number of Distribution Centers: 3
- Number of Candidate Shelters: 11
- Number of Damage Points: 5
- Number of Hospitals: 4
- Number of Candidate Temporary Medical Centers: 10

Chromosome Structure (7 parts):
- Note on part sizes:
    1. The length of Part 1, Part 2, and Part 3 = number_of_candidate_shelters.
    2. Rows of Part 4 and Part 5 = number_of_damage_points; columns = number_of_hospitals.
    3. Rows of Part 6 and Part 7 = number_of_damage_points; columns = number_of_hospitals + number_of_candidate_temporary_med_centers.
Parts:
1. Part 1: Distribution center assignment to shelters (integer list) — discrete categorical (0..N_centers)
2. Part 2: Flow values from distribution centers (float list in [0,1]) — continuous
3. Part 3: Damage point to shelter assignment (integer list)
   - Part 3 Section 1 (indices 0..number_of_damage_points-1): permutation of damage-point indices (no duplicates)
   - Part 3 Section 2 (indices number_of_damage_points..number_of_shelters-1): free assignment (integers, duplicates allowed)
4. Part 4: Severe injured to hospital assignment (float matrix) — rows=damage points, cols=hospitals (values in {
            0
        } or [0,1])
5. Part 5: Percentage of severe injured transported by ambulance (float matrix) — same dims as Part 4 (values in [0,1])
6. Part 6: Minor injured to hospitals + temporary centers assignment (float matrix)
7. Part 7: Percentage of minor injured transported by ambulance (float matrix)

Available Methods:
15 different Crossover Methods:
- one_point_crossover_list    (compatible only for Part 1, Part 2, Part 3 Section 2)
- two_point_crossover_list    (compatible only for Part 1, Part 2, Part 3 Section 2)
- multi_point_crossover_list  (compatible only for Part 1, Part 2, Part 3 Section 2)
- uniform_crossover_list      (compatible only for Part 1, Part 2, Part 3 Section 2)
- order_crossover_list        (compatible only for Part 3 Section 1)
- partially_mapped_crossover  (compatible only for Part 3 Section 1)
- arithmetic_crossover_list   (compatible only for Part 2)
- blend_crossover_list        (compatible only for Part 2)
- one_point_crossover_matrix  (compatible only for Parts 4, 5, 6, 7)
- two_point_crossover_matrix  (compatible only for Parts 4, 5, 6, 7)
- uniform_crossover_matrix    (compatible only for Parts 4, 5, 6, 7)
- block_crossover_matrix      (compatible only for Parts 4, 5, 6, 7)
- row_wise_crossover_matrix   (compatible only for Parts 4, 5, 6, 7)
- arithmetic_crossover_matrix (compatible only for Parts 4, 5, 6, 7)
- simulated_binary_crossover  (compatible only for Part 2)

16 different Mutation Methods:
- bit_flip_mutation_list         (compatible only for Part 1)
- swap_mutation_list             (compatible only for Part 1, Part 3 Section 1, Part 3 Section 2)
- inversion_mutation_list        (compatible only for Part 1, Part 3 Section 1, Part 3 Section 2)
- scramble_mutation_list         (compatible only for Part 1, Part 3 Section 1, Part 3 Section 2)
- insertion_mutation_list        (compatible only for Part 1, Part 3 Section 1, Part 3 Section 2)
- displacement_mutation_list     (compatible only for Part 1, Part 3 Section 1, Part 3 Section 2)
- gaussian_mutation_list         (compatible only for Part 2)
- uniform_mutation_list          (compatible only for Part 2)
- polynomial_mutation_list       (compatible only for Part 2)
- boundary_mutation_list         (compatible only for Part 2)
- random_element_mutation_matrix (compatible only for Parts 4, 5, 6, 7)
- gaussian_mutation_matrix       (compatible only for Parts 4, 5, 6, 7)
- row_mutation_matrix            (compatible only for Parts 4, 5, 6, 7)
- column_mutation_matrix         (compatible only for Parts 4, 5, 6, 7)
- block_mutation_matrix          (compatible only for Parts 4, 5, 6, 7)
- creep_mutation_matrix          (compatible only for Parts 4, 5, 6, 7)

7 different Selection Methods:
- crowded_binary_tournament
- adaptive_k_tournament
- feasibility_first_tournament
- rank_based_roulette
- reference_biased_tournament
- age_diversity_tournament
- epsilon_dominance_tournament

*Be sure to select the merge and mutate methods for each section if they are compatible.*
<Specification>


<Performance>
what you will receive and how to use it:
Every 5 generations I will provide you a metrics payload containing:
- Global metrics per generation: Hypervolume (HV), Spacing, Spread, Number of Pareto solutions, Average crowding distance, Population diversity
- Population objective statistics: Mean, Min, and Standard deviation of objectives (F1, F2, F3) calculated from the entire population of each generation
- Section-state (for each part or section): variance (for continuous parts/matrices) or entropy/uniqueness (for discrete/permutation parts))
- Operator performance stats for the last interval (usage ratios and success rates per operator), if available
Use the variance/entropy of each chromosome segment to identify which segments are convergent or still widely dispersed:
- Low variance (continuous) or low entropy (discrete/permutation) → indicates convergence on that section → increase exploration (stronger mutation, more disruptive crossover) for that section.
- High variance/entropy → indicates wide exploration or noise → favor exploitative, smoothing operators for that section.
<Performance>

The data you need to analyze and based on that, suggest the things I wanted for the next 5 generations:
* Performance Metrics History (from generation 0 to {generation}):
* Current generation: {generation}
* Metrics History (Metric history from the first run of the algorithm to the current run):
{self._format_metrics_history(metrics_history)}
* History of methods and their possibilities: {current_methods.get("history", {})}
* Latest Generation Metrics:
- Hypervolume (HV): {metrics.get("hypervolume", 0):.6f}
- Spacing: {metrics.get("spacing", 0):.6f}
- Spread: {metrics.get("spread", 0):.6f}
- Number of Pareto solutions: {metrics.get("pareto_count", 0)}
- Average crowding distance: {metrics.get("avg_crowding_distance", 0):.6f}
- Population diversity: {metrics.get("diversity", 0):.6f}
- Elitism rate: {metrics.get("elitism_rate", 0):.4f} ({metrics.get("elitism_rate", 0) * 100:.1f}% of population preserved as elite)
- variance/entropy of parts: {metrics.get("section_stats", {})}
- Population objective statistics (from entire population):
  * Mean objectives: F1={metrics.get("mean_objectives", [0,0,0])[0]:.6f}, F2={metrics.get("mean_objectives", [0,0,0])[1]:.6f}, F3={metrics.get("mean_objectives", [0,0,0])[2]:.6f}
  * Min objectives: F1={metrics.get("min_objectives", [0,0,0])[0]:.6f}, F2={metrics.get("min_objectives", [0,0,0])[1]:.6f}, F3={metrics.get("min_objectives", [0,0,0])[2]:.6f}
  * Std objectives: F1={metrics.get("std_objectives", [0,0,0])[0]:.6f}, F2={metrics.get("std_objectives", [0,0,0])[1]:.6f}, F3={metrics.get("std_objectives", [0,0,0])[2]:.6f}

Why section statistics (entropy/variance) matter:
- Entropy (for discrete & permutation parts) measures how many distinct allele values exist and how balanced their frequencies are. Low entropy → most individuals share the same allele(s) → risk of premature convergence on that decision subspace.
- Variance (for continuous parts/matrices) measures spread of values. Low variance → convergence; high variance → noisy exploration.

Behavioral rules & constraints:
- If any discrete/permutation part shows very low entropy (e.g., entropy < 0.2) recommend stronger permutation-preserving exploration (e.g., partially_mapped_crossover + swap/inversion mutations) for that section.
- If any continuous/matrix part has extremely low variance (e.g., variance < 1e-4) recommend aggressive continuous mutations (polynomial_mutation_list or gaussian_mutation_matrix) and higher crossover disruption for that section.
- Prefer feasibility_first_tournament or epsilon_dominance_tournament when feasibility ratio is low or constraints are tight.
- Prefer age_diversity_tournament or reference_biased_tournament to maintain diversity if diversity metric drops below 0.25.
- Elitism rate: The percentage of best individuals preserved from generation to generation (range: 0.05 to 0.3, typically 0.1-0.2). Lower values (0.05-0.1) increase exploration but may slow convergence. Higher values (0.2-0.3) preserve more good solutions but may reduce diversity. Adjust based on convergence speed and diversity metrics.


Provide your response in the following JSON format (Try not to use capital letters in the keys of this output):
```json
{{
    "analysis": "Brief analysis of current situation and recommendations",
    "recommendations": {{
            "crossover_methods": {{
                "part1": "crossover_method_name",
            "part2": "crossover_method_name",
            "part3_p1": "crossover_method_name",
            "part3_p2": "crossover_method_name",
            "part4": "crossover_method_name",
            "part5": "crossover_method_name",
            "part6": "crossover_method_name",
            "part7": "crossover_method_name",
            "global_crossover_probability": 0.0,
            "crossover_part_probability": {{
                    "part1": ,
                            "part2": 0.0,
                            "part3_p1": 0.0,
                            "part3_p2": 0.0,
                            "part4": 0.0,
                            "part5": 0.0,
                            "part6": 0.0,
                            "part7": 0.0,
                }}
            }},
        "mutation_methods": {{
                "part1": "mutation_method_name",
            "part2": "mutation_method_name",
            "part3_p1": "mutation_method_name",
            "part3_p2": "mutation_method_name",
            "part4": "mutation_method_name",
            "part5": "mutation_method_name",
            "part6": "mutation_method_name",
            "part7": "mutation_method_name",
            "global_mutation_probability": 0.0,
            "mutation_inner_rate": 0.0,
            "mutation_part_probability": {{
                    "part1": 0.0,
                            "part2": 0.0,
                            "part3_p1": 0.0,
                            "part3_p2": 0.0,
                            "part4": 0.0,
                            "part5": 0.0,
                            "part6": 0.0,
                            "part7": 0.0,
                    }}
            }},
        "selection_method": "selection_method_name",
        "elitism_rate": 0.1
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
            # Enable built-in reasoning on models that support it
            # "reasoning": {"effort": "high"},
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
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")

            # برخی مدل‌ها محتوای چندبخشی یا غیررشته‌ای می‌دهند
            if isinstance(content, list):
                # ترکیب قطعات متنی
                content = "\n".join([str(part.get("text", part)) for part in content])
            elif not isinstance(content, str):
                content = str(content)

            import re

            # 1) اول تلاش برای یافتن بلاک های ```json ... ```
            fenced = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", content)
            if fenced:
                # اگر چند بلاک بود، آنی که شامل کلید recommendations است ترجیح دارد
                for block in fenced:
                    try:
                        obj = json.loads(block)
                        if isinstance(obj, dict) and (
                            "recommendations" in obj
                            or "crossover" in obj
                            or "mutation" in obj
                        ):
                            return obj
                    except json.JSONDecodeError:
                        continue
                # در غیر این صورت اولین بلاک را امتحان می‌کنیم
                try:
                    return json.loads(fenced[0])
                except json.JSONDecodeError:
                    pass

            # 2) تلاش برای پارس مستقیم کل متن (اگر پاسخ فقط JSON باشد)
            try:
                obj = json.loads(content)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

            # 3) تلاش برای استخراج اولین آبجکت JSON با شمارش براکت‌ها
            start_idx = content.find("{")
            while start_idx != -1:
                brace = 0
                for end_idx in range(start_idx, len(content)):
                    ch = content[end_idx]
                    if ch == "{":
                        brace += 1
                    elif ch == "}":
                        brace -= 1
                        if brace == 0:
                            candidate = content[start_idx : end_idx + 1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict):
                                    return obj
                            except json.JSONDecodeError:
                                break
                start_idx = content.find("{", start_idx + 1)

            print("Could not extract JSON from AI response")
            print(f"AI response: {content}")
            return None

        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing AI response: {e}")
            return None

    def get_recommendations(
        self,
        generation: int,
        metrics: Dict[str, float],
        current_methods: Dict[str, Any],
        metrics_history: list[Dict[str, Any]],
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
        prompt = self.create_prompt(generation, metrics, current_methods, metrics_history)

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
            if "crossover" in rec and "part_probability" in rec["crossover"]:
                pp = rec["crossover"]["part_probability"]
                if not isinstance(pp, dict):
                    print("⚠️ Invalid crossover part_probability: not a dict")
                    return False
                for k, v in pp.items():
                    if v is None:
                        continue
                    if not (0 <= float(v) <= 1):
                        print(f"⚠️ Invalid crossover part_probability for {k}: {v}")
                        return False

            if "mutation" in rec and "probability" in rec["mutation"]:
                prob = rec["mutation"]["probability"]
                if not (0 <= prob <= 1):
                    print(f"⚠️ Invalid mutation probability: {prob}")
                    return False
            if "mutation" in rec and "part_probability" in rec["mutation"]:
                pp = rec["mutation"]["part_probability"]
                if not isinstance(pp, dict):
                    print("⚠️ Invalid mutation part_probability: not a dict")
                    return False
                for k, v in pp.items():
                    if v is None:
                        continue
                    if not (0 <= float(v) <= 1):
                        print(f"⚠️ Invalid mutation part_probability for {k}: {v}")
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
