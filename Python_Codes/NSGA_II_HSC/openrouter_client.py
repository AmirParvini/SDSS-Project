import requests
import json
import time
import os
from typing import Dict, Any, Optional
from copy import deepcopy
import re

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI library not installed. Install it with: pip install openai")

class OpenRouterClient:
    """
    کلاینت برای ارتباط با OpenRouter API برای بهبود الگوریتم NSGA-II

    این کلاس مسئولیت‌های زیر را دارد:
    1. ارسال اطلاعات مسئله و کروموزوم به AI
    2. ارسال شاخص‌های عملکرد (HV, Spacing, etc.) هر 5 نسل
    3. دریافت پیشنهادات AI برای متدهای ترکیب، جهش و انتخاب
    4. پارس کردن پاسخ AI و تبدیل به فرمت قابل استفاده
    """

    def __init__(self, request_type:str, api_key: str, model: str, base_url:str, reasoning=None, store=False):
        """
        Constructor برای OpenRouter Client

        Parameters:
        -----------
        api_key: کلید API از OpenRouter
        model: مدل AI مورد استفاده (پیش‌فرض: مدل رایگان Llama)
        """
        self.request_type = request_type
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "NSGA-II Optimization",  # اختیاری
        }
        self.reasoning = reasoning
        self.store = store
        # برای OpenAI: ذخیره previous_response_id برای ادامه چت فقط برای مدل‌های GPT
        self.is_gpt_model = "gpt" in self.model.lower()
        self.previous_response_id_file = None
        self.previous_response_id = None
        self.openai_client = None
        if OPENAI_AVAILABLE:
            # استفاده از OpenAI با API key از OpenRouter
            self.openai_client = OpenAI(
                api_key=api_key,
                base_url=self.base_url
            )
            if self.is_gpt_model:
                # بارگذاری previous_response_id از فایل JSON فقط برای مدل‌های GPT
                self.previous_response_id_file = os.path.join(os.path.dirname(__file__), "exports/openrouter_previous_response_id.json")
                self.previous_response_id = self.load_previous_response_id()

    def create_init_pop_prompt(self, pop_size: int, dims: Dict[str, int], ids: Dict[str, list], distances: Dict[str, Any],
                               homeless: Dict[int, float],sever_injured:Dict[int, float], minor_injured:Dict[int, float], cost: Dict[str, Any], capacity: Dict[str, Any]) -> str:
        """
        Build a strict prompt asking the model to generate an initial population.

        dims must include: n_shelters, n_distribution, n_damage_points, n_hospitals, n_temp_medical.
        """
        n_shelters = dims["n_shelters"]
        n_distribution = dims["n_distribution"]
        n_damage_points = dims["n_damage_points"]
        n_hospitals = dims["n_hospitals"]
        n_temp_medical = dims["n_temp_medical"]

        # Injected problem context for AI seeding
        idc_ids = ids.get("idc_id", [])
        ec_ids = ids.get("ec_id", [])
        da_ids = ids.get("da_id", [])
        h_ids = ids.get("h_id", [])
        tmc_ids = ids.get("tmc_id", [])
        ids_str = ",".join(str(x) for x in da_ids)
        import json as _json
        distances_json = _json.dumps(distances, ensure_ascii=False)
        homeless_json = _json.dumps(homeless, ensure_ascii=False)
        sever_injured_json = _json.dumps(sever_injured, ensure_ascii=False)
        minor_injured_json = _json.dumps(minor_injured, ensure_ascii=False)
        cost_json = _json.dumps(cost, ensure_ascii=False)
        capacity_json = _json.dumps(capacity, ensure_ascii=False)

        prompt = {
            "system": f"You are generating the INITIAL POPULATION for an NSGA-II run on a humanitarian logistics problem. Your goal is to create a **feasible, diverse, and well-seeded initial population** that respects all structural and logical constraints of the chromosome design.Return a SINGLE JSON object with a key 'population' that is a list of length {pop_size}.",
            "user": f""" 
Problem Information:
- Problem Type: Humanitarian Logistics Optimization
- Number of Objectives: 3 (Min F1: Total distances of homeless people assigned to selected shelters, Min F2: Unmet Demand, Min F3: Death Probability)
- Chromosome Structure: 7 parts
- Number of Generations: 300
- Population Size: 150
- Number of Distribution Centers: 3
- Number of Candidate Shelters: 11
- Number of Damage Points: 5
- Number of Hospitals: 4
- Number of Candidate Temporary Medical Centers: 10

Problem Data (IDs, distances, and demand context):
- Distribution Center IDs (idc_id): {idc_ids}
- Shelter IDs (ec_id): {ec_ids}
- Damage Point IDs (da_id): {da_ids}
- Hospital IDs (h_id): {h_ids}
- Temporary Medical Center IDs (tmc_id): {tmc_ids}
- Distances (JSON): {distances_json}
- Homeless counts per damage point (JSON): {homeless_json}
- sever injured counts per damage point (JSON): {sever_injured_json}
- minor injured counts per damage point (JSON): {minor_injured_json}
- Cost parameters (JSON): {cost_json}
- Capacity parameters (JSON): {capacity_json}

Chromosome parts and strict shapes:
1) part1 (length={n_shelters}): integer list in [0,{n_distribution}] where 0 means shelter not selected and 1..{n_distribution} is the distribution center ID.
2) part2 (length={n_shelters}): float list in [0,1] for flow ratios.
3) part3 (length={n_shelters}): integer list. Indices 0..{n_damage_points-1} must be a permutation of the {n_damage_points} distinct damage point IDs drawn from the following set: {{{ids_str}}}. Indices {n_damage_points}..{n_shelters-1} can be any element from the same ID set and 0 (duplicates allowed).
4) part4 (shape={n_damage_points}x{n_hospitals}): float matrix in [0,1] with at least one positive per row; rows need be normalized.
5) part5 (shape={n_damage_points}x{n_hospitals}): float matrix in [0,1].
6) part6 (shape={n_damage_points}x{n_hospitals + n_temp_medical}): float matrix in [0,1] with at least one positive per row; rows need be normalized.
7) part7 (shape={n_damage_points}x{n_hospitals + n_temp_medical}): float matrix in [0,1].

How to allocate in chromosome segments:
part1: An array that specifies which distribution center each shelter is supplied from. Represents the number of candidate shelter locations, whose values ​​are either zero or a random number between the distribution center indices.
part2: A percentage of the demand for active shelters. In this part of the Chromosome, where the allocations of populations to shelters were determined and the demand was calculated accordingly, the values ​​of the elements in the second part determine how much of the demand for the desired shelter should be met. For example, if the second index receives a value of 0.7 and if this index was non-zero in the first part, it means that 0.7 of the demand for that shelter will be shipped.
part3: for each non-zero element in part1, assign the next available damage point from part3, skipping zeros in part1. If the number of active shelters exceeds the number of damage points, continue cyclically from the start of part3 until all active shelters have an assigned damage point. For example, if part1 = [0,2,0,1,3,0] and part3 = [4,7,3,6,5,7], the assignments are (2→4), (1→7), (3→3); and if part1 = [3,1,2,3,1,2,3] and part3 = [4,6,7,5,3], the assignments are (3→4), (1→6), (2→7), (3→5), (1→3), (2→4), (3→6).”
part4: is a matrix with rows equal to the number of affected points and columns equal to the number of hospitals. The values ​​of the elements in each row are either zero or a number between 0 and 1. The sum of the values ​​in each row must equal 1 to ensure that all injured people in the affected point are transported to the hospital of interest.
part5: Percentage of seriously injured people transported to hospital by ambulance. The sum of these values ​​determines the percentage of people transported by helicopter. Any row values ​​whose index is equal to the index of zero values ​​in part4 are ineffective because the desired hospital was not selected.
part6: There is a matrix with rows equal to the number of hospitals and candidate locations for temporary care facilities. This means that minor injuries can be transferred to both hospitals and temporary care facilities. The values ​​in this matrix are similar to those in part4.
part7: Same as part5

The impact of each chromosome segment on objs functions:
A fixed budget is considered for the problem. 
Part1 has a direct effect on the budget. By allocating nearby shelters, the costs of transporting relief items are also reduced, but it may increase F1. On the other hand, the greater the number of non-zero elements, the higher the cost of establishing the shelter, but it can have a positive effect on F2.
Part2 has a direct effect on F2. Increasing the values ​​of this part reduces F2 but increases the costs of transporting items. 
Part3 has a direct effect on F1. By allocating damaged points to nearby shelters, F1 is reduced, but it may increase the distance from the distribution centers to the shelters, and as a result, the cost increases. 
Part4 has a direct effect on F3. Allocating damaged points to nearby hospitals reduces the time for transporting the injured and reduces F3, but it can still increase it due to the lack of hospital capacity.
Part5 has a direct effect on F3. High values ​​of elements indicate that a large percentage of the injured are transported by ambulance and a small percentage of the injured are transported by helicopter. Transporting by ambulance may increase the time it takes to transport the injured and increase F3, but it uses less budget, but the helicopter is the opposite.
Part6 and part7 are the same as part 4 and part 5, except that large values ​​of 0 reduce budget consumption but may increase F3.

Note:
*Carefully examine the problem data provided to you, considering the impact of each chromosome part on the budget and objective functions, and analyze and use them to create the initial population.

Follow these design principles strictly:
1️⃣ **Feasibility first**
- Every chromosome must fully satisfy all problem constraints (capacity, flow balance, assignment rules, logical structure).
- If a generated chromosome violates a constraint, repair it immediately (e.g., normalize rows, reassign excess, fix duplicates, ensure each row has at least one positive value).
- At least 80% of the population should be feasible from the start.

2️⃣ **High diversity**
- Use stratified sampling:
  - For continuous variables → use Latin Hypercube or Sobol sampling in [0,1].
  - For discrete or categorical variables → sample uniformly while avoiding duplicates.
  - For permutation sections → use Fisher–Yates shuffling plus a few heuristic orderings.
- Remove any near-identical individuals (Euclidean or Hamming distance threshold).
- Include boundary and mid-range individuals to cover the decision space broadly.

3️⃣ **Intelligent seeding**
- Insert a few heuristic individuals derived from the problem logic, e.g.:
  - “Minimum-distance” or “nearest assignment”
  - “Capacity-balanced” or “min-unmet-demand”
  - “Uniform distribution of flows”
- Include at least one “extreme” solution per objective (favoring one objective strongly while ignoring others) to ensure corner coverage on the Pareto front.

4️⃣ **Structure-aware generation**
- For each chromosome section:
  - **Binary/multiclass selection part:** ensure valid selection counts and diversity.
  - **Continuous ratio/matrix part:** ensure ≥0 and normalized where required.
  - **Permutation part:** guarantee valid ordering of IDs.
  - **Matrix sections (e.g., flows):** each row must contain at least one positive entry.

5️⃣ **Controlled randomness and reproducibility**
- Use a fixed random seed for consistency.
- Population size: between 50–200 (or about 4–10× number of decision variables).

6️⃣ **Quality check before evolution**
- Evaluate all objectives for the initial population.
- Print the percentage of feasible individuals and diversity metrics (spread/spacing).
- If diversity < threshold or feasibility < 60%, resample and repair again.

Hard constraints:
- Respect all lengths and shapes exactly.
- Use only numeric values (no null/NaN/strings).
- Keep values within the specified ranges.

Recommendation:
Use the data and explanations about the impact of each chromosome segment on the target functions and form the initial population based on the analysis of these segments.

Output format (no extra text):
{{
  "population": [
    {{
      "part1": [ints...],
      "part2": [floats...],
      "part3": [ints...],
      "part4": [[floats...], ...],
      "part5": [[floats...], ...],
      "part6": [[floats...], ...],
      "part7": [[floats...], ...]
    }},
    ... {pop_size} items total ...
  ]
}}
"""
        }
        # Ensure DA IDs are correctly embedded in the permutation specification
        return prompt

    def parse_init_population(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract a JSON object that contains a key "population" (list of chromosomes).
        """
        try:
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join([str(part.get("text", part)) for part in content])
            elif not isinstance(content, str):
                content = str(content)

            import re
            import json

            fenced = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", content)
            candidates = []
            if fenced:
                candidates.extend(fenced)
            # try raw content as well
            candidates.append(content)

            for cand in candidates:
                try:
                    obj = json.loads(cand)
                    if isinstance(obj, dict) and isinstance(obj.get("population"), list):
                        return obj
                except json.JSONDecodeError:
                    continue
            return None
        except Exception:
            return None

    def get_initial_population(self, pop_size: int, dims: Dict[str, int], ids: Dict[str, list],
                               distances: Dict[str, Any], homeless: Dict[int, float], severe_injured: Dict[int, float], minor_injured: Dict[int, float],
                               cost: Dict[str, Any], capacity: Dict[str, Any], max_retries: int = 3) -> Optional[list]:
        """
        Request an initial population from the AI. Returns a list of chromosome dicts
        or None on failure.
        """
        # Build prompt with actual DA IDs injected
        print('get_initial_population...')
        prompt = self.create_init_pop_prompt(pop_size, dims, ids, distances, homeless, severe_injured, minor_injured, cost, capacity)
        
        print('send_request...')
        response = self.send_request_openai(prompt)
        if response is None:
            return None

        print('parse_init_population...')
        parsed = self.parse_init_population(response)
        if not parsed:
            return None

        population = parsed.get("population")
        if not isinstance(population, list):
            return None
        return population

    def create_prompt(
        self,
        **prompt_args
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
        metrics_history: Dict = prompt_args.get('metrics_history',{})
        llm_iter: int = prompt_args.get('llm_iter',{})
        generation: int = prompt_args.get('it',{})
        offspring_survival_history: Dict = prompt_args.get('offspring_survival_history',{})
        current_methods: Dict = prompt_args.get('current_methods',{})
        section_stats_history: Dict = prompt_args.get('section_stats_history',{})
        dims: Dict = prompt_args.get('dims',{})
        n_shelters = dims["n_shelters"]
        n_distribution = dims["n_distribution"]
        n_damage_points = dims["n_damage_points"]
        n_hospitals = dims["n_hospitals"]
        n_temp_medical = dims["n_temp_medical"]
        
        # Helper function to safely get last value from list
        def get_last_value(history_list, default_value=0.0):
            if history_list and len(history_list) > 0:
                return history_list[-1]
            return default_value
        
        # Helper function to safely get objective values
        def get_objective_value(history_list, obj_index, default_value=0.0):
            if history_list and len(history_list) > 0:
                last_obj = history_list[-1]
                if isinstance(last_obj, (list, tuple)) and len(last_obj) > obj_index:
                    return last_obj[obj_index]
            return default_value
        
        # Extract last values for objectives
        mean_objs = metrics_history.get("mean_objectives", [])
        min_objs = metrics_history.get("min_objectives", [])
        std_objs = metrics_history.get("std_objectives", [])
        
        mean_f1 = get_objective_value(mean_objs, 0)
        mean_f2 = get_objective_value(mean_objs, 1)
        mean_f3 = get_objective_value(mean_objs, 2)
        
        min_f1 = get_objective_value(min_objs, 0)
        min_f2 = get_objective_value(min_objs, 1)
        min_f3 = get_objective_value(min_objs, 2)
        
        std_f1 = get_objective_value(std_objs, 0)
        std_f2 = get_objective_value(std_objs, 1)
        std_f3 = get_objective_value(std_objs, 2)
        
        prompt = {
            "system": f"""
You are an expert consultant (adaptive operator selector) for multi-objective evolutionary algorithms specializing in NSGA-II and adaptive operator control. Your task is to analyze the periodic performance of the received metrics and detect diversity in each chromosome part and recommend crossover, mutation and their probabilities and rates and values ​​of input arguments for each part and a selection method for the next generations {llm_iter}.
Your recommendations should be actionable (method names are extracted from the list of *Available Methods*) and justified by the diagnostics (e.g. low variance or entropy → increase exploration in that part). 
""",
            "user": f""" 
You are assisting an ongoing research project that applies NSGA-II to a multi-objective humanitarian logistics optimization problem.
The problem decisions are encoded in a 7-part chromosome representing distribution center assignments, shelter flows, damage-point assignments, and transport/triage matrices for severe and minor casualties.
The optimization aims to minimize three objectives: F1, F2, F3.
The run configuration and problem scale (population size, number of generations, numbers of facilities/nodes) are provided below.

=== PROBLEM DESCRIPTION ===
- Problem Type: Humanitarian Logistics Optimization
- Number of Objectives: 3 (Min F1: Total distances of homeless people assigned to selected shelters, Min F2: Unmet Demand, Min F3: Death Probability)
- Chromosome Structure: 7 parts
- Number of Generations: 300
- Population Size: 150
- Number of Distribution Centers: 3
- Number of Candidate Shelters: 11
- Number of Damage Points: 5
- Number of Hospitals: 4
- Number of Candidate Temporary Medical Centers: 10

=== CHROMOSOM STRUCTURE ===
Chromosome parts and strict shapes:
1) part1 (length={n_shelters}): integer list in [0,{n_distribution}] where 0 means shelter not selected and 1..{n_distribution} is the distribution center ID.
2) part2 (length={n_shelters}): float list in [0,1] for flow ratios.
3) part3 (length={n_shelters}): integer list. Indices 0..{n_damage_points-1} must be a permutation of the {n_damage_points} distinct damage point IDs drawn from the following set: [3,4,5,6,7]. Indices {n_damage_points}..{n_shelters-1} can be any element from the same ID set and 0 (duplicates allowed).
4) part4 (shape={n_damage_points}x{n_hospitals}): float matrix in [0,1] with at least one positive per row; rows need be normalized.
5) part5 (shape={n_damage_points}x{n_hospitals}): float matrix in [0,1].
6) part6 (shape={n_damage_points}x{n_hospitals + n_temp_medical}): float matrix in [0,1] with at least one positive per row; rows need be normalized.
7) part7 (shape={n_damage_points}x{n_hospitals + n_temp_medical}): float matrix in [0,1].

How to allocate in chromosome segments:
part1: An array that specifies which distribution center each shelter is supplied from. Represents the number of candidate shelter locations, whose values ​​are either zero or a random number between the distribution center indices.
part2: A percentage of the demand for active shelters. In this part of the Chromosome, where the allocations of populations to shelters were determined and the demand was calculated accordingly, the values ​​of the elements in the second part determine how much of the demand for the desired shelter should be met. For example, if the second index receives a value of 0.7 and if this index was non-zero in the first part, it means that 0.7 of the demand for that shelter will be shipped.
part3: for each non-zero element in part1, assign the next available damage point from part3, skipping zeros in part1. If the number of active shelters exceeds the number of damage points, continue cyclically from the start of part3 until all active shelters have an assigned damage point. For example, if part1 = [0,2,0,1,3,0] and part3 = [4,7,3,6,5,7], the assignments are (2→4), (1→7), (3→3); and if part1 = [3,1,2,3,1,2,3] and part3 = [4,6,7,5,3], the assignments are (3→4), (1→6), (2→7), (3→5), (1→3), (2→4), (3→6).”
part4: is a matrix with rows equal to the number of affected points and columns equal to the number of hospitals. The values ​​of the elements in each row are either zero or a number between 0 and 1. The sum of the values ​​in each row must equal 1 to ensure that all injured people in the affected point are transported to the hospital of interest.
part5: Percentage of seriously injured people transported to hospital by ambulance. The sum of these values ​​determines the percentage of people transported by helicopter. Any row values ​​whose index is equal to the index of zero values ​​in part4 are ineffective because the desired hospital was not selected.
part6: There is a matrix with rows equal to the number of hospitals and candidate locations for temporary care facilities. This means that minor injuries can be transferred to both hospitals and temporary care facilities. The values ​​in this matrix are similar to those in part4.
part7: Same as part5

The impact of each chromosome segment on objs functions:
A fixed budget is considered for the problem. 
Part1 has a direct effect on the budget. By allocating nearby shelters, the costs of transporting relief items are also reduced, but it may increase F1. On the other hand, the greater the number of non-zero elements, the higher the cost of establishing the shelter, but it can have a positive effect on F2.
Part2 has a direct effect on F2. Increasing the values ​​of this part reduces F2 but increases the costs of transporting items. 
Part3 has a direct effect on F1. By allocating damaged points to nearby shelters, F1 is reduced, but it may increase the distance from the distribution centers to the shelters, and as a result, the cost increases. 
Part4 has a direct effect on F3. Allocating damaged points to nearby hospitals reduces the time for transporting the injured and reduces F3, but it can still increase it due to the lack of hospital capacity.
Part5 has a direct effect on F3. High values ​​of elements indicate that a large percentage of the injured are transported by ambulance and a small percentage of the injured are transported by helicopter. Transporting by ambulance may increase the time it takes to transport the injured and increase F3, but it uses less budget, but the helicopter is the opposite.
Part6 and part7 are the same as part 4 and part 5, except that large values ​​of 0 reduce budget consumption but may increase F3.

Instruction:
Using the problem information, method catalogs, and periodic criteria provided, generate an adaptive operator scheme for the next generations {llm_iter}.
In other words, you will play the role of an intelligent Multi-Armed Bandit that will suggest appropriate methods and values ​​for the next generations based on both the analysis of the historical performance of the methods and their rates, probabilities, and input argument values, as well as your own knowledge and understanding of them.
Be concise but include a technical justification (1–3 sentences) for each major choice.

what you will receive and how to use it:
Every {llm_iter} generations I will provide you a metrics payload containing:
- Hostory of the names of the methods used in each generation
- History of input argument values ​​for crossover, mutation, and selection methods
- History of the global probability of crossover and mutation methods in each generation
- History of the global probability of selection method in each generation
- History of the probability of crossover and mutation of each part of the chromosome in each generation
- History of the global metrics per generation: Hypervolume (HV), Spacing, Spread, Number of Pareto solutions, Average crowding distance
- Population objective statistics: Mean, Min, and Standard deviation of objectives (F1, F2, F3) calculated from the entire population of each generation
- Section-state (For each part of the chromosome at the gene level): variance (for part 2,4,5,6,7) or entropy/uniqueness (part 1,3)
- The number of offspring produced from the current generation that remain in the new population.
Use the variance/entropy of each chromosome segment to identify which segments are convergent or still widely dispersed:
- Low variance (continuous) or low entropy (discrete/permutation) → indicates convergence on that section → increase exploration (stronger mutation, more disruptive crossover) for that section.
- High variance/entropy → indicates wide exploration or noise → favor exploitative, smoothing operators for that section.

Note: 
*The key values ​​are stored in the method history as a list, and the value of each element is used for {llm_iter} generations in order, but is written only once to avoid prompt overhead.
For example, the value of the key 'global_crossover_probability_history' stored as [0.8, 0.7] means that the value 0.7 is used for generations 0 to 4 and the value 0.8 is used for generations 5 to 9.

=== DATA HISTORY ===
The data you need to analyze and based on that, suggest the things I wanted for the next {llm_iter} generations:
1. Performance Metrics History (from generation 0 to {generation}):
2. Current generation: {generation}
3. Metrics History (Metric history from the first run of the algorithm to the current run):
    - pareto_front Hypervolume history: {metrics_history.get("hypervolume", [])}
    - pareto_front Spacing history: {metrics_history.get("spacing", [])}
    - pareto_front Spread history: {metrics_history.get("spread", [])}
    - Number of Pareto solutions history: {metrics_history.get("pareto_count", [])}
    - pareto_front Average crowding distance history: {metrics_history.get("avg_crowding_distance", [])}
    - offspring survival history: {offspring_survival_history}
    - variance/entropy of parts history (The values ​​for each part are stored as tuples. for exam -> "entropy":(variance_per_gene(list), avg_variance(float))): {section_stats_history}
4. History of methods and their possibilities: {current_methods.get("history", {})}
5. Population objective statistics (from entire population):
    - Mean objectives: F1={mean_f1:.6f}, F2={mean_f2:.6f}, F3={mean_f3:.6f}
    - Min objectives: F1={min_f1:.6f}, F2={min_f2:.6f}, F3={min_f3:.6f}
    - Std objectives: F1={std_f1:.6f}, F2={std_f2:.6f}, F3={std_f3:.6f}
 
Note: 
*The default methods Selection, Crossover, and Mutation are the methods that are used by "default" for the algorithm in the early generations.
*If the algorithm performs well using these default methods, you can suggest them (i.e. use the default instead of the method name). No data from the input arguments of these methods will be sent to you.
*Review and analyze the progress of the data provided to you and use it in your suggestions.
*Argue for yourself why you chose the methods, their probabilities, rates, and input argument values.
*When specifying the values ​​of the input arguments of the methods, be careful to only assign values ​​to the input arguments of the methods you propose and display them in the output.

=== AVALIBLE METHODS ===
The methods you suggest should be selected from this method:
"crossover_methods":[
- one_point_crossover -> Use only in 'part1'&'part2'&'part3_p2'
- two_point_crossover -> Use only in 'part1'&'part2'&'part3_p2'
- multi_point_crossover -> Use only in 'part1'&'part2'&'part3_p2'
- uniform_crossover -> Use only in 'part1'&'part2'&'part3_p2'
- order_crossover -> Use only in 'part3_p1'
- partially_mapped_crossover -> Use only in 'part3_p1'
- arithmetic_crossover -> Use only in 'part2'
- blend_crossover -> Use only in 'part2'
- one_point_crossover_matrix -> Use only in 'part4'&'part5'&'part6'&'part7'
- two_point_crossover_matrix -> Use only in 'part4'&'part5'&'part6'&'part7'
- uniform_crossover_matrix -> Use only in 'part4'&'part5'&'part6'&'part7'
- block_crossover_matrix -> Use only in 'part4'&'part5'&'part6'&'part7'
- row_wise_crossover_matrix -> Use only in 'part4'&'part5'&'part6'&'part7'
- arithmetic_crossover_matrix -> Use only in 'part4'&'part5'&'part6'&'part7'
- simulated_binary_crossover -> Use only in 'part2'
]
"mutation_methods":[
- bit_flip_mutation(mutation_rate, value_range) -> Use only in 'part1' 
- swap_mutation(n_swaps) -> Use only in 'part1'&'part3_p1'&'part3_p2'
- inversion_mutation(no arguments) -> Use only in 'part1'&'part3_p1'&'part3_p2'
- scramble_mutation(no arguments) -> Use only in 'part1'&'part3_p1'&'part3_p2'
- insertion_mutation(no arguments) -> Use only in 'part1'&'part3_p1'&'part3_p2'
- displacement_mutation(no arguments) -> Use only in 'part1'&'part3_p1'&'part3_p2'
- gaussian_mutation(mutation_rate, sigma) -> Use only in 'part2'
- uniform_mutation(mutation_rate) -> Use only in 'part2'
- polynomial_mutation(mutation_rate, eta) -> Use only in 'part2'
- boundary_mutation(mutation_rate) -> Use only in 'part2'
- random_element_mutation_matrix(mutation_rate, value_range) -> Use only in 'part4'&'part5'&'part6'&'part7'
- gaussian_mutation_matrix(mutation_rate, sigma) -> Use only in 'part4'&'part5'&'part6'&'part7'
- row_mutation_matrix(mutation_rate) -> Use only in 'part4'&'part5'&'part6'&'part7'
- column_mutation_matrix(mutation_rate) -> Use only in 'part4'&'part5'&'part6'&'part7'
- block_mutation_matrix(block_size(tuple/list of length 2)) -> Use only in 'part4'&'part5'&'part6'&'part7'
- creep_mutation_matrix(mutation_rate, step_size) -> Use only in 'part4'&'part5'&'part6'&'part7'
]
"selection_methods":[
- crowded_binary_tournament(pop, k)
- adaptive_k_tournament(pop, k)
- rank_based_roulette(pop, power)
- reference_biased_tournament(pop, refs=refs, k=k, tau=tau)
- age_diversity_tournament(pop, k=k, prefer_younger=prefer_younger)
- epsilon_dominance_tournament(pop, k=k, eps=eps)
]

=== STRICT ENFORCEMENT RULES ===
1.Use the *AVALIBLE METHODS* only in the mentioned parts (for example partially_mapped_crossover -> Use only in 'part3_p1' .This means that this method should only be used in Part3_p1).
2.Avoid changing or removing characters of the method names in *AVAILABLE METHODS*. Make sure the method names you use for the parts exactly match *AVAILABLE METHODS*.
3.Avoid changing the characters in output patternt keys.
4.Be careful not to leave out the keys used in the output patternt.
5.not to use capital letters in the keys of JSON output

=== OUTPU PATTERN ===
Your output should be just a valid JSON object that follows the pattern below.:
```json
{{
"recommendations": {{
    "crossover_methods": {{
        "part1": "",
        "part2": "",
        "part3_p1": "",
        "part3_p2": "",
        "part4": "",
        "part5": "",
        "part6": "",
        "part7": "",
        }},
    "global_crossover_probability": 0.0,
    "crossover_part_probability": {{
        "part1": 0.0,
        "part2": 0.0,
        "part3_p1": 0.0,
        "part3_p2": 0.0,
        "part4": 0.0,
        "part5": 0.0,
        "part6": 0.0,
        "part7": 0.0,
        }},
    "mutation_methods": {{
        "part1": "",
        "part2": "",
        "part3_p1": "",
        "part3_p2": "",
        "part4": "",
        "part5": "",
        "part6": "",
        "part7": "",
        }},
    "global_mutation_probability": 0.0,
    "mutation_args_per_part": {{
        "part1": {{"mutation_rate": 0.05}},
        "part2": {{"mutation_rate": 0.15, "sigma": 0.2}},
        "part3_p1": {{"n_swaps": 2}},
        "part3_p2": {{"n_swaps": 1}},
        "part4": {{"mutation_rate": 0.08, "eta": 20}},
        "part5": {{"mutation_rate": 0.08}},
        "part6": {{"mutation_rate": 0.12}},
        "part7": {{"mutation_rate": 0.12}}
        }},
    "mutation_part_probability": {{
        "part1": 0.0,
        "part2": 0.0,
        "part3_p1": 0.0,
        "part3_p2": 0.0,
        "part4": 0.0,
        "part5": 0.0,
        "part6": 0.0,
        "part7": 0.0,
        }},
    "selection_method": "",
    "selection_args": {{
        "k": 2,
        "power": 1.0,
        "refs": "[[float, float, float],[float, float, float],...]",
        "tau": 1.0,
        "prefer_younger": true,
        "eps": 0.0
        }}
    }}
}}
```
"""
        }
        return prompt

    def send_request(
        self, prompt: str, max_retries: int = 3000
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
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            # Enable built-in reasoning on models that support it
            # "reasoning": {"effort": "high"},
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url, headers=self.headers, json=payload
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

    def extract_metrics_data(
        self,
        **prompt_args
    ) -> str:
        """
        استخراج بخش metrics data از prompt
        
        Parameters:
        -----------
        generation: نسل فعلی
        current_methods: متدهای فعلی
        metrics_history: تاریخچه شاخص‌های عملکرد
        offspring_survival_history: تاریخچه بقای فرزندان
        
        Returns:
        --------
        str: متن metrics data
        """
        metrics_history: Dict = prompt_args.get('metrics_history',{})
        generation: int = prompt_args.get('it',{})
        offspring_survival_history: Dict = prompt_args.get('offspring_survival_history',{})
        current_methods: Dict = prompt_args.get('current_methods',{})
        section_stats_history: Dict = prompt_args.get('section_stats_history',{})
        
        metrics_text = f"""
* Performance Metrics History (from generation 0 to {generation}):
* Current generation: {generation}
* Metrics History (Metric history from the first run of the algorithm to the current run):
    - pareto_front Hypervolume history: {metrics_history.get("hypervolume", [0])[-1] if isinstance(metrics_history.get("hypervolume"), list) else metrics_history.get("hypervolume", 0):.6f}
    - pareto_front Spacing history: {metrics_history.get("spacing", [0])[-1] if isinstance(metrics_history.get("spacing"), list) else metrics_history.get("spacing", 0):.6f}
    - pareto_front Spread history: {metrics_history.get("spread", [0])[-1] if isinstance(metrics_history.get("spread"), list) else metrics_history.get("spread", 0):.6f}
    - Number of Pareto solutions history: {metrics_history.get("pareto_count", [0])[-1] if isinstance(metrics_history.get("pareto_count"), list) else metrics_history.get("pareto_count", 0)}
    - pareto_front Average crowding distance history: {metrics_history.get("avg_crowding_distance", [0])[-1] if isinstance(metrics_history.get("avg_crowding_distance"), list) else metrics_history.get("avg_crowding_distance", 0):.6f}
    - offspring survival history: {offspring_survival_history}
    - variance/entropy of parts history (The values ​​for each part are stored as tuples. for exam -> "entropy":(variance_per_gene, avg_variance)): {section_stats_history}
* History of methods and their possibilities: {current_methods.get("history", {})}
* Population objective statistics (from entire population):
    - Mean objectives: F1={metrics_history.get("mean_objectives", [[0,0,0]])[-1][0] if isinstance(metrics_history.get("mean_objectives"), list) and len(metrics_history.get("mean_objectives", [])) > 0 else (metrics_history.get("mean_objectives", [0,0,0])[0] if isinstance(metrics_history.get("mean_objectives"), list) else 0):.6f}, F2={metrics_history.get("mean_objectives", [[0,0,0]])[-1][1] if isinstance(metrics_history.get("mean_objectives"), list) and len(metrics_history.get("mean_objectives", [])) > 0 else (metrics_history.get("mean_objectives", [0,0,0])[1] if isinstance(metrics_history.get("mean_objectives"), list) else 0):.6f}, F3={metrics_history.get("mean_objectives", [[0,0,0]])[-1][2] if isinstance(metrics_history.get("mean_objectives"), list) and len(metrics_history.get("mean_objectives", [])) > 0 else (metrics_history.get("mean_objectives", [0,0,0])[2] if isinstance(metrics_history.get("mean_objectives"), list) else 0):.6f}
    - Min objectives: F1={metrics_history.get("min_objectives", [[0,0,0]])[-1][0] if isinstance(metrics_history.get("min_objectives"), list) and len(metrics_history.get("min_objectives", [])) > 0 else (metrics_history.get("min_objectives", [0,0,0])[0] if isinstance(metrics_history.get("min_objectives"), list) else 0):.6f}, F2={metrics_history.get("min_objectives", [[0,0,0]])[-1][1] if isinstance(metrics_history.get("min_objectives"), list) and len(metrics_history.get("min_objectives", [])) > 0 else (metrics_history.get("min_objectives", [0,0,0])[1] if isinstance(metrics_history.get("min_objectives"), list) else 0):.6f}, F3={metrics_history.get("min_objectives", [[0,0,0]])[-1][2] if isinstance(metrics_history.get("min_objectives"), list) and len(metrics_history.get("min_objectives", [])) > 0 else (metrics_history.get("min_objectives", [0,0,0])[2] if isinstance(metrics_history.get("min_objectives"), list) else 0):.6f}
    - Std objectives: F1={metrics_history.get("std_objectives", [[0,0,0]])[-1][0] if isinstance(metrics_history.get("std_objectives"), list) and len(metrics_history.get("std_objectives", [])) > 0 else (metrics_history.get("std_objectives", [0,0,0])[0] if isinstance(metrics_history.get("std_objectives"), list) else 0):.6f}, F2={metrics_history.get("std_objectives", [[0,0,0]])[-1][1] if isinstance(metrics_history.get("std_objectives"), list) and len(metrics_history.get("std_objectives", [])) > 0 else (metrics_history.get("std_objectives", [0,0,0])[1] if isinstance(metrics_history.get("std_objectives"), list) else 0):.6f}, F3={metrics_history.get("std_objectives", [[0,0,0]])[-1][2] if isinstance(metrics_history.get("std_objectives"), list) and len(metrics_history.get("std_objectives", [])) > 0 else (metrics_history.get("std_objectives", [0,0,0])[2] if isinstance(metrics_history.get("std_objectives"), list) else 0):.6f}

From now on, I will only send you the history of the parameters without any additional writing, and you will return your suggestions by analyzing them carefully, completely in accordance with the output format."""
        return metrics_text

    def load_previous_response_id(self) -> Optional[str]:
        """
        بارگذاری previous_response_id از فایل JSON
        
        Returns:
        --------
        str: previous_response_id یا None در صورت عدم وجود
        """
        try:
            if os.path.exists(self.previous_response_id_file):
                with open(self.previous_response_id_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    previous_response_id = data.get("previous_response_id")
                    if previous_response_id:
                        return previous_response_id
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"⚠️ Error loading previous_response_id from file: {e}")
        return None

    def save_previous_response_id(self, previous_response_id: str) -> bool:
        """
        ذخیره previous_response_id در فایل JSON
        
        Parameters:
        -----------
        previous_response_id: previous_response_id برای ذخیره
        
        Returns:
        --------
        bool: True در صورت موفقیت، False در صورت خطا
        """
        try:
            data = {"previous_response_id": previous_response_id}
            with open(self.previous_response_id_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except IOError as e:
            print(f"⚠️ Error saving previous_response_id to file: {e}")
            return False

    def send_request_openai(
        self,
        store: bool,
        prompt: Optional[Dict[str, str]] = None,
        metrics_data: Optional[str] = None,
        max_retries: int = 30000
    ) -> Optional[Dict[str, Any]]:
        """
        ارسال درخواست به OpenRouter API با استفاده از کتابخانه OpenAI

        Parameters:
        -----------
        prompt: prompt کامل برای اولین درخواست (از create_prompt)
        metrics_data: فقط بخش metrics data برای درخواست‌های بعدی (از extract_metrics_data)
        store: آیا thread را ذخیره کند (برای ادامه چت)
        max_retries: حداکثر تعداد تلاش مجدد

        Returns:
        --------
        Dict: پاسخ AI یا None در صورت خطا
        """
        if not OPENAI_AVAILABLE or self.openai_client is None:
            print("⚠️ OpenAI library not available. Falling back to requests method.")
            if prompt:
                return self.send_request(prompt, max_retries)
            return None

        for attempt in range(max_retries):
            try:
                # اگر previous_response_id وجود دارد و مدل GPT است، از آن استفاده می‌کنیم (درخواست بعدی)
                if self.is_gpt_model and self.previous_response_id is not None and metrics_data is not None:
                    # ادامه چت با thread موجود
                    response = self.openai_client.responses.create(
                        model=self.model,
                        input=[
                            {"role": "user", "content": metrics_data}
                        ],
                        extra_headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "X-Title": "NSGA-II Optimization",  # اختیاری
                        },
                        reasoning = self.reasoning,
                        store=self.store,
                        previous_response_id=self.previous_response_id
                    )
                else:
                    # اولین درخواست: ارسال prompt کامل
                    if prompt is None:
                        print("⚠️ No prompt provided for first request")
                        return None

                    input = [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]}
                    ]

                    response = self.openai_client.responses.create(
                        model=self.model,
                        input=input,
                        extra_headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "X-Title": "NSGA-II Optimization",  # اختیاری
                        },
                        reasoning = self.reasoning,
                    )

                    # ذخیره previous_response_id برای درخواست‌های بعدی فقط برای مدل‌های GPT
                    if self.store and self.is_gpt_model:
                        if hasattr(response, 'previous_response_id') and response.previous_response_id:
                            self.previous_response_id = response.previous_response_id
                        elif hasattr(response, 'id') and response.id:
                            self.previous_response_id = response.id

                        # ذخیره previous_response_id در فایل JSON
                        if self.previous_response_id:
                            self.save_previous_response_id(self.previous_response_id)
                
                # بررسی خطا در پاسخ
                if hasattr(response, 'error') and response.error:
                    print(f"❌ OpenAI API error (attempt {attempt + 1}): {response.error}")
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                        continue
                    return None
                
                # تبدیل پاسخ به فرمت dict مشابه send_request
                # Responses API از output_text یا output استفاده می‌کند نه choices
                content = None
                
                # روش 1: استفاده از output_text property (راحت‌تر)
                if hasattr(response, 'output_text'):
                    try:
                        content = response.output_text
                    except Exception:
                        pass
                
                # روش 2: استخراج از output list
                if not content and hasattr(response, 'output') and response.output:
                    # پیدا کردن اولین output_text در لیست output
                    for output_item in response.output:
                        if hasattr(output_item, 'type') and output_item.type == "output_text":
                            if hasattr(output_item, 'text'):
                                content = output_item.text
                                break
                            elif hasattr(output_item, 'content'):
                                # اگر content یک لیست است، آن را ترکیب می‌کنیم
                                if isinstance(output_item.content, list):
                                    content = "".join([str(item) for item in output_item.content])
                                else:
                                    content = str(output_item.content)
                                break
                
                if content:
                    result = {
                        "choices": [{
                            "message": {
                                "content": content,
                                "role": "assistant"
                            }
                        }],
                        "id": getattr(response, 'id', None),
                        "previous_response_id": getattr(response, 'previous_response_id', self.previous_response_id)
                    }
                    return result
                else:
                    print(f"⚠️ Empty response from OpenAI (attempt {attempt + 1})")
                    print(f"Response structure: {type(response)}")
                    print(f"Response attributes: {dir(response)}")
                    if hasattr(response, 'output'):
                        print(f"Output items: {len(response.output) if response.output else 0}")
                    if hasattr(response, 'incomplete_details') and response.incomplete_details:
                        print(f"Incomplete details: {response.incomplete_details}")
                    
            except Exception as e:
                print(f"API connection error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
        
        return None

    def parse_ai_response(self, response: Dict[str, Any]) -> dict:
        
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        # 1) اول سعی می‌کنیم بلوک ```json ... ``` را پیدا کنیم
        m = re.search(r"```json(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
        if m:
            json_str = m.group(1).strip()
        else:
            # اگر کد بلاک نبود، از اولین { تا آخرین } رو می‌گیریم
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object found in the given text.")
            json_str = content[start:end+1]

        # Fix trailing commas in JSON objects (invalid in standard JSON)
        json_str = re.sub(r',\s*}', '}', json_str)

        data = json.loads(json_str)

        # اگر ساختار به صورت {"recommendations": {...}} است:
        rec = data.get("recommendations", data)

        # 2) اسکلت استانداردی که می‌خوای (اعداد مهم نیستن، فقط فرمت)
        template = {
            "recommendations": {
                "crossover_methods": {
                    "part1": "",
                    "part2": "",
                    "part3_p1": "",
                    "part3_p2": "",
                    "part4": "",
                    "part5": "",
                    "part6": "",
                    "part7": "",
                },
                "global_crossover_probability": 0.0,
                "crossover_part_probability": {
                    "part1": 0.0,
                    "part2": 0.0,
                    "part3_p1": 0.0,
                    "part3_p2": 0.0,
                    "part4": 0.0,
                    "part5": 0.0,
                    "part6": 0.0,
                    "part7": 0.0,
                },
                "mutation_methods": {
                    "part1": "",
                    "part2": "",
                    "part3_p1": "",
                    "part3_p2": "",
                    "part4": "",
                    "part5": "",
                    "part6": "",
                    "part7": "",
                },
                "global_mutation_probability": 0.0,
                "mutation_args_per_part": {
                    "part1": {"mutation_rate": 0.05},
                    "part2": {"mutation_rate": 0.15, "sigma": 0.2},
                    "part3_p1": {"n_swaps": 2},
                    "part3_p2": {"n_swaps": 1},
                    "part4": {"mutation_rate": 0.08, "eta": 20},
                    "part5": {"mutation_rate": 0.08},
                    "part6": {"mutation_rate": 0.12},
                    "part7": {"mutation_rate": 0.12},
                },
                "mutation_part_probability": {
                    "part1": 0.0,
                    "part2": 0.0,
                    "part3_p1": 0.0,
                    "part3_p2": 0.0,
                    "part4": 0.0,
                    "part5": 0.0,
                    "part6": 0.0,
                    "part7": 0.0,
                },
                "selection_method": "",
                "selection_args": {
                    "k": 2,
                    "power": 1.0,
                    "refs": "[[float, float, float],[float, float, float],...]",
                    "tau": 1.0,
                    "prefer_younger": True,
                    "eps": 0.0,
                }
            }
        }

        result = deepcopy(template)
        tmpl_rec = result["recommendations"]

        # Helper برای پر کردن دیکشنری‌های ساده‌ی partها
        def fill_part_dict(tmpl_dict, src_dict):
            for k in tmpl_dict.keys():
                if isinstance(tmpl_dict[k], dict):
                    # برای mutation_args_per_part که خودش dict است
                    if k in src_dict and isinstance(src_dict[k], dict):
                        # اگه خواستی عددها رو نگه داری:
                        tmpl_dict[k].update(src_dict[k])
                else:
                    if k in src_dict:
                        tmpl_dict[k] = src_dict[k]

        # 3) پر کردن از روی JSON مدل (اگه مقادیر موجود باشن)
        if "crossover_methods" in rec:
            fill_part_dict(tmpl_rec["crossover_methods"], rec["crossover_methods"])

        if "global_crossover_probability" in rec:
            tmpl_rec["global_crossover_probability"] = rec["global_crossover_probability"]

        if "crossover_part_probability" in rec:
            fill_part_dict(tmpl_rec["crossover_part_probability"], rec["crossover_part_probability"])

        if "mutation_methods" in rec:
            fill_part_dict(tmpl_rec["mutation_methods"], rec["mutation_methods"])

        if "global_mutation_probability" in rec:
            tmpl_rec["global_mutation_probability"] = rec["global_mutation_probability"]

        if "mutation_args_per_part" in rec:
            fill_part_dict(tmpl_rec["mutation_args_per_part"], rec["mutation_args_per_part"])

        if "mutation_part_probability" in rec:
            fill_part_dict(tmpl_rec["mutation_part_probability"], rec["mutation_part_probability"])

        if "selection_method" in rec:
            tmpl_rec["selection_method"] = rec["selection_method"]

        if "selection_args" in rec and isinstance(rec["selection_args"], dict):
            tmpl_rec["selection_args"].update(rec["selection_args"])

        return result

    def get_recommendations(
        self,
        **prompt_args
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
        if self.request_type == 'openai':
            # استفاده از OpenAI
            if not self.is_gpt_model or self.previous_response_id is None:
                # اولین درخواست یا مدل غیر-GPT: ارسال prompt کامل
                prompt = self.create_prompt(**prompt_args)
                response = self.send_request_openai(prompt=prompt, store=self.store)
            else:
                # درخواست‌های بعدی برای مدل‌های GPT: فقط ارسال metrics data
                metrics_data = self.extract_metrics_data(**prompt_args)
                response = self.send_request_openai(metrics_data=metrics_data, store=self.store)
        else:
            # استفاده از requests (متد قدیمی)
            prompt = self.create_prompt(**prompt_args)
            response = self.send_request(prompt)

        if response is None:
            return None

        # پارس کردن پاسخ
        recommendations = self.parse_ai_response(response)
        return recommendations

    def validate_recommendations(self, recommendations: Dict[str, Any]) -> bool:
        """
        اعتبارسنجی پیشنهادات AI مطابق با فرمت خروجی create_prompt

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
                "default",
                "one_point_crossover",
                "two_point_crossover",
                "uniform_crossover",
                "order_crossover",
                "arithmetic_crossover",
                "blend_crossover",
                "one_point_crossover_matrix",
                "two_point_crossover_matrix",
                "uniform_crossover_matrix",
                "block_crossover_matrix",
                "row_wise_crossover_matrix",
                "arithmetic_crossover_matrix",
                "multi_point_crossover",
                "partially_mapped_crossover",
                "simulated_binary_crossover",
            ]

            valid_mutation = [
                "default",
                "bit_flip_mutation",
                "swap_mutation",
                "inversion_mutation",
                "scramble_mutation",
                "insertion_mutation",
                "displacement_mutation",
                "gaussian_mutation",
                "uniform_mutation",
                "polynomial_mutation",
                "boundary_mutation",
                "random_element_mutation_matrix",
                "gaussian_mutation_matrix",
                "row_mutation_matrix",
                "column_mutation_matrix",
                "block_mutation_matrix",
                "creep_mutation_matrix",
            ]

            valid_selection = [
                "default",
                "crowded_binary_tournament",
                "adaptive_k_tournament",
                "rank_based_roulette",
                "reference_biased_tournament",
                "age_diversity_tournament",
                "epsilon_dominance_tournament",
            ]

            # بررسی crossover_methods
            if "crossover_methods" in rec:
                crossover_methods = rec["crossover_methods"]
                # بررسی متدهای پیشنهادی برای هر بخش
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
                    if part in crossover_methods:
                        if crossover_methods[part] not in valid_crossover:
                            print(
                                f"⚠️ Invalid crossover method for {part}: {crossover_methods[part]}"
                            )
                            return False

                # بررسی global_crossover_probability
                if "global_crossover_probability" in crossover_methods:
                    prob = crossover_methods["global_crossover_probability"]
                    if not isinstance(prob, (int, float)) or not (0 <= prob <= 1):
                        print(f"⚠️ Invalid global_crossover_probability: {prob}")
                        return False

                # بررسی crossover_part_probability
                if "crossover_part_probability" in crossover_methods:
                    pp = crossover_methods["crossover_part_probability"]
                    if not isinstance(pp, dict):
                        print("⚠️ Invalid crossover_part_probability: not a dict")
                        return False
                    for k, v in pp.items():
                        if v is not None and (not isinstance(v, (int, float)) or not (0 <= v <= 1)):
                            print(f"⚠️ Invalid crossover_part_probability for {k}: {v}")
                            return False

            # بررسی mutation_methods
            if "mutation_methods" in rec:
                mutation_methods = rec["mutation_methods"]
                # بررسی متدهای پیشنهادی برای هر بخش
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
                    if part in mutation_methods:
                        if mutation_methods[part] not in valid_mutation:
                            print(
                                f"⚠️ Invalid mutation method for {part}: {mutation_methods[part]}"
                            )
                            return False

                # بررسی global_mutation_probability
                if "global_mutation_probability" in mutation_methods:
                    prob = mutation_methods["global_mutation_probability"]
                    if not isinstance(prob, (int, float)) or not (0 <= prob <= 1):
                        print(f"⚠️ Invalid global_mutation_probability: {prob}")
                        return False
                # بررسی mutation_part_probability
                if "mutation_part_probability" in mutation_methods:
                    pp = mutation_methods["mutation_part_probability"]
                    if not isinstance(pp, dict):
                        print("⚠️ Invalid mutation_part_probability: not a dict")
                        return False
                    for k, v in pp.items():
                        if v is not None and (not isinstance(v, (int, float)) or not (0 <= v <= 1)):
                            print(f"⚠️ Invalid mutation_part_probability for {k}: {v}")
                            return False

            # بررسی selection_method
            if "selection_method" in rec:
                if rec["selection_method"] not in valid_selection:
                    print(f"⚠️ Invalid selection method: {rec['selection_method']}")
                    return False
                
            return True

        except Exception as e:
            print(f"❌ Error validating recommendations: {e}")
            return False
