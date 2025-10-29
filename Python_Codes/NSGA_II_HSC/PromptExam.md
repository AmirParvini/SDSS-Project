# prompt 1 ------------------------------------------------------------------------
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

# Prompt 2 ------------------------------------------------------------------------
Context:
You are assisting an ongoing research project that applies NSGA-II to a multi-objective humanitarian logistics optimization problem.
The problem decisions are encoded in a 7-part chromosome representing distribution center assignments, shelter flows, damage-point assignments, and transport/triage matrices for severe and minor casualties.
The optimization aims to minimize three objectives: F1 (Total Distance), F2 (Unmet Demand), F3 (Death Probability).
The run configuration and problem scale (population size, number of generations, numbers of facilities/nodes) are provided below.

Role:
You are an expert advisor (algorithm designer / adaptive operator selector) for multi-objective evolutionary algorithms, specialized in NSGA-II and adaptive operator control.
Your job is to analyze periodic performance and section-wise diversity diagnostics and recommend **per-section** crossover, mutation, and a selection method — plus tuned probabilities — for the next 5 generations.
Your recommendations should be actionable (method names are drawn from the available-method lists) and justified by the diagnostics (e.g., low variance or entropy → increase exploration on that section).

Instruction:
Using the supplied problem information, method catalogs, and periodic metrics, produce an adaptive operator plan for the next five generations.
Your response must be a single JSON object (see the "Output Example" below).
Focus on:
- Per-part crossover method
- Per-part mutation method
- Global crossover probability and mutation probability
- Per-gene (inner) mutation rate if applicable
- A single selection method
- Short analysis text (why you chose these operators, and what triggers should change them in the future)
Be concise but include a technical justification (1–3 sentences) for each major choice.
When reasoning about which operator to prefer for a part, use the part data type: discrete/permutation (Parts 1 & Part 3 section 1), continuous (Part 2), matrix-continuous (Parts 4–7), and the special two-section layout of Part 3.

Specification (problem + chromosome + available methods):
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
4. Part 4: Severe injured to hospital assignment (float matrix) — rows=damage points, cols=hospitals (values in {0} or [0,1])
5. Part 5: Percentage of severe injured transported by ambulance (float matrix) — same dims as Part 4 (values in [0,1])
6. Part 6: Minor injured to hospitals + temporary centers assignment (float matrix)
7. Part 7: Percentage of minor injured transported by ambulance (float matrix)

Available Methods:
Crossover (15):
- one_point_crossover_list    (Suitable only for Part 1, Part 2, Part 3 Section 2)
- two_point_crossover_list    (Suitable only for Part 1, Part 2, Part 3 Section 2)
- multi_point_crossover_list  (Suitable only for Part 1, Part 2, Part 3 Section 2)
- uniform_crossover_list      (Suitable only for Part 1, Part 2, Part 3 Section 2)
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

Mutation (16):
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

Selection (7):
- crowded_binary_tournament
- adaptive_k_tournament
- feasibility_first_tournament
- rank_based_roulette
- reference_biased_tournament
- age_diversity_tournament
- epsilon_dominance_tournament

Performance (what you will receive and how to use it):
Every 5 generations I will provide you a metrics payload containing:
- Global metrics per generation: Hypervolume (HV), Spacing, Spread, Number of Pareto solutions, Average crowding distance, Population diversity
- Section-wise diagnostics (for each part or section): variance (for continuous parts/matrices) or entropy/uniqueness (for discrete/permutation parts), and repair_rate (fraction of individuals repaired)
- Operator performance stats for the last interval (usage ratios and success rates per operator), if available
Use section-wise variance/entropy to detect which parts are converging or still widely spread:
- Low variance (continuous) or low entropy (discrete/permutation) → indicates convergence on that section → increase exploration (stronger mutation, more disruptive crossover) for that section.
- High variance/entropy → indicates wide exploration or noise → favor exploitative, smoothing operators for that section.

Input Format (what you will receive every 5 gens):
A JSON-like object containing:
- generation (int)
- metrics: { "history": [...], "hypervolume": float, "spacing": float, "spread": float, "pareto_count": int, "avg_crowding_distance": float, "diversity": float }
- section_stats: { "part1": {"entropy": float, "repair_rate": float}, "part2": {"variance": float}, "part3_p1": {"entropy": float}, "part3_p2": {"entropy": float, "repair_rate": float}, "part4": {"variance": float}, ... }
- current_methods: { "crossover_probability": float, "mutation_probability": float, "mutation_inner_rate": float, "selection_method": str, "crossover_methods": {...}, "mutation_methods": {...} }

Example (Performance & Input Example):
Current generation: {generation}
Metrics History: {metrics_history}

Latest Generation Metrics (example fields):
- Hypervolume (HV): {metrics.get("hypervolume", 0):.6f}
- Spacing: {metrics.get("spacing", 0):.6f}
- Spread: {metrics.get("spread", 0):.6f}
- Number of Pareto solutions: {metrics.get("pareto_count", 0)}
- Average crowding distance: {metrics.get("avg_crowding_distance", 0):.6f}
- Population diversity: {metrics.get("diversity", 0):.6f}

Why section statistics (entropy/variance) matter:
- Entropy (for discrete & permutation parts) measures how many distinct allele values exist and how balanced their frequencies are. Low entropy → most individuals share the same allele(s) → risk of premature convergence on that decision subspace.
- Variance (for continuous parts/matrices) measures spread of values. Low variance → convergence; high variance → noisy exploration.
- Repair rate highlights constraint pressure: high repair usage suggests operators generate many infeasible solutions — prefer conservative operators or stronger repair after variation.

Example Output (the exact JSON shape you MUST return):
Return a single JSON object with keys: "analysis" and "recommendations".
The "recommendations" object must contain "crossover", "mutation", and "selection" subobjects. Each crossover/mutation subobject must list method for part1, part2, part3_p1, part3_p2, part4, part5, part6, part7, plus global probabilities and inner rates as shown below.

Output Example:
{
    "analysis": "Brief analysis of current situation and recommendations",
    "recommendations": {
        "crossover": {
            "part1": "one_point_crossover_list",
            "part2": "simulated_binary_crossover",
            "part3_p1": "order_crossover_list",
            "part3_p2": "uniform_crossover_list",
            "part4": "row_wise_crossover_matrix",
            "part5": "one_point_crossover_matrix",
            "part6": "block_crossover_matrix",
            "part7": "uniform_crossover_matrix",
            "probability": 0.85
        },
        "mutation": {
            "part1": "swap_mutation_list",
            "part2": "polynomial_mutation_list",
            "part3_p1": "inversion_mutation_list",
            "part3_p2": "bit_flip_mutation_list",
            "part4": "gaussian_mutation_matrix",
            "part5": "creep_mutation_matrix",
            "part6": "random_element_mutation_matrix",
            "part7": "row_mutation_matrix",
            "probability": 0.15,
            "mutation_rate": 0.08
        },
        "selection": {
            "method": "feasibility_first_tournament"
        }
    }
}

Example (concrete minimal input) you might receive from the caller:
{
  "generation": 20,
  "metrics": {
    "history": [
      {"generation": 0, "hypervolume": 0.00, "spacing": 0.5, "spread": 0.3, "pareto_count": 2, "avg_crowding_distance": 0.1, "diversity": 0.9},
      ...
    ],
    "hypervolume": 0.1245,
    "spacing": 0.0123,
    "spread": 0.0456,
    "pareto_count": 28,
    "avg_crowding_distance": 0.0234,
    "diversity": 0.34
  },
  "section_stats": {
    "part1": {"entropy": 0.45, "repair_rate": 0.12},
    "part2": {"variance": 0.0012},
    "part3_p1": {"entropy": 0.10},
    "part3_p2": {"entropy": 0.6, "repair_rate": 0.05},
    "part4": {"variance": 0.002},
    "part5": {"variance": 0.0021},
    "part6": {"variance": 0.05},
    "part7": {"variance": 0.0015}
  },
  "current_methods": {
    "crossover_probability": 0.7,
    "mutation_probability": 0.3,
    "mutation_inner_rate": 0.1,
    "selection_method": "crowded_binary_tournament",
    "crossover_methods": {"part1": "one_point_crossover_list", ...},
    "mutation_methods": {"part1": "bit_flip_mutation_list", ...}
  }
}

Behavioral rules & constraints:
- If any discrete/permutation part shows very low entropy (e.g., entropy < 0.2) recommend stronger permutation-preserving exploration (e.g., partially_mapped_crossover + swap/inversion mutations) for that section.
- If any continuous/matrix part has extremely low variance (e.g., variance < 1e-4) recommend aggressive continuous mutations (polynomial_mutation_list or gaussian_mutation_matrix) and higher crossover disruption for that section.
- If repair_rate is high (> 0.2) on any section, favor conservative operators for that section and recommend improving repair or constraint handling (repair heuristics or feasibility-first selection).
- Prefer feasibility_first_tournament or epsilon_dominance_tournament when feasibility ratio is low or constraints are tight.
- Prefer age_diversity_tournament or reference_biased_tournament to maintain diversity if diversity metric drops below 0.25.

Performance:
Return recommendations focused on generations {generation + 1} to {generation + 5}.
If you detect a clear trend of premature convergence (hypervolume stagnation for the last 3 records + diversity drop), recommend a short adaptive schedule: temporarily increase mutation probability by +0.15 and/or change per-section operators to more disruptive ones for 5 generations, then re-evaluate.
If operator success_rate data is provided, prefer operators with statistically higher success_rate for exploitation; otherwise rely on diagnostics above.

Example Response (must follow exactly the JSON shape shown earlier):
- Provide "analysis" with a short diagnosis (2–4 sentences)
- Provide "recommendations" with "crossover", "mutation" and "selection" blocks (methods must be one of the available methods)
- Provide numeric probabilities rounded to two decimal places (e.g., 0.85, 0.12)

Now: analyze the incoming JSON payload I provided and return the JSON recommendations object.