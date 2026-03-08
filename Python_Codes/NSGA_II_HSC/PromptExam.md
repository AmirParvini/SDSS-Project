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
<Context>
You are assisting an ongoing research project that applies NSGA-II to a multi-objective humanitarian logistics optimization problem.
The problem decisions are encoded in a 7-part chromosome representing distribution center assignments, shelter flows, damage-point assignments, and transport/triage matrices for severe and minor casualties.
The optimization aims to minimize three objectives: F1 (Total Distance), F2 (Unmet Demand), F3 (Death Probability).
The run configuration and problem scale (population size, number of generations, numbers of facilities/nodes) are provided below.
<context>

<Role>
You are an expert advisor (algorithm designer / adaptive operator selector) for multi-objective evolutionary algorithms, specialized in NSGA-II and adaptive operator control.
Your job is to analyze periodic performance and section-wise diversity diagnostics and recommend **per-section** crossover, mutation, and a selection method — plus tuned probabilities — for the next 10 generations.
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
When reasoning about which operator to prefer for a part, use the part data type: discrete/permutation (parts 1 & Part 3 section 1), continuous (Part 2), matrix-continuous (parts 4–7), and the special two-section layout of Part 3.
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
parts:
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
- one_point_crossover_list    (compatible only for part1, part2, section2 of part3)
- two_point_crossover_list    (compatible only for part1, part2, section2 of part3)
- multi_point_crossover_list  (compatible only for part1, part2, section2 of part3)
- uniform_crossover_list      (compatible only for part1, part2, section2 of part3)
- order_crossover_list        (compatible only for section1 of part3)
- partially_mapped_crossover  (compatible only for section1 of part3)
- arithmetic_crossover_list   (compatible only for part2)
- blend_crossover_list        (compatible only for part2)
- one_point_crossover_matrix  (compatible only for parts 4, 5, 6, 7)
- two_point_crossover_matrix  (compatible only for parts 4, 5, 6, 7)
- uniform_crossover_matrix    (compatible only for parts 4, 5, 6, 7)
- block_crossover_matrix      (compatible only for parts 4, 5, 6, 7)
- row_wise_crossover_matrix   (compatible only for parts 4, 5, 6, 7)
- arithmetic_crossover_matrix (compatible only for parts 4, 5, 6, 7)
- simulated_binary_crossover  (compatible only for part2)

16 different Mutation Methods:
- bit_flip_mutation_list         (compatible only for part1)
- swap_mutation_list             (compatible only for part1, section1 of part3, section2 of part3)
- inversion_mutation_list        (compatible only for part1, section1 of part3, section2 of part3)
- scramble_mutation_list         (compatible only for part1, section1 of part3, section2 of part3)
- insertion_mutation_list        (compatible only for part1, section1 of part3, section2 of part3)
- displacement_mutation_list     (compatible only for part1, section1 of part3, section2 of part3)
- gaussian_mutation_list         (compatible only for part2)
- uniform_mutation_list          (compatible only for part2)
- polynomial_mutation_list       (compatible only for part2)
- boundary_mutation_list         (compatible only for part2)
- random_element_mutation_matrix (compatible only for parts 4, 5, 6, 7)
- gaussian_mutation_matrix       (compatible only for parts 4, 5, 6, 7)
- row_mutation_matrix            (compatible only for parts 4, 5, 6, 7)
- column_mutation_matrix         (compatible only for parts 4, 5, 6, 7)
- block_mutation_matrix          (compatible only for parts 4, 5, 6, 7)
- creep_mutation_matrix          (compatible only for parts 4, 5, 6, 7)

7 different Selection Methods:
- crowded_binary_tournament
- adaptive_k_tournament
- feasibility_first_tournament
- rank_based_roulette
- reference_biased_tournament
- age_diversity_tournament
- epsilon_dominance_tournament

*Be sure to select the crossover and mutate methods for each section if they are compatible.*
<Specification>


<Performance>
what you will receive and how to use it:
Every 10 generations I will provide you a metrics payload containing:
- Global metrics per generation: Hypervolume (HV), Spacing, Spread, Number of Pareto solutions, Average crowding distance, Population diversity
- Population objective statistics: Mean, Min, and Standard deviation of objectives (F1, F2, F3) calculated from the entire population of each generation
- Section-state (for each part or section): variance (for continuous parts/matrices) or entropy/uniqueness (for discrete/permutation parts))
- Operator performance stats for the last interval (usage ratios and success rates per operator), if available
Use the variance/entropy of each chromosome segment to identify which segments are convergent or still widely dispersed:
- Low variance (continuous) or low entropy (discrete/permutation) → indicates convergence on that section → increase exploration (stronger mutation, more disruptive crossover) for that section.
- High variance/entropy → indicates wide exploration or noise → favor exploitative, smoothing operators for that section.
<Performance>

The data you need to analyze and based on that, suggest the things I wanted for the next 10 generations:
* Performance Metrics History (from generation 0 to {generation}):
* Current generation: {generation}
* Metrics History (Metric history from the first run of the algorithm to the current run):
    - pareto_front Hypervolume history: {metrics_history.get("hypervolume", 0):.6f}
    - pareto_front Spacing history: {metrics_history.get("spacing", 0):.6f}
    - pareto_front Spread history: {metrics_history.get("spread", 0):.6f}
    - Population diversity history: {metrics_history.get("diversity", 0):.6f}
    - Number of Pareto solutions history: {metrics_history.get("pareto_count", 0)}
    - pareto_front Average crowding distance history: {metrics_history.get("avg_crowding_distance", 0):.6f}
    - Elitism rate history: {metrics_history.get("elitism_rate", 0):.4f} ({metrics_history.get("elitism_rate", 0) * 100:.1f}% of population preserved as elite)
    - variance/entropy of parts history: {metrics_history.get("section_stats", {})}
* History of methods and their possibilities: {current_methods.get("history", {})}
* Population objective statistics (from entire population):
    - Mean objectives: F1={metrics_history.get("mean_objectives", [0,0,0])[0]:.6f}, F2={metrics_history.get("mean_objectives", [0,0,0])[1]:.6f}, F3={metrics_history.get("mean_objectives", [0,0,0])[2]:.6f}
    - Min objectives: F1={metrics_history.get("min_objectives", [0,0,0])[0]:.6f}, F2={metrics_history.get("min_objectives", [0,0,0])[1]:.6f}, F3={metrics_history.get("min_objectives", [0,0,0])[2]:.6f}
    - Std objectives: F1={metrics_history.get("std_objectives", [0,0,0])[0]:.6f}, F2={metrics_history.get("std_objectives", [0,0,0])[1]:.6f}, F3={metrics_history.get("std_objectives", [0,0,0])[2]:.6f}

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

# Prompt 3 ------------------------------------------------------------------------
prompt = {
            "system": f"""You are an expert consultant (adaptive operator selector) for multi-objective evolutionary algorithms specializing in NSGA-II and adaptive operator control. Your task is to analyze the periodic performance of the received metrics and detect diversity in each chromosome part and recommend crossover, mutation and their probabilities and rates and values ​​of input arguments for each part and a selection method for the next generations {llm_iter}.
            Your recommendations should be actionable (method names are extracted from the list of available methods) and justified by the diagnostics (e.g. low variance or entropy → increase exploration in that part).
            Your output MUST be only a single valid JSON object following the template.""",
            "user": f""" You are assisting an ongoing research project that applies NSGA-II to a multi-objective humanitarian logistics optimization problem.
The problem decisions are encoded in a 7-part chromosome representing distribution center assignments, shelter flows, damage-point assignments, and transport/triage matrices for severe and minor casualties.
The optimization aims to minimize three objectives: F1, F2, F3.
The run configuration and problem scale (population size, number of generations, numbers of facilities/nodes) are provided below.

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

Available Methods:
15 different Crossover Methods:
- one_point_crossover_list (compatible only for part1, part2, part3_p2)
- two_point_crossover_list (compatible only for part1, part2, part3_p2)
- multi_point_crossover_list (compatible only for part1, part2, part3_p2)
- uniform_crossover_list (compatible only for part1, part2, part3_p2)
- order_crossover_list (compatible only for part3_p1)
- partially_mapped_crossover (compatible only for part3_p1)
- arithmetic_crossover_list (compatible only for part2)
- blend_crossover_list (compatible only for part2)
- one_point_crossover_matrix (compatible only for parts 4, 5, 6, 7)
- two_point_crossover_matrix (compatible only for parts 4, 5, 6, 7)
- uniform_crossover_matrix (compatible only for parts 4, 5, 6, 7)
- block_crossover_matrix (compatible only for parts 4, 5, 6, 7)
- row_wise_crossover_matrix (compatible only for parts 4, 5, 6, 7)
- arithmetic_crossover_matrix (compatible only for parts 4, 5, 6, 7)
- simulated_binary_crossover (compatible only for part2)

16 different Mutation Methods:
- bit_flip_mutation_list(mutation_rate, value_range) (compatible only for part1) 
- swap_mutation_list(n_swaps) (compatible only for part1, part3_p1, part3_p2) 
- inversion_mutation_list(no arguments) (compatible only for part1, part3_p1, part3_p2)
- scramble_mutation_list(no arguments) (compatible only for part1, part3_p1, part3_p2) 
- insertion_mutation_list(no arguments) (compatible only for part1, part3_p1, part3_p2) 
- displacement_mutation_list(no arguments) (compatible only for part1, part3_p1, part3_p2) 
- gaussian_mutation_list(mutation_rate, sigma) (compatible only for part2) 
- uniform_mutation_list(mutation_rate) (compatible only for part2) 
- polynomial_mutation_list(mutation_rate, eta) (compatible only for part2) 
- boundary_mutation_list(mutation_rate) (compatible only for part2) 
- random_element_mutation_matrix(mutation_rate, value_range) (compatible only for parts 4, 5, 6, 7) 
- gaussian_mutation_matrix(mutation_rate, sigma) (compatible only for parts 4, 5, 6, 7) 
- row_mutation_matrix(mutation_rate) (compatible only for parts 4, 5, 6, 7) 
- column_mutation_matrix(mutation_rate) (compatible only for parts 4, 5, 6, 7) 
- block_mutation_matrix(block_size(tuple/list of length 2)) (compatible only for parts 4, 5, 6, 7) 
- creep_mutation_matrix(mutation_rate, step_size) (compatible only for parts 4, 5, 6, 7) 

6 different Selection Methods:
- crowded_binary_tournament(pop, k)
- adaptive_k_tournament(pop, k)
- rank_based_roulette(pop, power)
- reference_biased_tournament(pop, refs=refs, k=k, tau=tau)
- age_diversity_tournament(pop, k=k, prefer_younger=prefer_younger)
- epsilon_dominance_tournament(pop, k=k, eps=eps)

Note: *Be sure to select the crossover and mutate methods for each section if they are compatible.

what you will receive and how to use it:
Every {llm_iter} generations I will provide you a metrics payload containing:
- Hostory of the names of the methods used in each generation
- History of input argument values ​​for crossover, mutation, and selection methods
- History of the global probability of crossover and mutation methods in each generation
- History of the probability of crossover and mutation of each part of the chromosome in each generation
- History of the global metrics per generation: Hypervolume (HV), Spacing, Spread, Number of Pareto solutions, Average crowding distance
- Population objective statistics: Mean, Min, and Standard deviation of objectives (F1, F2, F3) calculated from the entire population of each generation
- Section-state (For each part of the chromosome at the gene level): variance (for part 2,4,5,6,7) or entropy/uniqueness (part 1,3)
- The number of offspring produced from the current generation that remain in the new population.
Use the variance/entropy of each chromosome segment to identify which segments are convergent or still widely dispersed:
- Low variance (continuous) or low entropy (discrete/permutation) → indicates convergence on that section → increase exploration (stronger mutation, more disruptive crossover) for that section.
- High variance/entropy → indicates wide exploration or noise → favor exploitative, smoothing operators for that section.

Note: *The key values ​​are stored in the method history as a list, and the value of each element is used for {llm_iter} generations in order, but is written only once to avoid prompt overhead.
For example, the value of the key 'global_crossover_probability_history' stored as [0.8, 0.7] means that the value 0.7 is used for generations 0 to 4 and the value 0.8 is used for generations 5 to 9.

The data you need to analyze and based on that, suggest the things I wanted for the next {llm_iter} generations:
* Performance Metrics History (from generation 0 to {generation}):
* Current generation: {generation}
* Metrics History (Metric history from the first run of the algorithm to the current run):
    - pareto_front Hypervolume history: {metrics_history.get("hypervolume", [])}
    - pareto_front Spacing history: {metrics_history.get("spacing", [])}
    - pareto_front Spread history: {metrics_history.get("spread", [])}
    - Number of Pareto solutions history: {metrics_history.get("pareto_count", [])}
    - pareto_front Average crowding distance history: {metrics_history.get("avg_crowding_distance", [])}
    - offspring survival history: {offspring_survival_history}
    - variance/entropy of parts history (The values ​​for each part are stored as tuples. for exam -> "entropy":(variance_per_gene(list), avg_variance(float))): {section_stats_history}
* History of methods and their possibilities: {current_methods.get("history", {})}
* Population objective statistics (from entire population):
    - Mean objectives: F1={mean_f1:.6f}, F2={mean_f2:.6f}, F3={mean_f3:.6f}
    - Min objectives: F1={min_f1:.6f}, F2={min_f2:.6f}, F3={min_f3:.6f}
    - Std objectives: F1={std_f1:.6f}, F2={std_f2:.6f}, F3={std_f3:.6f}

Notes: 
*The default methods Selection, Crossover, and Mutation are the methods that are used by "default" for the algorithm in the early generations.
*If the algorithm performs well using these default methods, you can suggest them (i.e. use the default instead of the method name). No data from the input arguments of these methods will be sent to you.
*Review and analyze the progress of the data provided to you and use it in your suggestions.
*Argue for yourself why you chose the methods, their probabilities, rates, and input argument values.
*When specifying the values ​​of the input arguments of the methods, be careful to only assign values ​​to the input arguments of the methods you propose and display them in the output.

Provide your response in the following JSON template (Try not to use capital letters in the keys of this output):
```json
{{
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
            }}
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
                "part1": "mutation_method_name",
            "part2": "mutation_method_name",
            "part3_p1": "mutation_method_name",
            "part3_p2": "mutation_method_name",
            "part4": "mutation_method_name",
            "part5": "mutation_method_name",
            "part6": "mutation_method_name",
            "part7": "mutation_method_name",
            }}
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
        "selection_method": "selection_method_name",
        "selection_args": {{
            "k": 2,
            "power": 1.0,
            "refs": "[[...], ...]",
            "tau": 1.0,
            "prefer_younger": true,
            "eps": 0.0
        }}
    }}
}}
```
"""
        }

# Prompt 4 ------------------------------------------------------------------------
prompt = {
            "system": 
                f"""
                You are an adaptive operator selector for NSGA-II. You must output ONLY a single valid JSON object.
The problem decisions are encoded in a 7-part chromosome representing distribution center assignments, shelter flows, damage-point assignments, and transport/triage matrices for severe and minor casualties.
The optimization aims to minimize three objectives: F1, F2, F3.
The run configuration and problem scale (population size, number of generations, numbers of facilities/nodes) are provided below.

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

STRICT RULES (the model MUST obey them):
1. For each chromosome part, ONLY select methods from the compatible list below.
2. NEVER use mutation methods inside crossover fields, and NEVER use crossover methods inside mutation fields.
3. If a method is not listed as compatible with a part, it MUST NOT be used.

Instruction:
Using the supplied problem information, method catalogs, and periodic metrics, produce an adaptive operator plan for the next {llm_iter} generations.
Your response must be a single JSON object (see the "Output Example" below).
Focus on:
- Per-part crossover method
- Per-part mutation method
- Global crossover probability and mutation probability
- Per-gene (inner) mutation rate if applicable
- A single selection method
- Elitism rate (elitism_rate): percentage of best individuals to preserve (0.05 to 0.3, typically 0.1-0.2)
- Per-part probabilities (optional): crossover_part_probability, mutation_part_probability
Be concise but include a technical justification (1–3 sentences) for each major choice.
Be sure to select the crossover and mutate methods for each section if they are compatible.

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

COMPATIBILITY CONSTRAINTS:
crossover_methods:
        "part1": {"one_point_crossover", "two_point_crossover", "multi_point_crossover", "uniform_crossover"},
        "part2": {"one_point_crossover", "two_point_crossover", "multi_point_crossover", "uniform_crossover",
                      "arithmetic_crossover", "blend_crossover", "simulated_binary_crossover"},
        "part3_p1": {"order_crossover", "partially_mapped_crossover"},
        "part3_p2": {"one_point_crossover", "two_point_crossover", "multi_point_crossover", "uniform_crossover"},
        "part4", "part5", "part6", "part7":{"one_point_crossover_matrix", "two_point_crossover_matrix", "uniform_crossover_matrix",
                      "block_crossover_matrix", "row_wise_crossover_matrix", "arithmetic_crossover_matrix"}
mutation_methods:
        "part1": {"bit_flip_mutation", "swap_mutation", "inversion_mutation", "scramble_mutation", "insertion_mutation", "displacement_mutation"},
        "part2": {"gaussian_mutation", "uniform_mutation", "polynomial_mutation", "boundary_mutation"},
        "part3_p1", "part3_p2":  {"swap_mutation", "inversion_mutation", "scramble_mutation", "insertion_mutation", "displacement_mutation"},
        "part4", "part5", "part6", "part7": {"random_element_mutation_matrix", "gaussian_mutation_matrix", "row_mutation_matrix",
                      "column_mutation_matrix", "block_mutation_matrix", "creep_mutation_matrix"}
selection_methods: {"crowded_binary_tournament", "adaptive_k_tournament","rank_based_roulette", "reference_biased_tournament",
"age_diversity_tournament", "epsilon_dominance_tournament"}

Your task:
-analyze the periodic performance of the received metrics and detect diversity in each chromosome part and recommend crossover, mutation and their probabilities and rates and values ​​of input arguments for each part and a selection method for the next generations {llm_iter}.
-Your recommendations should be actionable (method names are extracted from the list of available methods) and justified by the diagnostics (e.g. low variance or entropy → increase exploration in that part).
-Please write the names of the proposed methods exactly as they are written.

Your output MUST be valid JSON. Do NOT include explanations outside of the JSON.""",

            "user": 
                f""" You are assisting an ongoing research project that applies NSGA-II to a multi-objective humanitarian logistics optimization problem.

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

The data you need to analyze and based on that, suggest the things I wanted for the next {llm_iter} generations:
* Performance Metrics History (from generation 0 to {generation}):
* Current generation: {generation}
* Metrics History (Metric history from the first run of the algorithm to the current run):
    - pareto_front Hypervolume history: {metrics_history.get("hypervolume", [])}
    - pareto_front Spacing history: {metrics_history.get("spacing", [])}
    - pareto_front Spread history: {metrics_history.get("spread", [])}
    - Number of Pareto solutions history: {metrics_history.get("pareto_count", [])}
    - pareto_front Average crowding distance history: {metrics_history.get("avg_crowding_distance", [])}
    - offspring survival history: {offspring_survival_history}
    - Elitism rate history: {metrics_history.get("elitism_rate", [])}
    - variance/entropy of parts history (The values ​​for each part are stored as tuples. for exam -> "entropy":(variance_per_gene(list), avg_variance(float))): {section_stats_history}
* History of methods and their possibilities: {current_methods.get("history", {})}
* Population objective statistics (from entire population):
    - Mean objectives: F1={mean_f1:.6f}, F2={mean_f2:.6f}, F3={mean_f3:.6f}
    - Min objectives: F1={min_f1:.6f}, F2={min_f2:.6f}, F3={min_f3:.6f}
    - Std objectives: F1={std_f1:.6f}, F2={std_f2:.6f}, F3={std_f3:.6f}

Note: 
*The default methods Selection, Crossover, and Mutation are the methods that are used by "default" for the algorithm in the early generations.
*If the algorithm performs well using these default methods, you can suggest them (i.e. use the default instead of the method name). No data from the input arguments of these methods will be sent to you.
*Review and analyze the progress of the data provided to you and use it in your suggestions.
*Argue for yourself why you chose the methods, their probabilities, rates, and input argument values.
*When specifying the values ​​of the input arguments of the methods, be careful to only assign values ​​to the input arguments of the methods you propose and display them in the output.

From now on, I will only send you the history of the parameters without any additional writing, and you will return your suggestions by analyzing them carefully, completely in accordance with the output format.

Provide your response in the following JSON template (Try not to use capital letters in the keys of this output):
```json
{{
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
            }}
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
                "part1": "mutation_method_name",
            "part2": "mutation_method_name",
            "part3_p1": "mutation_method_name",
            "part3_p2": "mutation_method_name",
            "part4": "mutation_method_name",
            "part5": "mutation_method_name",
            "part6": "mutation_method_name",
            "part7": "mutation_method_name",
            }}
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
        "selection_method": "selection_method_name",
        "selection_args": {{
            "k": 2,
            "power": 1.0,
            "refs": "[[...], ...]",
            "tau": 1.0,
            "prefer_younger": true,
            "eps": 0.0
        }},
        "elitism_rate": 0.1
    }}
}}
```
"""
        }

# Prompt 5 ------------------------------------------------------------------------
prompt = {
            "system": f"""
You are an expert consultant (adaptive operator selector) for multi-objective evolutionary algorithms specializing in NSGA-II and adaptive operator control. Your task is to analyze the periodic performance of the received metrics and detect diversity in each chromosome part and recommend crossover, mutation and their probabilities and rates and values ​​of input arguments for each part and a selection method for the next generations {llm_iter}.
Your recommendations should be actionable (method names are extracted from the list of *Available Methods*) and justified by the diagnostics (e.g. low variance or entropy → increase exploration in that part).

AVALIBLE METHODS:
"crossover_methods":[
- one_point_crossover (Use only in 'part1'&'part2'&'part3_p2')
- two_point_crossover (Use only in 'part1'&'part2'&'part3_p2')
- multi_point_crossover (Use only in 'part1'&'part2'&'part3_p2')
- uniform_crossover (Use only in 'part1'&'part2'&'part3_p2')
- order_crossover (Use only in 'part3_p1')
- partially_mapped_crossover (Use only in 'part3_p1')
- arithmetic_crossover (Use only in 'part2')
- blend_crossover (Use only in 'part2')
- one_point_crossover_matrix (Use only in 'part4'&'part5'&'part6'&'part7')
- two_point_crossover_matrix (Use only in 'part4'&'part5'&'part6'&'part7')
- uniform_crossover_matrix (Use only in 'part4'&'part5'&'part6'&'part7')
- block_crossover_matrix (Use only in 'part4'&'part5'&'part6'&'part7')
- row_wise_crossover_matrix (Use only in 'part4'&'part5'&'part6'&'part7')
- arithmetic_crossover_matrix (Use only in 'part4'&'part5'&'part6'&'part7')
- simulated_binary_crossover (Use only in 'part2')
]
"mutation_methods":[
- bit_flip_mutation(mutation_rate, value_range) (Use only in 'part1') 
- swap_mutation(n_swaps) (Use only in 'part1'&'part3_p1'&'part3_p2') 
- inversion_mutation(no arguments) (Use only in 'part1'&'part3_p1'&'part3_p2')
- scramble_mutation(no arguments) (Use only in 'part1'&'part3_p1'&'part3_p2') 
- insertion_mutation(no arguments) (Use only in 'part1'&'part3_p1'&'part3_p2') 
- displacement_mutation(no arguments) (Use only in 'part1'&'part3_p1'&'part3_p2') 
- gaussian_mutation(mutation_rate, sigma) (Use only in 'part2') 
- uniform_mutation(mutation_rate) (Use only in 'part2') 
- polynomial_mutation(mutation_rate, eta) (Use only in 'part2') 
- boundary_mutation(mutation_rate) (Use only in 'part2') 
- random_element_mutation_matrix(mutation_rate, value_range) (Use only in 'part4'&'part5'&'part6'&'part7') 
- gaussian_mutation_matrix(mutation_rate, sigma) (Use only in 'part4'&'part5'&'part6'&'part7') 
- row_mutation_matrix(mutation_rate) (Use only in 'part4'&'part5'&'part6'&'part7') 
- column_mutation_matrix(mutation_rate) (Use only in 'part4'&'part5'&'part6'&'part7') 
- block_mutation_matrix(block_size(tuple/list of length 2)) (Use only in 'part4'&'part5'&'part6'&'part7') 
- creep_mutation_matrix(mutation_rate, step_size) (Use only in 'part4'&'part5'&'part6'&'part7') 
]
"selection_methods":[
- crowded_binary_tournament(pop, k)
- adaptive_k_tournament(pop, k)
- rank_based_roulette(pop, power)
- reference_biased_tournament(pop, refs=refs, k=k, tau=tau)
- age_diversity_tournament(pop, k=k, prefer_younger=prefer_younger)
- epsilon_dominance_tournament(pop, k=k, eps=eps)
]

STRICT ENFORCEMENT RULES:
1.Use the mentioned methods only in the mentioned parts.
2.Avoid changing or removing part of the method names in AVAILABLE METHODS. Make sure the method names you use for the parts exactly match AVAILABLE METHODS.
3.Avoid changing the characters in output template keys.
4.Be careful not to leave out the keys used in the output template.
5.not to use capital letters in the keys of JSON output
6.Your output MUST be only a single valid JSON object following the template. 

Provide your response in the following JSON template:
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
            }}
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
            }}
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
        }},
        "elitism_rate": 0.1
    }}
}}
```
""",
            "user": f""" 
You are assisting an ongoing research project that applies NSGA-II to a multi-objective humanitarian logistics optimization problem.
The problem decisions are encoded in a 7-part chromosome representing distribution center assignments, shelter flows, damage-point assignments, and transport/triage matrices for severe and minor casualties.
The optimization aims to minimize three objectives: F1, F2, F3.
The run configuration and problem scale (population size, number of generations, numbers of facilities/nodes) are provided below.

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

Instruction:
Using the supplied problem information, method catalogs, and periodic metrics, produce an adaptive operator plan for the next {llm_iter} generations.
Your response must be a single JSON object (see the "Output Example" below).
Focus on:
- Per-part crossover method
- Per-part mutation method
- Global crossover probability and mutation probability
- Per-gene (inner) mutation rate if applicable
- A single selection method
- Elitism rate (elitism_rate): percentage of best individuals to preserve (0.05 to 0.3, typically 0.1-0.2)
- Per-part probabilities (optional): crossover_part_probability, mutation_part_probability
Be concise but include a technical justification (1–3 sentences) for each major choice.

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

The data you need to analyze and based on that, suggest the things I wanted for the next {llm_iter} generations:
* Performance Metrics History (from generation 0 to {generation}):
* Current generation: {generation}
* Metrics History (Metric history from the first run of the algorithm to the current run):
    - pareto_front Hypervolume history: {metrics_history.get("hypervolume", [])}
    - pareto_front Spacing history: {metrics_history.get("spacing", [])}
    - pareto_front Spread history: {metrics_history.get("spread", [])}
    - Number of Pareto solutions history: {metrics_history.get("pareto_count", [])}
    - pareto_front Average crowding distance history: {metrics_history.get("avg_crowding_distance", [])}
    - offspring survival history: {offspring_survival_history}
    - Elitism rate history: {metrics_history.get("elitism_rate", [])}
    - variance/entropy of parts history (The values ​​for each part are stored as tuples. for exam -> "entropy":(variance_per_gene(list), avg_variance(float))): {section_stats_history}
* History of methods and their possibilities: {current_methods.get("history", {})}
* Population objective statistics (from entire population):
    - Mean objectives: F1={mean_f1:.6f}, F2={mean_f2:.6f}, F3={mean_f3:.6f}
    - Min objectives: F1={min_f1:.6f}, F2={min_f2:.6f}, F3={min_f3:.6f}
    - Std objectives: F1={std_f1:.6f}, F2={std_f2:.6f}, F3={std_f3:.6f}

Note: 

*The default methods Selection, Crossover, and Mutation are the methods that are used by "default" for the algorithm in the early generations.
*If the algorithm performs well using these default methods, you can suggest them (i.e. use the default instead of the method name). No data from the input arguments of these methods will be sent to you.
*Review and analyze the progress of the data provided to you and use it in your suggestions.
*Argue for yourself why you chose the methods, their probabilities, rates, and input argument values.
*When specifying the values ​​of the input arguments of the methods, be careful to only assign values ​​to the input arguments of the methods you propose and display them in the output.

From now on, I will only send you the history of the parameters without any additional writing, and you will return your suggestions by analyzing them carefully, completely in accordance with the output format.

"""
        }