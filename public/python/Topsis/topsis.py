import json
import sys

import numpy as np


def get_best_solution_id(ids, X, weights):
    # تبدیل به آرایه عددی برای محاسبات
    X_arr = np.array(X, dtype=float)
    weights = np.array(weights, dtype=float)

    # مراحل TOPSIS
    denom = np.sqrt((X_arr**2).sum(axis=0))
    R = X_arr / denom
    V = R * weights

    # برای مدل شما (همه cost هستند)
    A_plus = np.min(V, axis=0)  
    A_minus = np.max(V, axis=0) 

    D_plus = np.sqrt(((V - A_plus)**2).sum(axis=1))
    D_minus = np.sqrt(((V - A_minus)**2).sum(axis=1))

    CC = D_minus / (D_plus + D_minus)

    # پیدا کردن ایندکس بهترین جواب
    best_index = np.argmax(CC)
    
    # برگرداندن آیدی متناظر با آن ایندکس
    best_id = ids[best_index]
    
    return best_id


if __name__ == "__main__":
    # Entry point invoked by App\Services\TopsisService::run().
    # A single JSON object is written to stdin with the shape:
    #   {
    #     "ids":     [solution_id, ...],          // one per row
    #     "X":       [[z1, z2, z3], ...],         // objective matrix, aligned with "ids"
    #     "weights": [w1, w2, w3]                 // one weight per objective/column
    #   }
    try:
        payload = json.load(sys.stdin)

        best_id = get_best_solution_id(
            payload["ids"],
            payload["X"],
            payload["weights"],
        )

        # The id may be a numpy scalar; cast back to a plain Python type so
        # json.dumps can serialize it and PHP receives a plain int/string.
        if isinstance(best_id, np.generic):
            best_id = best_id.item()

        # Only the best solution id is expected back by TopsisService::run().
        print(json.dumps({"best_solution_id": best_id}))
    except Exception as exc:
        # Surface a readable error on stdout (exit code stays 0) so
        # TopsisService::run() can decode it and raise a meaningful
        # message instead of a bare traceback.
        print(json.dumps({"error": str(exc)}))
