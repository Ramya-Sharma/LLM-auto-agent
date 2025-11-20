# # # app/agents/synthetic_agent.py
# # import pandas as pd
# # import numpy as np
# # from pathlib import Path
# # from faker import Faker
# # from app.storage import ensure_dirs

# # fake = Faker()
# # ensure_dirs()

# # def synthesize_dataset(run_id, schema_hint):
# #     rows = int(schema_hint.get("rows", 1000))
# #     columns = schema_hint.get("columns", [])
# #     data = {}
# #     for col in columns:
# #         name = col.get("name","col")
# #         ctype = col.get("type","float")
# #         if ctype.startswith("int"):
# #             low,high = col.get("range",[0,100])
# #             data[name] = np.random.randint(low, high+1, size=rows)
# #         elif ctype == "float":
# #             low,high = col.get("range",[0.0,1.0])
# #             data[name] = np.random.uniform(low, high, size=rows)
# #         elif ctype == "binary":
# #             p = float(col.get("imbalance_ratio", 0.2))
# #             data[name] = np.random.choice([0,1], size=rows, p=[1-p,p])
# #         else:
# #             data[name] = [fake.word() for _ in range(rows)]
# #     df = pd.DataFrame(data)
# #     Path("data").mkdir(exist_ok=True)
# #     path = f"data/{run_id}_synthetic.csv"
# #     df.to_csv(path, index=False)
# #     return path




# # app/agents/synthetic_agent.py

# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from pathlib import Path
# from faker import Faker
# from app.storage import ensure_dirs

# fake = Faker()
# ensure_dirs()


# def _generate_datetime_series(rows, start_year=2015, end_year=2024):
#     """Generate realistic datetime values."""
#     dates = []
#     for _ in range(rows):
#         year = np.random.randint(start_year, end_year+1)
#         month = np.random.randint(1, 13)
#         day = np.random.randint(1, 28)
#         hour = np.random.randint(0, 24)
#         minute = np.random.randint(0, 60)
#         sec = np.random.randint(0, 60)
#         dates.append(datetime(year, month, day, hour, minute, sec))
#     return dates


# def _generate_categorical_low_card(rows, values):
#     """Generate categorical column with low-cardinality labels."""
#     return np.random.choice(values, size=rows)


# def synthesize_dataset(run_id, schema_hint):
#     """
#     Generates a synthetic dataset aligned with LLM-based schema hint.
#     Supports:
#     - int, float, binary
#     - categorical low-cardinality
#     - text columns
#     - datetime columns
#     - target column generation
#     """

#     rows = int(schema_hint.get("rows", 1000))
#     columns = schema_hint.get("columns", [])

#     data = {}

#     for col in columns:
#         name = col.get("name", "col")
#         ctype = col.get("type", "float")

#         # -----------------
#         # INTEGER COLUMN
#         # -----------------
#         if ctype.startswith("int"):
#             low, high = col.get("range", [0, 100])
#             data[name] = np.random.randint(low, high+1, size=rows)

#         # -----------------
#         # FLOAT COLUMN
#         # -----------------
#         elif ctype == "float":
#             low, high = col.get("range", [0.0, 1.0])
#             data[name] = np.random.uniform(low, high, size=rows)

#         # -----------------
#         # BINARY / TARGET
#         # -----------------
#         elif ctype == "binary":
#             p = float(col.get("imbalance_ratio", 0.2))
#             data[name] = np.random.choice([0, 1], size=rows, p=[1-p, p])

#         # -----------------
#         # CATEGORICAL (LLM-driven)
#         # -----------------
#         elif ctype == "categorical":
#             values = col.get("values", ["A", "B", "C"])
#             data[name] = _generate_categorical_low_card(rows, values)

#         # -----------------
#         # DATETIME COLUMN
#         # -----------------
#         elif ctype == "datetime":
#             data[name] = _generate_datetime_series(rows)

#         # -----------------
#         # TEXT COLUMN
#         # -----------------
#         elif ctype == "text":
#             data[name] = [fake.sentence(nb_words=col.get("words", 5)) for _ in range(rows)]

#         # -----------------
#         # FALLBACK → string
#         # -----------------
#         else:
#             data[name] = [fake.word() for _ in range(rows)]

#     # Convert to dataframe
#     df = pd.DataFrame(data)

#     # Save dataset
#     Path("data").mkdir(exist_ok=True)
#     path = f"data/{run_id}_synthetic.csv"
#     df.to_csv(path, index=False)

#     return path

# app/agents/synthetic_agent.py
import numpy as np
import pandas as pd
from pathlib import Path
from faker import Faker
fake = Faker()
Path("data").mkdir(exist_ok=True)

def synthesize_dataset(run_id, schema_hint):
    rows = int(schema_hint.get("rows", 1000))
    columns = schema_hint.get("columns", [])
    data = {}
    for col in columns:
        name = col.get("name","col")
        ctype = col.get("type","float")
        if ctype.startswith("int"):
            low,high = col.get("range",[0,100])
            data[name] = np.random.randint(low, high+1, size=rows)
        elif ctype == "float":
            low,high = col.get("range",[0.0,1.0])
            data[name] = np.random.uniform(low, high, size=rows)
        elif ctype == "binary":
            p = float(col.get("imbalance_ratio", 0.2))
            data[name] = np.random.choice([0,1], size=rows, p=[1-p,p])
        else:
            data[name] = [fake.word() for _ in range(rows)]
    df = pd.DataFrame(data)
    path = f"data/{run_id}_synthetic.csv"
    df.to_csv(path, index=False)
    return path
