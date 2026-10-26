import numpy as np
import math as m
import pandas as pd



import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 * X.flatten() + 5 + np.random.randn(100) * 2
df = pd.DataFrame({"Feature": X.flatten(), "Target": y})
model = LinearRegression()
model.fit(df[["Feature"]], df["Target"])
print(f"Coefficient: {model.coef_[0]:.2f}, R²: {model.score(df[['Feature']], df['Target']):.3f}")


# This was a comment added by me. Frank Gao
