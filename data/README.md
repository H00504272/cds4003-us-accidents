# Data

The full file is not on GitHub because it is 189 MB and GitHub allows 100 MB per file.

- **Source:** US Accidents (March 2023 version), Moosavi et al. Download it from
  https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
- **Our file:** a 500,000-row sample of that dataset, called
  `US_Accidents_March23_sampled_500k.csv`. Put it in this folder before you run the notebooks.
- **`sample_accidents.csv`:** 1,000 rows picked at random from our file, so you can see the columns
  without downloading anything.
- **Licence:** CC BY-NC-SA 4.0 (non-commercial, research and study use, share alike).

If you only have the full Kaggle file, you can make a similar sample (the rows will be
different, so the numbers will change a little):

```python
import pandas as pd
df = pd.read_csv("US_Accidents_March23.csv")
df.sample(500_000, random_state=42).to_csv("US_Accidents_March23_sampled_500k.csv", index=False)
```
