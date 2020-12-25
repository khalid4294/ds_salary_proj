import scraping as gs
import pandas as pd
path = "/Users/khalidalolayan/Desktop/document/ds_salary_proj/chromedriver"
df = gs.get_jobs('marketing manager', 15, False, path, 15)