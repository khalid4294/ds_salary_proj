import scraping as gs
import pandas as pd
path = "/Users/khalidalolayan/Desktop/ds_salary_project/chromedriver"
df = gs.get_jobs('marketing manager', 15, False, path, 15)