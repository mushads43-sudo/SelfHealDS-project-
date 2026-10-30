import pandas as pd
from .rules import POLICY

def quality_score(df, issues):
    if df.empty: return 100.0
    weights={"Error":POLICY["quality_error_weight"],"Warning":POLICY["quality_warning_weight"],"Info":POLICY["quality_info_weight"]}
    cells=max(1,df.shape[0]*df.shape[1])
    penalty=sum(weights.get(i.severity,.1) for i in issues)
    return round(max(0,min(100,100*(1-penalty/cells))),2)

def quality_summary(df,issues):
    return {"score":quality_score(df,issues),"issues":len(issues),
        "errors":sum(i.severity=="Error" for i in issues),"warnings":sum(i.severity=="Warning" for i in issues),
        "info":sum(i.severity=="Info" for i in issues),"repairable":sum(i.repairable for i in issues),
        "error_cells":len({(i.row,i.column) for i in issues if i.severity=="Error"})}

def quality_dimensions(df,issues):
    dimensions={}
    for i in issues:
        dimensions.setdefault(i.category,{"findings":0,"errors":0,"warnings":0})
        dimensions[i.category]["findings"]+=1
        dimensions[i.category]["errors"]+=int(i.severity=="Error")
        dimensions[i.category]["warnings"]+=int(i.severity=="Warning")
    return dimensions
