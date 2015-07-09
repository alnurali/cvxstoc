from coopr.pyomo import *

# Helper functions
def obj_rule(model):
    return model.FirstStageCost + model.SecondStageCost

def ComputeFirstStageCost_rule(model):
    return (model.FirstStageCost - (model.b*model.x)) == 0

def ComputeSecondStageCost_rule(model):
    return (model.SecondStageCost - (-model.s*model.y1 - model.r*model.y2)) == 0

def constr1_rule(model):
    return (model.y1+model.y2) <= model.x

def constr2_rule(model, i):
    return 0 <= model.y1 and model.y1 <= model.d[i]

# Initialize problem data
model = AbstractModel()    

model.b = Param()

model.s = Param()
model.r = Param()

model.scens = Set()
model.d = Param(model.scens)

model.u = Param()

# Setup all stage problems
model.x = Var(bounds=(0.0, model.u))
model.y1 = Var()
model.y2 = Var(within=NonNegativeReals)

model.obj = Objective(rule=obj_rule, sense=minimize)
model.FirstStageCost = Var()
model.SecondStageCost = Var()
model.ComputeFirstStageCost = Constraint(rule=ComputeFirstStageCost_rule)
model.ComputeSecondStageCost = Constraint(rule=ComputeSecondStageCost_rule)

model.constr1 = Constraint(rule=constr1_rule)
model.constr2 = Constraint(model.scens, rule=constr2_rule)

"""
To solve this optimiation problem (from the shell), execute:

either
runef --solver-manager=neos --solver=cbc --solve

or
runef --solver=glpk --solve
(after, of course, installin glpk first).
"""
